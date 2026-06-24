"""
src/data/hf_loader.py
─────────────────────
Downloads angkul07/abc-ego from HuggingFace Hub (MCAP files),
reads them with the mcap library, and converts to the internal
trajectory Episode schema.

The dataset stores each episode as:
  data/train/<task>/episode_<uuid>/episode.mcap

MCAP is a container (like a ROS2 bag) holding *messages* on *channels*
(topics).  We discover all channel schemas during schema-discovery and
then extract observations / actions / timestamps accordingly.

Two-phase design
────────────────
Phase 1 – bulk_download()
    Downloads up to `max_episodes` MCAP files in parallel (thread pool)
    and writes a resume manifest (download_manifest.json) so a crashed
    session can be continued without re-downloading.

Phase 2 – iter_episodes() / load_episode()
    Reads already-cached MCAPs from disk.  Zero network I/O.
    Can be run independently after Phase 1 completes (even in a new
    SSH session).

Usage
─────
    loader = HFLoader(task="place_the_bread", max_episodes=200)
    loader.bulk_download()          # Phase 1  – run once, survives SSH drops
    for ep in loader.iter_episodes():   # Phase 2  – pure disk reads
        process(ep)
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Internal trajectory schema
# ─────────────────────────────────────────────────────────────

@dataclass
class Observation:
    image: Optional[np.ndarray] = None        # (H,W,C) uint8
    robot_state: Optional[np.ndarray] = None  # (D,) float32
    depth: Optional[np.ndarray] = None        # (H,W) float32


@dataclass
class Episode:
    episode_id: str
    observations: List[Observation] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    def validate(self) -> None:
        """Raise ValueError on integrity failures."""
        if len(self.timestamps) == 0:
            raise ValueError(f"[{self.episode_id}] Empty episode – no timesteps.")
        n = len(self.timestamps)
        if len(self.observations) != n:
            raise ValueError(
                f"[{self.episode_id}] Observation count ({len(self.observations)}) "
                f"!= timestamp count ({n})."
            )
        if len(self.actions) != n:
            raise ValueError(
                f"[{self.episode_id}] Action count ({len(self.actions)}) "
                f"!= timestamp count ({n})."
            )
        if n < 2:
            raise ValueError(f"[{self.episode_id}] Episode too short ({n} steps).")


# ─────────────────────────────────────────────────────────────
# MCAP helpers
# ─────────────────────────────────────────────────────────────

def _open_mcap_reader(path: Path):
    """Return a decoded mcap reader using protobuf DecoderFactory."""
    from mcap.reader import make_reader
    try:
        from mcap_protobuf.decoder import DecoderFactory
        decoder_factories = [DecoderFactory()]
    except ImportError:
        log.warning("mcap-protobuf-support not installed — falling back to raw bytes")
        decoder_factories = []
    fh = open(path, "rb")
    return make_reader(fh, decoder_factories=decoder_factories), fh


def _parse_messages(reader) -> Tuple[List[Dict], List[str]]:
    raw_records: List[Dict] = []
    topics: set = set()

    for schema, channel, message, decoded in reader.iter_decoded_messages():
        topics.add(channel.topic)
        raw_records.append({
            "timestamp_ns": message.log_time,
            "topic": channel.topic,
            "schema": schema.name if schema else "",
            "decoded": decoded,
        })

    return sorted(raw_records, key=lambda r: r["timestamp_ns"]), sorted(topics)


# ─────────────────────────────────────────────────────────────
# Protobuf field extraction
# ─────────────────────────────────────────────────────────────

def _extract_robot_state_array(msg) -> Optional[np.ndarray]:
    parts = []
    if hasattr(msg, 'position') and len(msg.position) > 0:
        parts.extend(msg.position)
    if hasattr(msg, 'pose') and len(msg.pose) > 0:
        parts.extend(msg.pose)
    return np.array(parts, dtype=np.float32) if parts else None


def _extract_gripper_state_array(msg) -> Optional[np.ndarray]:
    if hasattr(msg, 'position') and len(msg.position) > 0:
        return np.array(list(msg.position), dtype=np.float32)
    return None


def _extract_video_frame(msg) -> Optional[np.ndarray]:
    if not hasattr(msg, 'data') or not msg.data:
        return None
    raw = bytes(msg.data)
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw))
        return np.array(img)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Topic classifier
# ─────────────────────────────────────────────────────────────

_IMAGE_HINTS = ("image", "rgb", "color", "camera", "video", "compressed")
_DEPTH_HINTS = ("depth",)
_STATE_HINTS = ("state", "joint", "pose", "odom", "robot", "arm", "gripper", "ee_pose")
_ACTION_HINTS = ("action", "cmd", "command", "control", "target")


def _classify_topic(topic: str, schema: str) -> str:
    t = topic.lower()
    s = schema.lower()
    for h in _IMAGE_HINTS:
        if h in t or h in s:
            return "image"
    for h in _DEPTH_HINTS:
        if h in t:
            return "depth"
    for h in _ACTION_HINTS:
        if h in t or h in s:
            return "action"
    for h in _STATE_HINTS:
        if h in t or h in s:
            return "state"
    return "unknown"


# ─────────────────────────────────────────────────────────────
# Episode builder
# ─────────────────────────────────────────────────────────────

def _build_episode(episode_id: str, raw_records: List[Dict]) -> Episode:
    if not raw_records:
        raise ValueError(f"[{episode_id}] No messages decoded from MCAP.")

    topic_class: Dict[str, str] = {}
    for rec in raw_records:
        t, s = rec["topic"], rec["schema"]
        if t not in topic_class:
            topic_class[t] = _classify_topic(t, s)

    BUCKET_NS = 50_000_000  # 50 ms
    t_start = raw_records[0]["timestamp_ns"]
    buckets: Dict[int, List[Dict]] = {}
    for rec in raw_records:
        b = (rec["timestamp_ns"] - t_start) // BUCKET_NS
        buckets.setdefault(b, []).append(rec)

    timestamps: List[float] = []
    observations: List[Observation] = []
    actions: List[np.ndarray] = []

    for b_key in sorted(buckets.keys()):
        bucket = buckets[b_key]
        ts_s = (t_start + b_key * BUCKET_NS) / 1e9
        timestamps.append(ts_s)

        images, depths, states, acts = [], [], [], []
        for rec in bucket:
            cls = topic_class.get(rec["topic"], "unknown")
            decoded = rec["decoded"]
            schema_name = rec["schema"]

            if cls == "image":
                if schema_name in ("foxglove.CompressedVideo", "foxglove.CompressedImage"):
                    arr = _extract_video_frame(decoded)
                    if arr is not None:
                        images.append(arr)
            elif cls in ("state", "action"):
                if schema_name == "RobotState":
                    arr = _extract_robot_state_array(decoded)
                elif schema_name == "GripperState":
                    arr = _extract_gripper_state_array(decoded)
                else:
                    arr = None
                if arr is not None:
                    (states if cls == "state" else acts).append(arr)

        if not acts:
            timestamps.pop()
            continue

        observations.append(Observation(
            image=images[0] if images else None,
            robot_state=np.concatenate(states).astype(np.float32) if states else None,
            depth=depths[0] if depths else None,
        ))
        actions.append(np.concatenate(acts).astype(np.float32))

    return Episode(
        episode_id=episode_id,
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        metadata={"topic_classes": topic_class, "n_buckets": len(buckets)},
    )


# ─────────────────────────────────────────────────────────────
# Schema Discovery
# ─────────────────────────────────────────────────────────────

def discover_schema(mcap_path: Path) -> Dict[str, Any]:
    reader, fh = _open_mcap_reader(mcap_path)
    topics_info: Dict[str, Dict] = {}

    for schema, channel, message in reader.iter_messages():
        t = channel.topic
        if t not in topics_info:
            topics_info[t] = {
                "schema": schema.name if schema else "unknown",
                "class": _classify_topic(t, schema.name if schema else ""),
                "message_count": 0,
                "first_ts_ns": message.log_time,
                "last_ts_ns": message.log_time,
            }
        topics_info[t]["message_count"] += 1
        topics_info[t]["last_ts_ns"] = message.log_time

    fh.close()
    return {"topics": topics_info, "source_file": str(mcap_path)}


# ─────────────────────────────────────────────────────────────
# HFLoader  (two-phase: bulk_download → iter_episodes)
# ─────────────────────────────────────────────────────────────

class HFLoader:
    """
    Phase 1 — bulk_download()
        Downloads up to `max_episodes` MCAPs in parallel using a thread
        pool.  Writes a JSON manifest so interrupted runs resume from
        where they left off without re-downloading already-cached files.

    Phase 2 — iter_episodes() / load_episode()
        Reads cached MCAPs from disk; no network I/O.  Safe to run in a
        fresh SSH session after Phase 1 completes.

    Parameters
    ----------
    repo_id      : HuggingFace dataset repo, e.g. "angkul07/abc-ego"
    task         : Task subfolder, e.g. "place_the_bread"
    cache_dir    : Root directory for cached MCAP files
    max_episodes : Hard cap on number of episodes to download/process
    n_workers    : Parallel download threads (default 8)
    """

    REPO_TYPE = "dataset"
    MANIFEST_NAME = "download_manifest.json"

    def __init__(
        self,
        repo_id: str = "angkul07/abc-ego",
        # task: str = "place_the_bread",
        task: str = "put_the_screwdriver_in_the_bin",
        cache_dir: str = "./cache",
        max_episodes: Optional[int] = None,
        n_workers: int = 8,
    ):
        self.repo_id = repo_id
        self.task = task
        self.cache_dir = Path(cache_dir) / task
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self.n_workers = n_workers
        self._episode_files: Optional[List[str]] = None
        self._manifest_path = self.cache_dir / self.MANIFEST_NAME
        self._manifest_lock = threading.Lock()

    # ── Manifest (resume support) ─────────────────────────────

    def _load_manifest(self) -> Dict[str, str]:
        """Returns {repo_path: local_path_str} for already-downloaded episodes."""
        if not self._manifest_path.exists():
            return {}
        with open(self._manifest_path) as f:
            return json.load(f)

    def _update_manifest(self, repo_path: str, local_path: Path) -> None:
        """Thread-safe manifest update after each successful download."""
        with self._manifest_lock:
            manifest = self._load_manifest()
            manifest[repo_path] = str(local_path)
            with open(self._manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    # ── File listing ──────────────────────────────────────────

    def list_episode_files(self) -> List[str]:
        """List all episode MCAP paths for the given task in the HF repo."""
        if self._episode_files is not None:
            return self._episode_files

        prefix = f"data/train/{self.task}/"
        log.info(f"Listing repo files for task '{self.task}' …")
        all_files = list(list_repo_files(self.repo_id, repo_type=self.REPO_TYPE))
        episode_files = [
            f for f in all_files
            if f.startswith(prefix) and f.endswith("episode.mcap")
        ]
        log.info(f"  Found {len(episode_files)} episode files for task '{self.task}'.")
        self._episode_files = episode_files
        return episode_files

    # ── Phase 1: Bulk download ────────────────────────────────

    def bulk_download(self) -> List[Path]:
        """
        Download up to `max_episodes` MCAPs in parallel.

        Already-cached files (tracked in download_manifest.json) are
        skipped so re-running after a crash continues from where it
        stopped.

        Returns a list of local Paths for all downloaded episodes.
        """
        files = self.list_episode_files()
        if self.max_episodes is not None:
            files = files[: self.max_episodes]

        manifest = self._load_manifest()
        pending = [f for f in files if f not in manifest]
        already_done = len(files) - len(pending)

        log.info(
            f"Bulk download: {len(files)} total  |  "
            f"{already_done} cached  |  {len(pending)} to download"
        )

        if not pending:
            log.info("All episodes already cached — skipping download.")
            return [Path(manifest[f]) for f in files if f in manifest]

        completed: List[Path] = [Path(manifest[f]) for f in files if f in manifest]
        failed: List[str] = []
        done_count = already_done

        def _download_one(repo_path: str) -> Tuple[str, Optional[Path]]:
            """Download a single MCAP; returns (repo_path, local_path | None)."""
            parts = repo_path.split("/")
            episode_dir_name = parts[-2]
            local_dir = self.cache_dir / episode_dir_name
            local_path = local_dir / "episode.mcap"

            if local_path.exists():
                return repo_path, local_path

            local_dir.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_path,
                    repo_type=self.REPO_TYPE,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
                # Locate the downloaded file (hf_hub_download nesting varies)
                if not local_path.exists():
                    found = list(local_dir.rglob("episode.mcap"))
                    if found:
                        found[0].rename(local_path)
                return repo_path, local_path if local_path.exists() else None
            except Exception as exc:
                log.warning(f"  FAILED {repo_path}: {exc}")
                return repo_path, None

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(_download_one, rp): rp for rp in pending}
            for future in as_completed(futures):
                repo_path, local_path = future.result()
                done_count += 1
                if local_path is not None:
                    completed.append(local_path)
                    self._update_manifest(repo_path, local_path)
                    log.info(f"  [{done_count}/{len(files)}] OK  {repo_path}")
                else:
                    failed.append(repo_path)
                    log.warning(f"  [{done_count}/{len(files)}] FAIL {repo_path}")

        log.info(
            f"Download complete: {len(completed)} succeeded, {len(failed)} failed."
        )
        if failed:
            log.warning(f"  Failed paths: {failed}")

        return completed

    # ── Phase 2: Parse from disk ──────────────────────────────

    def _local_path_for(self, repo_path: str) -> Optional[Path]:
        """Return the local cache path for a repo_path, or None if not cached."""
        parts = repo_path.split("/")
        episode_dir_name = parts[-2]
        local_path = self.cache_dir / episode_dir_name / "episode.mcap"
        return local_path if local_path.exists() else None

    def load_episode(self, repo_path: str) -> Episode:
        """
        Parse one episode from the local cache.
        Raises FileNotFoundError if the MCAP has not been downloaded yet.
        """
        local_path = self._local_path_for(repo_path)
        if local_path is None:
            raise FileNotFoundError(
                f"Episode not cached: {repo_path}\n"
                f"Run loader.bulk_download() first."
            )

        parts = repo_path.split("/")
        episode_dir = parts[-2]
        episode_id = (
            episode_dir[len("episode_"):]
            if episode_dir.startswith("episode_")
            else episode_dir
        )

        reader, fh = _open_mcap_reader(local_path)
        raw_records, topics = _parse_messages(reader)
        fh.close()

        episode = _build_episode(episode_id, raw_records)
        episode.metadata["source_file"] = str(local_path)
        episode.metadata["topics"] = topics
        episode.metadata["task"] = self.task
        return episode

    def iter_episodes(self, validate: bool = True) -> Iterator[Episode]:
        """
        Yield Episode objects for all cached episodes.

        This is pure disk I/O — no network calls.  If an episode has not
        been downloaded yet it is logged as a warning and skipped.
        """
        files = self.list_episode_files()
        if self.max_episodes is not None:
            files = files[: self.max_episodes]

        # Only iterate what we actually have on disk
        manifest = self._load_manifest()
        available = [f for f in files if f in manifest or self._local_path_for(f) is not None]
        missing = len(files) - len(available)
        if missing:
            log.warning(
                f"{missing} episodes not yet downloaded — run bulk_download() first."
            )

        errors: List[str] = []
        for i, fpath in enumerate(available):
            log.info(f"  [{i+1}/{len(available)}] parsing {fpath}")
            try:
                ep = self.load_episode(fpath)
                if validate:
                    ep.validate()
                yield ep
            except Exception as exc:
                msg = f"SKIP [{fpath}]: {exc}"
                log.warning(msg)
                errors.append(msg)

        if errors:
            log.warning(f"{len(errors)} episodes skipped due to parse/validation errors.")

    # ── Schema discovery ──────────────────────────────────────

    def discover_one_schema(self) -> Dict[str, Any]:
        """Inspect the schema of the first cached episode."""
        files = self.list_episode_files()
        if not files:
            raise RuntimeError(f"No episode files found for task '{self.task}'.")

        # Prefer the first cached file; fall back to downloading it
        for f in files:
            local = self._local_path_for(f)
            if local is not None:
                break
        else:
            log.info("No cached episodes found; downloading first episode for schema discovery.")
            self.bulk_download_n(1)
            local = self._local_path_for(files[0])

        info = discover_schema(local)
        info["episode_id"] = files[0].split("/")[-2]
        info["task"] = self.task
        info["total_episodes"] = len(files)
        return info

    def bulk_download_n(self, n: int) -> List[Path]:
        """Convenience: download exactly n episodes (ignores self.max_episodes)."""
        orig = self.max_episodes
        self.max_episodes = n
        result = self.bulk_download()
        self.max_episodes = orig
        return result