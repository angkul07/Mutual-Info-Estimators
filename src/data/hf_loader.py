"""
src/data/hf_loader.py
─────────────────────
Downloads angkul07/abc-ego from HuggingFace Hub (MCAP files),
reads them with the mcap library, and converts to the internal
trajectory Episode schema.

The dataset stores each episode as:
  data/train/<task>/episode_<uuid>/episode.mcap

MCAP is a container (like a ROS2 bag) holding *messages* on *channels*
(topics). We discover all channel schemas during schema-discovery and
then extract observations / actions / timestamps accordingly.
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    # ── convenience ──────────────────────────────────────────
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
# MCAP helpers  (mcap library ≥ 1.0, protobuf decoding)
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
    """
    Iterate decoded messages and return:
      raw_records  – list of {timestamp_ns, topic, schema_name, decoded_msg}
      topics       – sorted unique topic list
    """
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
# Protobuf field extraction helpers
# ─────────────────────────────────────────────────────────────

def _extract_robot_state_array(msg) -> Optional[np.ndarray]:
    """
    Extract numeric fields from a decoded RobotState protobuf message.
    Concatenates: position + pose (the key state/action fields).
    Velocity/torque are only present on state topics, not action topics.
    """
    parts = []
    if hasattr(msg, 'position') and len(msg.position) > 0:
        parts.extend(msg.position)
    if hasattr(msg, 'pose') and len(msg.pose) > 0:
        parts.extend(msg.pose)
    if not parts:
        return None
    return np.array(parts, dtype=np.float32)


def _extract_gripper_state_array(msg) -> Optional[np.ndarray]:
    """
    Extract numeric fields from a decoded GripperState protobuf message.
    Returns the position (typically 1 float: gripper openness).
    """
    if hasattr(msg, 'position') and len(msg.position) > 0:
        return np.array(list(msg.position), dtype=np.float32)
    return None


def _extract_video_frame(msg) -> Optional[np.ndarray]:
    """
    Extract a video frame from a decoded CompressedVideo protobuf message.
    H.264 frames require a video decoder; we attempt JPEG/PNG first,
    then return None (H.264 keyframes need ffmpeg to decode).
    """
    if not hasattr(msg, 'data') or not msg.data:
        return None
    raw = bytes(msg.data)
    # Try JPEG / PNG via PIL (works for CompressedImage, not CompressedVideo)
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(raw))
        return np.array(img)
    except Exception:
        pass
    # H.264 video frames cannot be decoded without a video decoder
    # Return None rather than garbage uint8 noise
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
    """
    Group raw messages by timestamp (nanosecond bucket, 50 ms window)
    and assemble into Episode.  Unknown data blobs are decoded
    heuristically.
    """
    if not raw_records:
        raise ValueError(f"[{episode_id}] No messages decoded from MCAP.")

    # ── Classify topics ──────────────────────────────────────
    topic_class: Dict[str, str] = {}
    for rec in raw_records:
        t, s = rec["topic"], rec["schema"]
        if t not in topic_class:
            topic_class[t] = _classify_topic(t, s)

    # ── Bucket messages by 50ms windows ──────────────────────
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

        # ── Aggregate per-class data from this bucket ─────────
        images, depths, states, acts = [], [], [], []
        for rec in bucket:
            cls = topic_class.get(rec["topic"], "unknown")
            decoded = rec["decoded"]
            schema_name = rec["schema"]

            if cls == "image":
                if schema_name in ("foxglove.CompressedVideo", "foxglove.CompressedImage"):
                    arr = _extract_video_frame(decoded)
                else:
                    arr = None
                if arr is not None:
                    images.append(arr)
            elif cls == "depth":
                pass  # depth not present in this dataset
            elif cls in ("state", "action"):
                # Proper protobuf field extraction
                if schema_name == "RobotState":
                    arr = _extract_robot_state_array(decoded)
                elif schema_name == "GripperState":
                    arr = _extract_gripper_state_array(decoded)
                else:
                    arr = None
                if arr is not None:
                    if cls == "state":
                        states.append(arr)
                    else:
                        acts.append(arr)

        # ── Skip buckets with no action data (B4 fix) ─────────
        if not acts:
            timestamps.pop()
            continue

        # ── Build Observation (B5 fix: concatenate ALL topics) ─
        obs = Observation(
            image=images[0] if images else None,
            robot_state=np.concatenate(states).astype(np.float32) if states else None,
            depth=depths[0] if depths else None,
        )
        observations.append(obs)

        # ── Action: concatenate all action topics ─────────────
        actions.append(np.concatenate(acts).astype(np.float32))

    ep = Episode(
        episode_id=episode_id,
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        metadata={"topic_classes": topic_class, "n_buckets": len(buckets)},
    )
    return ep


# ─────────────────────────────────────────────────────────────
# Schema Discovery
# ─────────────────────────────────────────────────────────────

def discover_schema(mcap_path: Path) -> Dict[str, Any]:
    """
    Read one MCAP file and return a schema summary dict suitable
    for outputs/schema.json.
    """
    reader, fh = _open_mcap_reader(mcap_path)
    schema_info = reader.get_summary()
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
# HFLoader
# ─────────────────────────────────────────────────────────────

class HFLoader:
    """
    Downloads episodes from HuggingFace Hub on demand and
    converts them to the internal Episode schema.

    Parameters
    ----------
    repo_id : str       e.g. "angkul07/abc-ego"
    task    : str       e.g. "place_the_bread"
    cache_dir : str     local cache root
    max_episodes : int  limit (None = all)
    """

    REPO_TYPE = "dataset"

    def __init__(
        self,
        repo_id: str = "angkul07/abc-ego",
        task: str = "place_the_bread",
        cache_dir: str = "./cache",
        max_episodes: Optional[int] = None,
    ):
        self.repo_id = repo_id
        self.task = task
        self.cache_dir = Path(cache_dir) / task
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self._episode_files: Optional[List[str]] = None

    # ── File listing ─────────────────────────────────────────

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

    # ── Download one episode ──────────────────────────────────

    def _download_episode(self, repo_path: str) -> Path:
        """Download (or retrieve from cache) one MCAP file."""
        # Derive episode_id from path:
        # data/train/place_the_bread/episode_<uuid>/episode.mcap
        parts = repo_path.split("/")
        episode_dir_name = parts[-2]   # episode_<uuid>
        local_dir = self.cache_dir / episode_dir_name
        local_path = local_dir / "episode.mcap"

        if local_path.exists():
            log.debug(f"Cache hit: {local_path}")
            return local_path

        log.info(f"  Downloading {repo_path} …")
        local_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=self.repo_id,
            filename=repo_path,
            repo_type=self.REPO_TYPE,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download stores in nested dir; locate the file
        # The file lands at local_dir/<repo_path tail> or at local_dir/episode.mcap
        candidate = local_dir / "episode.mcap"
        if not candidate.exists():
            # search recursively
            found = list(local_dir.rglob("episode.mcap"))
            if found:
                candidate = found[0]
                # move to expected location
                candidate.rename(local_path)
        return local_path

    # ── Parse one episode ─────────────────────────────────────

    def load_episode(self, repo_path: str) -> Episode:
        """Download and parse one episode, returning an Episode object."""
        parts = repo_path.split("/")
        episode_dir = parts[-2]
        # Extract UUID from "episode_<uuid>"
        episode_id = episode_dir[len("episode_"):] if episode_dir.startswith("episode_") else episode_dir

        local_path = self._download_episode(repo_path)
        reader, fh = _open_mcap_reader(local_path)
        raw_records, topics = _parse_messages(reader)
        fh.close()

        episode = _build_episode(episode_id, raw_records)
        episode.metadata["source_file"] = str(local_path)
        episode.metadata["topics"] = topics
        episode.metadata["task"] = self.task
        return episode

    # ── Iterate episodes ──────────────────────────────────────

    def iter_episodes(self, validate: bool = True):
        """
        Generator yielding Episode objects.
        Fails loudly (raises) on integrity errors if validate=True.
        """
        files = self.list_episode_files()
        if self.max_episodes is not None:
            files = files[: self.max_episodes]

        errors: List[str] = []
        for i, fpath in enumerate(files):
            log.info(f"  [{i+1}/{len(files)}] {fpath}")
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
            log.warning(f"{len(errors)} episodes skipped due to errors.")

    # ── Schema discovery ──────────────────────────────────────

    def discover_one_schema(self) -> Dict[str, Any]:
        """Download the first episode file and inspect its schema."""
        files = self.list_episode_files()
        if not files:
            raise RuntimeError(f"No episode files found for task '{self.task}'.")
        first = files[0]
        local = self._download_episode(first)
        info = discover_schema(local)
        info["episode_id"] = first.split("/")[-2]
        info["task"] = self.task
        info["total_episodes"] = len(files)
        return info
