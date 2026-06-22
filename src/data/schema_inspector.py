"""
src/data/schema_inspector.py
─────────────────────────────
Generates outputs/schema.json and outputs/dataset_summary.md
by inspecting a small number of MCAP episode files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────

REQUIRED_CLASSES = {"action", "state", "image"}


def _fail_if_missing(schema_info: Dict[str, Any]) -> None:
    """Raise if mandatory signal classes are absent."""
    topics = schema_info.get("topics", {})
    found_classes = {v["class"] for v in topics.values()}
    missing = REQUIRED_CLASSES - found_classes

    if "action" in missing:
        raise RuntimeError(
            "SCHEMA VALIDATION FAILED: No action topics found in the episode.\n"
            f"Topics discovered: {list(topics.keys())}\n"
            f"Classes found:     {found_classes}\n"
            "─── Aborting pipeline. ───"
        )
    if missing:
        log.warning(f"Optional signal classes missing: {missing}")


# ─────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────

def _build_report(schema_info: Dict[str, Any], episode_ids: List[str]) -> str:
    topics = schema_info.get("topics", {})
    total = schema_info.get("total_episodes", "?")
    task = schema_info.get("task", "unknown")
    source_file = schema_info.get("source_file", "?")

    image_keys   = [t for t, v in topics.items() if v["class"] == "image"]
    action_keys  = [t for t, v in topics.items() if v["class"] == "action"]
    state_keys   = [t for t, v in topics.items() if v["class"] == "state"]
    depth_keys   = [t for t, v in topics.items() if v["class"] == "depth"]
    unknown_keys = [t for t, v in topics.items() if v["class"] == "unknown"]

    lines = [
        f"# Dataset Summary — `{task}`",
        "",
        "## Source",
        f"- **Repository**: angkul07/abc-ego",
        f"- **Task**: `{task}`",
        f"- **Total episodes**: {total}",
        f"- **Format**: MCAP (ROS2 bag)",
        f"- **Inspected from**: `{source_file}`",
        "",
        "## Episode IDs (sample)",
        "",
        "| # | Episode ID |",
        "|---|------------|",
    ]
    for i, eid in enumerate(episode_ids[:20], 1):
        lines.append(f"| {i} | `{eid}` |")
    if len(episode_ids) > 20:
        lines.append(f"| … | *(+{len(episode_ids)-20} more)* |")

    lines += [
        "",
        "## Available Topics (Columns)",
        "",
        "| Topic | Schema | Class | Messages |",
        "|-------|--------|-------|----------|",
    ]
    for t, v in sorted(topics.items()):
        dur_s = (v["last_ts_ns"] - v["first_ts_ns"]) / 1e9 if v.get("last_ts_ns") else 0
        lines.append(f"| `{t}` | `{v['schema']}` | **{v['class']}** | {v['message_count']} |")

    def fmt_list(lst):
        return "\n".join(f"  - `{x}`" for x in lst) if lst else "  *(none)*"

    lines += [
        "",
        "## Key Signal Classification",
        "",
        f"### Image keys ({len(image_keys)})",
        fmt_list(image_keys),
        "",
        f"### Action keys ({len(action_keys)})",
        fmt_list(action_keys),
        "",
        f"### State keys ({len(state_keys)})",
        fmt_list(state_keys),
        "",
        f"### Depth keys ({len(depth_keys)})",
        fmt_list(depth_keys),
        "",
        f"### Unknown topics ({len(unknown_keys)})",
        fmt_list(unknown_keys),
        "",
        "## Integrity Checks",
        "- [x] actions present" if action_keys else "- [ ] **MISSING actions**",
        "- [x] robot state present" if state_keys else "- [ ] robot state absent (warn)",
        "- [x] images present" if image_keys else "- [ ] images absent (warn)",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_inspection(loader, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Use `loader` (HFLoader) to inspect the first episode and produce:
      outputs/schema.json
      outputs/dataset_summary.md

    Returns the schema_info dict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting schema discovery …")
    schema_info = loader.discover_one_schema()

    # Validate
    _fail_if_missing(schema_info)

    # Save JSON
    schema_path = out_dir / "schema.json"
    with open(schema_path, "w") as f:
        # numpy int64 → default int
        json.dump(schema_info, f, indent=2, default=str)
    log.info(f"Saved schema → {schema_path}")

    # Build episode ID list
    episode_paths = loader.list_episode_files()
    ep_ids = [p.split("/")[-2].replace("episode_", "") for p in episode_paths]

    # Render markdown
    md = _build_report(schema_info, ep_ids)
    md_path = out_dir / "dataset_summary.md"
    md_path.write_text(md)
    log.info(f"Saved summary → {md_path}")

    return schema_info
