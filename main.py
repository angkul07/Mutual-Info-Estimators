"""
main.py ── fidelity-eval pipeline CLI
═══════════════════════════════════════════════════════════════════════
Usage examples:

  python main.py download repo=angkul07/abc-ego task=place_the_bread
  python main.py inspect
  python main.py embed            (alias for 'inspect' in this version)
  python main.py compute_mi
  python main.py build_eval

Run all steps:
  python main.py download inspect compute_mi build_eval

Flags:
  --max N          process at most N episodes (for testing)
  --no-validate    skip episode integrity checks
  --seed N         random seed for trajectory sampling (default 42)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/source.yaml")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        log.warning(f"Config not found at {CONFIG_PATH}, using defaults.")
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_loader(cfg: dict, max_episodes: Optional[int] = None):
    from src.data.hf_loader import HFLoader
    src = cfg.get("source", {})
    return HFLoader(
        repo_id=src.get("repo", "angkul07/abc-ego"),
        # task=src.get("task", "place_the_bread"),
        task=src.get("task", "put_the_screwdriver_in_the_bin"),
        cache_dir=src.get("cache_dir", "./cache"),
        max_episodes=max_episodes,
        n_workers=src.get("n_workers", 8),
    )


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────

def cmd_download(args, cfg):
    """
    Step 0 — Download episodes (or verify cache).
    Supports optional partial download via --max.
    """
    header("STEP 0 — DOWNLOAD")
    loader = _make_loader(cfg, max_episodes=args.max)
    files = loader.list_episode_files()
    print(f"\n  Repo : {loader.repo_id}")
    print(f"  Task : {loader.task}")
    print(f"  Total episodes in repo: {len(files)}")
    print(f"  Will process: {args.max if args.max else 'ALL'}")
    print()

    # Trigger a download of the first file to verify connectivity
    if files:
        local_paths = loader.bulk_download()
        print(f"  ✓ {len(local_paths)} episodes cached to disk.")
    else:
        log.error("No episode files found — check repo/task name.")
        sys.exit(1)

    print("\n✓ Download step complete.\n")


def cmd_inspect(args, cfg):
    """
    Step 0.1 — Schema discovery + dataset summary report.
    """
    header("STEP 0.1 — INSPECT / SCHEMA DISCOVERY")
    from src.data.schema_inspector import run_inspection
    loader = _make_loader(cfg, max_episodes=args.max)
    schema = run_inspection(loader, output_dir="outputs")
    print("\n  Topics found:")
    for t, v in schema.get("topics", {}).items():
        print(f"    [{v['class']:10s}] {t}")
    print("\n✓ Inspection complete → see outputs/schema.json and outputs/dataset_summary.md\n")


def cmd_embed(args, cfg):
    """
    Step 0.2-0.3 — Filter task + convert to Parquet.
    'embed' here means: ingest + persist trajectories as embeddings/rows.
    """
    header("STEP 0.2-0.3 — TASK FILTER + TRAJECTORY CONVERSION → PARQUET")
    from src.data.converter import episodes_to_parquet
    loader = _make_loader(cfg, max_episodes=args.max)
    validate = not args.no_validate

    episodes = list(loader.iter_episodes(validate=validate))
    if not episodes:
        log.error("No valid episodes loaded.")
        sys.exit(1)

    out_path = cfg.get("output", {}).get("processed_parquet", "processed/episodes.parquet")
    n_rows = episodes_to_parquet(episodes, out_path=out_path)
    print(f"\n✓ Converted {len(episodes)} episodes → {n_rows} rows → {out_path}\n")
    return episodes


def cmd_compute_mi(args, cfg, episodes=None):
    """
    Step — MI estimation and trajectory ranking.
    Loads from Parquet if episodes not provided.
    """
    header("STEP — MUTUAL INFORMATION ESTIMATION")
    from src.mi_estimator import compute_and_rank

    # Load episodes
    if episodes is None:
        loader = _make_loader(cfg, max_episodes=args.max)
        validate = not args.no_validate
        episodes = list(loader.iter_episodes(validate=validate))

    if not episodes:
        log.error("No valid episodes to score.")
        sys.exit(1)

    mi_cfg = cfg.get("mi", {})
    easy, hard = compute_and_rank(
        episodes,
        k=mi_cfg.get("k_neighbors", 3),
        pca_components=mi_cfg.get("pca_components", 8),
        out_dir=".",
    )
    print(f"\n✓ MI computed for {len(episodes)} episodes")
    print(f"  Easy eval: {len(easy)} episodes → easy_eval.json")
    print(f"  Hard eval: {len(hard)} episodes → hard_eval.json\n")
    return easy, hard


def cmd_build_eval(args, cfg, episodes=None):
    """
    Step — Visual inspection + produce eval split files.
    """
    header("STEP 0.4 — VISUAL INSPECTION + BUILD EVAL")
    from src.data.visualizer import visualize_trajectories

    if episodes is None:
        loader = _make_loader(cfg, max_episodes=args.max)
        validate = not args.no_validate
        episodes = list(loader.iter_episodes(validate=validate))

    if not episodes:
        log.error("No valid episodes to visualize.")
        sys.exit(1)

    out_cfg = cfg.get("output", {})
    plots_dir = out_cfg.get("plots_dir", "plots/trajectory_preview")
    saved = visualize_trajectories(
        episodes,
        n_sample=cfg.get("processing", {}).get("n_sample_trajectories", 5),
        out_dir=plots_dir,
        seed=getattr(args, "seed", 42),
    )
    print(f"\n✓ Visual previews saved:")
    for p in saved:
        print(f"  {p}")
    print()

    # Produce eval JSONs if not already done
    easy_path = Path(out_cfg.get("easy_eval", "easy_eval.json"))
    hard_path = Path(out_cfg.get("hard_eval", "hard_eval.json"))
    if not easy_path.exists() or not hard_path.exists():
        cmd_compute_mi(args, cfg, episodes=episodes)


# ─────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────

def header(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

COMMAND_MAP = {
    "download":    cmd_download,
    "inspect":     cmd_inspect,
    "embed":       cmd_embed,
    "compute_mi":  cmd_compute_mi,
    "build_eval":  cmd_build_eval,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="fidelity-eval — HuggingFace MCAP dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "commands",
        nargs="+",
        choices=list(COMMAND_MAP.keys()) + ["all"],
        help="Pipeline step(s) to run",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: repo=... task=... (key=value pairs)",
    )
    p.add_argument("--max", type=int, default=None, help="Max number of episodes to process")
    p.add_argument("--no-validate", action="store_true", help="Skip episode integrity validation")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return p


def main():
    # Custom parsing to support both positional commands and key=value overrides
    raw = sys.argv[1:]
    commands = []
    overrides = {}
    flags = {}

    i = 0
    while i < len(raw):
        tok = raw[i]
        if tok in COMMAND_MAP or tok == "all":
            commands.append(tok)
        elif tok.startswith("--max"):
            if "=" in tok:
                flags["max"] = int(tok.split("=", 1)[1])
            else:
                i += 1
                flags["max"] = int(raw[i])
        elif tok == "--no-validate":
            flags["no_validate"] = True
        elif tok.startswith("--seed"):
            if "=" in tok:
                flags["seed"] = int(tok.split("=", 1)[1])
            else:
                i += 1
                flags["seed"] = int(raw[i])
        elif "=" in tok:
            k, v = tok.split("=", 1)
            overrides[k] = v
        i += 1

    if not commands:
        print(__doc__)
        sys.exit(0)

    if "all" in commands:
        commands = ["download", "inspect", "embed", "compute_mi", "build_eval"]

    # Build a simple namespace
    class Args:
        pass
    args = Args()
    args.max = flags.get("max", None)
    args.no_validate = flags.get("no_validate", False)
    args.seed = flags.get("seed", 42)

    # Load and patch config
    cfg = _load_config()
    for k, v in overrides.items():
        # Support top-level keys: repo, task
        if k == "repo":
            cfg.setdefault("source", {})["repo"] = v
        elif k == "task":
            cfg.setdefault("source", {})["task"] = v
        else:
            cfg[k] = v

    episodes = None
    easy = hard = None

    for cmd_name in commands:
        fn = COMMAND_MAP[cmd_name]
        # Pass episodes between steps to avoid re-downloading
        if cmd_name in ("embed",):
            episodes = fn(args, cfg)
            if episodes is None:
                # fallback: embed returns list via side effect
                episodes = episodes
        elif cmd_name in ("compute_mi",):
            easy, hard = fn(args, cfg, episodes=episodes)
        elif cmd_name in ("build_eval",):
            fn(args, cfg, episodes=episodes)
        else:
            fn(args, cfg)

    print("\n🎉  Pipeline complete.\n")


if __name__ == "__main__":
    main()
