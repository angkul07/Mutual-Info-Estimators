"""
src/data/converter.py
──────────────────────
Converts a stream of Episode objects into a flat Parquet table saved at
processed/episodes.parquet.

Schema (one row per timestep):
  episode_id   str
  step         int
  timestamp    float64
  action       list[float]
  robot_state  list[float]  (may be null)
  has_image    bool
  has_depth    bool
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.hf_loader import Episode

log = logging.getLogger(__name__)


def episode_to_rows(ep: Episode) -> List[dict]:
    """Flatten one Episode into a list of row dicts."""
    rows = []
    n = len(ep.timestamps)
    for i in range(n):
        obs = ep.observations[i]
        act = ep.actions[i]
        row = {
            "episode_id": ep.episode_id,
            "step": i,
            "timestamp": float(ep.timestamps[i]),
            "action": act.tolist() if isinstance(act, np.ndarray) else list(act),
            "robot_state": (
                obs.robot_state.tolist()
                if obs.robot_state is not None
                else None
            ),
            "has_image": obs.image is not None,
            "has_depth": obs.depth is not None,
        }
        rows.append(row)
    return rows


def episodes_to_parquet(
    episodes: Iterable[Episode],
    out_path: str = "processed/episodes.parquet",
    chunk_size: int = 500,
) -> int:
    """
    Convert an iterable of episodes to Parquet.
    Writes in chunks to keep memory low.
    Returns total number of rows written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    buffer: List[dict] = []

    def flush(buf):
        nonlocal writer, total_rows
        df = pd.DataFrame(buf)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out), table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(buf)

    for ep in episodes:
        rows = episode_to_rows(ep)
        buffer.extend(rows)
        if len(buffer) >= chunk_size:
            flush(buffer)
            buffer.clear()
            log.info(f"  Flushed chunk → {total_rows} rows total so far …")

    if buffer:
        flush(buffer)

    if writer:
        writer.close()

    log.info(f"Parquet written → {out}  ({total_rows} rows)")
    return total_rows
