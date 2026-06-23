"""
src/mi_estimator.py
────────────────────
DemInf-style mutual information estimation between states and actions.
Uses the Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator.

Pipeline:
1. PCA compress state+action vectors
2. KSG MI estimation per episode
3. Temporal alignment score
4. Final ranking → easy_eval.json + hard_eval.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# KSG MI estimator
# ─────────────────────────────────────────────────────────────

def _ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
    """
    KSG mutual information estimator (Kraskov et al. 2004, Eq. 8).
    X, Y : (n, dx) and (n, dy) float arrays.
    Returns MI estimate in nats.
    """
    from scipy.spatial import cKDTree

    n = X.shape[0]
    if n < k + 2:
        return 0.0

    XY = np.hstack([X, Y])
    # B2 fix: KSG requires Chebyshev (L∞) metric in the joint space
    # so estimation errors from S and A marginals cancel.
    tree_xy = cKDTree(XY)
    tree_x  = cKDTree(X)
    tree_y  = cKDTree(Y)

    dists, _ = tree_xy.query(XY, k=k + 1, p=np.inf)  # Chebyshev; includes self
    eps = dists[:, -1]  # distance to k-th neighbour in joint space

    # Count marginal neighbours within the joint-space k-NN radius (using L2 per marginal).
    # Paper uses ≤ (less-than-or-equal), so we use eps[i] directly with query_ball_point
    # which returns points with distance <= r.
    nx = np.array([len(tree_x.query_ball_point(X[i], eps[i], p=2)) - 1 for i in range(n)])
    ny = np.array([len(tree_y.query_ball_point(Y[i], eps[i], p=2)) - 1 for i in range(n)])

    # B1 fix: KSG formula is ψ(k) + ψ(N) − ⟨ψ(nx+1)⟩ − ⟨ψ(ny+1)⟩
    # The +1 is the bias-correction term (Kraskov et al. 2004, Eq. 8).
    mi = (
        _digamma(k)
        + _digamma(n)
        - np.mean(_digamma(nx + 1))
        - np.mean(_digamma(ny + 1))
    )
    return float(max(mi, 0.0))


def _digamma(x):
    """Vectorised digamma via scipy or fallback."""
    try:
        from scipy.special import digamma
        return digamma(x)
    except ImportError:
        # rough approximation: ψ(n) ≈ ln(n) - 1/(2n)
        x = np.asarray(x, dtype=float)
        return np.log(np.maximum(x, 1e-10)) - 1.0 / (2.0 * np.maximum(x, 1e-10))


# ─────────────────────────────────────────────────────────────
# PCA helper
# ─────────────────────────────────────────────────────────────

def _pca_reduce(arr: np.ndarray, n_components: int) -> np.ndarray:
    """Simple PCA via SVD. Returns (n, n_components) array."""
    if arr.shape[1] <= n_components:
        return arr
    arr = arr - arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(arr, full_matrices=False)
    return arr @ Vt[:n_components].T


# ─────────────────────────────────────────────────────────────
# Per-episode MI scoring
# ─────────────────────────────────────────────────────────────

def score_episode(
    episode,
    k: int = 3,
    pca_components: int = 8,
    temporal_lag: int = 1,
) -> Dict[str, float]:
    """
    Compute MI score for one Episode.
    Returns dict with keys: mi_score, temporal_alignment, composite_score
    """
    n = len(episode)
    if n < k + 5:
        return {"mi_score": 0.0, "temporal_alignment": 0.0, "composite_score": 0.0}

    # ── Build state matrix (pad to uniform width) ──────────
    states_raw = []
    for obs in episode.observations:
        if obs.robot_state is not None and len(obs.robot_state) > 0:
            states_raw.append(obs.robot_state.astype(float))
        else:
            states_raw.append(np.zeros(1, dtype=float))

    ds_max = max(s.shape[0] for s in states_raw)
    states = [np.pad(s, (0, ds_max - s.shape[0])) for s in states_raw]
    S = np.vstack(states)     # (n, ds_max)

    # ── Build action matrix (pad to uniform width) ─────────
    actions_raw = [a.astype(float) for a in episode.actions]
    da_max = max(a.shape[0] for a in actions_raw)
    actions_padded = [np.pad(a, (0, da_max - a.shape[0])) for a in actions_raw]
    A = np.vstack(actions_padded)  # (n, da_max)

    # ── Sanitise: replace NaN / Inf with 0 ──────────────
    # Raw MCAP blobs (images, strings) can accidentally decode as
    # float64 garbage full of inf/nan values; clip them away.
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # If both S and A are all-zeros (no usable signal), score zero
    if S.max() == 0.0 and A.max() == 0.0:
        return {"mi_score": 0.0, "temporal_alignment": 0.0, "composite_score": 0.0, "n_steps": n}

    # ── PCA ─────────────────────────────────────
    S_r = _pca_reduce(S, pca_components)
    A_r = _pca_reduce(A, pca_components)

    # ── KSG MI (state → action) ─────────────────────
    try:
        mi = _ksg_mi(S_r, A_r, k=k)
    except Exception as e:
        log.warning(f"MI failed for {episode.episode_id}: {e}")
        mi = 0.0

    # ── Temporal alignment: cross-correlation lag ─────────
    try:
        a_mag = np.linalg.norm(A_r, axis=1)
        s_mag = np.linalg.norm(S_r, axis=1)
        a_std = float(np.std(a_mag))
        s_std = float(np.std(s_mag))
        if a_std < 1e-9 or s_std < 1e-9:
            # degenerate (constant signal) — neutral alignment
            temporal_alignment = 0.5
        else:
            corr = np.correlate(a_mag - a_mag.mean(), s_mag - s_mag.mean(), mode="full")
            lags = np.arange(-n + 1, n)
            norm = (a_std * s_std * n) + 1e-9
            corr_n = corr / norm
            best_lag = int(abs(lags[np.argmax(np.abs(corr_n))]))
            temporal_alignment = float(1.0 / (1.0 + best_lag))
    except Exception:
        temporal_alignment = 0.5

    composite = 0.7 * mi + 0.3 * temporal_alignment

    return {
        "mi_score": mi,
        "temporal_alignment": temporal_alignment,
        "composite_score": composite,
        "n_steps": n,
    }


# ─────────────────────────────────────────────────────────────
# Batch scoring & export
# ─────────────────────────────────────────────────────────────

def compute_and_rank(
    episodes: List,
    k: int = 3,
    pca_components: int = 8,
    easy_quantile: float = 0.75,  # top-25% = easy
    hard_quantile: float = 0.25,  # bottom-25% = hard
    out_dir: str = ".",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Score all episodes, rank, and export easy/hard eval JSON files.
    Returns (easy_list, hard_list).
    """
    results = []
    for i, ep in enumerate(episodes):
        log.info(f"  Scoring episode {i+1}/{len(episodes)}  [{ep.episode_id[:14]}…]")
        scores = score_episode(ep, k=k, pca_components=pca_components)
        results.append({
            "episode_id": ep.episode_id,
            "task": ep.metadata.get("task", ""),
            **scores,
        })

    if not results:
        return [], []

    scores_arr = np.array([r["composite_score"] for r in results])
    q_easy = float(np.quantile(scores_arr, easy_quantile))
    q_hard = float(np.quantile(scores_arr, hard_quantile))

    easy = [r for r in results if r["composite_score"] >= q_easy]
    hard = [r for r in results if r["composite_score"] <= q_hard]

    # Sort
    easy.sort(key=lambda x: x["composite_score"], reverse=True)
    hard.sort(key=lambda x: x["composite_score"])

    out = Path(out_dir)
    easy_path = out / "easy_eval.json"
    hard_path = out / "hard_eval.json"

    easy_path.write_text(json.dumps(easy, indent=2))
    hard_path.write_text(json.dumps(hard, indent=2))

    log.info(f"Easy eval ({len(easy)} episodes) → {easy_path}")
    log.info(f"Hard eval ({len(hard)} episodes) → {hard_path}")

    return easy, hard
