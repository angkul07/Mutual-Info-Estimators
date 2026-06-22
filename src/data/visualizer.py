"""
src/data/visualizer.py
───────────────────────
Randomly samples 5 trajectories and generates frame preview plots.
Each plot shows:  frame_0 | frame_mid | frame_end
with overlays for episode_id and trajectory_length.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data.hf_loader import Episode

log = logging.getLogger(__name__)

PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#e6edf3",
    "axes.edgecolor": "#30363d",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.titlecolor": "#58a6ff",
}


def _get_frame_image(ep: Episode, step: int) -> np.ndarray | None:
    """Return the image at a given timestep or None."""
    if step >= len(ep.observations):
        return None
    obs = ep.observations[step]
    return obs.image  # may be None or ndarray


def _draw_frame(ax, img, title: str, subtitle: str = "") -> None:
    """Render one frame onto an axes."""
    ax.set_facecolor("#161b22")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    if img is not None:
        if img.ndim == 1:
            # flat array – reshape best-guess square
            side = int(np.sqrt(img.size))
            img_disp = img[: side * side].reshape(side, side)
            ax.imshow(img_disp, cmap="viridis", aspect="auto")
        elif img.ndim == 2:
            ax.imshow(img, cmap="gray", aspect="auto")
        else:
            ax.imshow(img, aspect="auto")
    else:
        # Placeholder gradient when no image data
        gradient = np.outer(np.linspace(0.05, 0.3, 64), np.ones(64))
        ax.imshow(gradient, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.text(
            0.5, 0.5, "no image\ndata",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9, color="#58a6ff", alpha=0.8,
        )

    ax.set_title(title, fontsize=10, color="#58a6ff", pad=4)
    if subtitle:
        ax.text(
            0.5, -0.06, subtitle,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7, color="#8b949e",
        )


def _plot_action_curve(ax, ep: Episode) -> None:
    """Small action magnitude curve for context."""
    if not ep.actions:
        return
    mags = [float(np.linalg.norm(a)) for a in ep.actions]
    xs = list(range(len(mags)))
    ax.plot(xs, mags, color="#58a6ff", linewidth=1.5, alpha=0.9)
    ax.fill_between(xs, mags, alpha=0.2, color="#58a6ff")
    ax.set_xlim(0, len(mags) - 1)
    ax.set_ylabel("‖action‖", fontsize=8, color="#8b949e")
    ax.set_xlabel("step", fontsize=8, color="#8b949e")
    ax.set_title("Action Magnitude", fontsize=9, color="#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def visualize_trajectories(
    episodes: List[Episode],
    n_sample: int = 5,
    out_dir: str = "plots/trajectory_preview",
    seed: int = 42,
) -> List[str]:
    """
    Randomly sample up to `n_sample` episodes and produce one PNG per episode.
    Returns list of saved file paths.
    """
    plt.rcParams.update(PLOT_STYLE)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sample = rng.sample(episodes, min(n_sample, len(episodes)))
    saved = []

    for ep in sample:
        n = len(ep)
        frame_0   = _get_frame_image(ep, 0)
        frame_mid = _get_frame_image(ep, n // 2)
        frame_end = _get_frame_image(ep, n - 1)

        fig = plt.figure(figsize=(14, 5), facecolor="#0d1117")
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                              top=0.85, bottom=0.12, left=0.06, right=0.97)

        ax0   = fig.add_subplot(gs[0, 0])
        ax_m  = fig.add_subplot(gs[0, 1])
        ax_e  = fig.add_subplot(gs[0, 2])
        ax_ac = fig.add_subplot(gs[1, :])

        _draw_frame(ax0,   frame_0,   "Frame 0",   f"t={ep.timestamps[0]:.2f}s")
        _draw_frame(ax_m,  frame_mid, f"Frame {n//2} (mid)", f"t={ep.timestamps[n//2]:.2f}s")
        _draw_frame(ax_e,  frame_end, f"Frame {n-1} (end)", f"t={ep.timestamps[-1]:.2f}s")
        _plot_action_curve(ax_ac, ep)

        # Super-title
        short_id = ep.episode_id[:18] + "…" if len(ep.episode_id) > 20 else ep.episode_id
        fig.suptitle(
            f"Episode: {short_id}    |    Length: {n} steps    |    Task: {ep.metadata.get('task','?')}",
            fontsize=11, color="#e6edf3", y=0.97,
        )

        fname = out / f"preview_{ep.episode_id[:12]}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        log.info(f"Saved preview → {fname}")
        saved.append(str(fname))

    return saved
