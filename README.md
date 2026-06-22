<div align="center">

# 🤖 fidelity-eval

### Egocentric Manipulation Trajectory Evaluation Pipeline

*Pull → Parse → Score → Rank*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-angkul07%2Fabc--ego-orange?logo=huggingface)](https://huggingface.co/datasets/angkul07/abc-ego)
[![MCAP](https://img.shields.io/badge/Format-MCAP%20%2F%20ROS2-red)](https://mcap.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [First Principles](#first-principles)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Technical Workflow](#technical-workflow)
6. [Mutual Information Theory](#mutual-information-theory)
7. [Quick Start](#quick-start)
8. [CLI Reference](#cli-reference)
9. [Output Files](#output-files)
10. [Project Structure](#project-structure)

---

## What This Is

`fidelity-eval` is an end-to-end data pipeline that:

1. **Pulls** egocentric robot manipulation demonstrations from HuggingFace (`angkul07/abc-ego`)
2. **Parses** raw MCAP (ROS2 bag) files without any manual preprocessing
3. **Converts** them into a clean, typed trajectory schema
4. **Scores** each trajectory using **Mutual Information** between robot state and actions (DemInf-style)
5. **Ranks** trajectories and exports `easy_eval.json` / `hard_eval.json` for downstream policy training

The evaluation split is **signal-theoretic**: trajectories where actions are highly *informative* about state are "easy" (high competence, consistent), while low-MI trajectories are "hard" (noisy, hesitant, poorly coordinated).

---

## First Principles

### Why Mutual Information for Trajectory Quality?

In robot learning, a **good demonstration** is one where the operator's actions are *tightly coupled* to the sensed state — the robot is doing the right thing at the right time, for the right reason. Poor demos show:

- Hesitation (large state changes, small actions)
- Noise (random action jitter uncorrelated with state)
- Phase mismatch (actions lag or lead state transitions badly)

**Mutual Information (MI)** between state `S` and action `A`:

```
I(S; A) = H(A) - H(A | S)
```

measures exactly how much knowing the state reduces uncertainty about the action. High MI → **coherent**, intent-driven behavior. Low MI → **noisy** or **inconsistent** demonstrations.

### Why KSG Estimator?

Both `S` and `A` are **continuous, high-dimensional** vectors. Histogram or kernel density methods scale exponentially with dimensionality. The **Kraskov-Stögbauer-Grassberger (KSG) estimator** uses *k*-nearest-neighbor distances in the joint space:

```
Î(X; Y) = ψ(k) + ψ(N) − ⟨ψ(nₓ + 1)⟩ − ⟨ψ(nᵧ + 1)⟩
```

where `ψ` is the digamma function. It's **consistent**, **bias-corrected**, and scales as `O(N log N)` — perfect for per-episode trajectory scoring.

### Why PCA Before MI?

Raw joint / end-effector state vectors often have **correlated dimensions** (e.g., adjacent joints move together). Projecting onto the top-K principal components:

1. Decorrelates the signal
2. Reduces curse-of-dimensionality pressure on k-NN search
3. Ensures the MI estimate reflects *effective* degrees of freedom

---

## Dataset

| Property | Value |
|----------|-------|
| **Repository** | [`angkul07/abc-ego`](https://huggingface.co/datasets/angkul07/abc-ego) |
| **Storage format** | **MCAP** (Foxglove / ROS2 bag container) |
| **Total size** | 81 GB |
| **Task: `place_the_bread`** | **1,362 episodes** |
| **Episode container** | `data/train/<task>/episode_<uuid>/episode.mcap` |
| **Viewer** | Not available (no HF dataset card / data files config) |

### Why MCAP?

MCAP is the successor to ROS bags. Each file is a **time-indexed stream of typed messages** on *channels* (topics). Topics carry semantically named signals:

```
/camera/rgb/image_compressed   → image frames
/robot/joint_states             → proprioceptive state
/robot/end_effector_pose        → Cartesian EE state
/action                         → commanded actions
/depth/image_raw                → depth maps
```

The pipeline **auto-discovers and classifies** these topics using keyword heuristics — no hardcoded topic names required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         HuggingFace Hub                              │
│  angkul07/abc-ego  (81 GB, MCAP episodes)    1 file per episode      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │  hf_hub_download()  (per-file streaming)
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  HFLoader  (src/data/hf_loader.py)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  MCAP Reader → iter_messages()                                  │ │
│  │     ↓ classify topic → image | depth | state | action | ?      │ │
│  │     ↓ bucket into 50ms time windows                            │ │
│  │     ↓ decode: PIL decode | float64 array | uint8 fallback      │ │
│  │     ↓ build Observation + Action per timestep                  │ │
│  └──────────────┬──────────────────────────────────────────────────┘ │
│                 │  Episode { observations, actions, timestamps }     │
└─────────────────┼────────────────────────────────────────────────────┘
                  │
        ┌─────────┴───────────┬───────────────────┐
        ▼                     ▼                   ▼
┌──────────────┐   ┌─────────────────────┐  ┌────────────────────┐
│ Schema       │   │  Converter          │  │ MI Estimator       │
│ Inspector    │   │  (converter.py)     │  │ (mi_estimator.py)  │
│              │   │                     │  │                    │
│ schema.json  │   │ episodes.parquet    │  │ PCA → KSG MI       │
│ summary.md   │   │ (columnar, chunked) │  │ temporal alignment │
└──────────────┘   └─────────────────────┘  └────────┬───────────┘
                                                      │
                                            ┌─────────┴──────────┐
                                            │  Ranking & Export  │
                                            │                    │
                                            │  easy_eval.json    │
                                            │  hard_eval.json    │
                                            └────────────────────┘
                                                      +
                                            ┌─────────────────────┐
                                            │  Visualizer         │
                                            │  frame_0/_mid/_end  │
                                            │  action curve       │
                                            └─────────────────────┘
```

---

## Technical Workflow

### Step 0 — Download (`python main.py download`)

```
HuggingFace Hub API
  └─ list_repo_files()     →  enumerate all episode paths for task
  └─ hf_hub_download()     →  fetch one MCAP file at a time
  └─ local cache           →  cache/place_the_bread/episode_<uuid>/episode.mcap
```

**Key decisions:**
- No full-repo clone — each 60 MB MCAP is fetched on demand
- Cache is episode-level; re-runs are instant for cached episodes
- `--max N` flag enables partial downloading for development

### Step 0.1 — Inspect (`python main.py inspect`)

Reads the **first episode** only (one MCAP). Iterates all messages, tags channels by schema type, and produces:

- `outputs/schema.json` — machine-readable topic manifest
- `outputs/dataset_summary.md` — human report (topic table, signal classes, counts)

**Fail conditions:**  
- No `action`-class topics → `RuntimeError` (pipeline aborts)

### Steps 0.2–0.3 — Embed (`python main.py embed`)

For each episode:

```
MCAP → messages → 50 ms buckets → Observation + Action per step → Episode
  ↓
episode_to_rows() → [{episode_id, step, timestamp, action, robot_state, has_image, …}]
  ↓
PyArrow → Snappy-compressed Parquet (chunked, 500 rows/flush)
```

Output: `processed/episodes.parquet`

### Mutual Information (`python main.py compute_mi`)

For each episode with `n` steps:

```
1.  S = stack(robot_state[0..n])   shape (n, d_s)
    A = stack(action[0..n])        shape (n, d_a)

2.  S_r = PCA(S, k=8)             shape (n, 8)
    A_r = PCA(A, k=8)             shape (n, 8)

3.  mi = KSG(S_r, A_r, k=3)       scalar (nats)

4.  corr = cross_correlate(‖A‖, ‖S‖)
    best_lag = argmax |corr|
    temporal_alignment = 1 / (1 + best_lag)

5.  composite = 0.7 × mi + 0.3 × temporal_alignment
```

### Build Eval (`python main.py build_eval`)

```
Sort episodes by composite_score
  Top 25% → easy_eval.json     (high MI, well-coordinated)
  Bottom 25% → hard_eval.json  (low MI, noisy/hesitant)
  + 5 random previews → plots/trajectory_preview/*.png
```

---

## Mutual Information Theory

### KSG Estimator (full derivation)

Given `N` i.i.d. samples from `p(x, y)`, for each point `zᵢ = (xᵢ, yᵢ)`:

1. Find the *k*-th nearest neighbor in the joint space under the Chebyshev metric → distance `εᵢ`
2. Count neighbors within `εᵢ` in the marginal spaces:  
   `nₓᵢ = |{j : ‖xᵢ − xⱼ‖ < εᵢ}|`   (strictly less than)
3. Estimate:

```
Î(X; Y) = ψ(k) + ψ(N) − (1/N) Σᵢ [ψ(nₓᵢ + 1) + ψ(nᵧᵢ + 1)]
```

This is **bias-corrected** (the `+1` terms compensate for boundary effects) and **consistent** as N → ∞.

### Composite Score

```
score = 0.7 × I(S; A)  +  0.3 × (1 / (1 + |lag*|))
```

Where `lag*` is the cross-correlation peak lag between action magnitude and state magnitude trajectories. This penalizes episodes where actions and state changes are badly phase-shifted (a common failure in distracted or hesitant teleoperation).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline (12 episodes — fast demo)
python main.py download inspect embed compute_mi build_eval --max 12

# 3. Run full production pipeline (1362 episodes, ~80 GB download)
python main.py download inspect embed compute_mi build_eval
```

### Individual Steps

```bash
python main.py download repo=angkul07/abc-ego task=place_the_bread
python main.py inspect
python main.py embed      --max 50
python main.py compute_mi --max 50
python main.py build_eval --max 50
```

---

## CLI Reference

```
python main.py <command(s)> [overrides] [--flags]

Commands:
  download      Verify HF connectivity, cache first episode
  inspect       Discover MCAP schema → outputs/schema.json + dataset_summary.md
  embed         Download all episodes → processed/episodes.parquet
  compute_mi    KSG MI scoring + ranking → easy_eval.json + hard_eval.json
  build_eval    Visual inspection plots + (re)build eval splits

Config overrides (key=value):
  repo=<hf_repo_id>          default: angkul07/abc-ego
  task=<task_name>           default: place_the_bread

Flags:
  --max N        Process at most N episodes (for development)
  --no-validate  Skip episode integrity validation
  --seed N       RNG seed for trajectory sampling (default: 42)
```

---

## Output Files

| Path | Format | Description |
|------|--------|-------------|
| `outputs/schema.json` | JSON | MCAP topic manifest: schema names, classes, message counts |
| `outputs/dataset_summary.md` | Markdown | Human-readable dataset report |
| `processed/episodes.parquet` | Parquet (Snappy) | Columnar table, one row per timestep |
| `plots/trajectory_preview/*.png` | PNG | 5 random episode previews (frame_0 / mid / end + action curve) |
| `easy_eval.json` | JSON | Top-25% episodes by composite MI score |
| `hard_eval.json` | JSON | Bottom-25% episodes by composite MI score |
| `cache/place_the_bread/` | Dir | Raw MCAP files (60 MB avg each) |

### easy_eval.json / hard_eval.json Schema

```json
[
  {
    "episode_id": "00242bb8-86f6-4b90-a381-15203ec1a501",
    "task": "place_the_bread",
    "mi_score": 0.84,
    "temporal_alignment": 0.91,
    "composite_score": 0.86,
    "n_steps": 143
  },
  ...
]
```

### processed/episodes.parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | string | UUID of the episode |
| `step` | int32 | Time step index |
| `timestamp` | float64 | Seconds from episode start |
| `action` | list[float] | Action vector at this step |
| `robot_state` | list[float] | Robot state vector (may be null) |
| `has_image` | bool | Whether image observation exists |
| `has_depth` | bool | Whether depth observation exists |

---

## Project Structure

```
fidelity-eval/
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── configs/
│   └── source.yaml                # Source + output configuration
├── src/
│   ├── __init__.py
│   ├── mi_estimator.py            # KSG MI + PCA + ranking
│   └── data/
│       ├── __init__.py
│       ├── hf_loader.py           # HF download + MCAP parse → Episode
│       ├── schema_inspector.py    # MCAP schema discovery + reports
│       ├── converter.py           # Episode → Parquet
│       └── visualizer.py          # Trajectory preview plots
├── outputs/                       # Schema + summary reports
├── cache/                         # Downloaded MCAP episode files
│   └── place_the_bread/
│       └── episode_<uuid>/
│           └── episode.mcap
├── processed/
│   └── episodes.parquet           # Flattened columnar dataset
├── plots/
│   └── trajectory_preview/        # PNG previews (5 random episodes)
├── easy_eval.json                 # High-MI evaluation split
└── hard_eval.json                 # Low-MI evaluation split
```

---

## Design Philosophy

### No Manual Preprocessing
The pipeline is self-contained. Given only a HuggingFace repo ID and task name, it discovers the schema, downloads selectively, and produces all outputs without any intermediate manual steps.

### Streaming / Partial Downloads
Each MCAP file is downloaded to cache before processing. The `--max N` flag enables working with a subset without touching the 81 GB full dataset, making iteration fast during development.

### Fail-Loud Validation
The system raises hard errors (not warnings) if:
- No action topics are detected in the schema
- An episode has zero timesteps
- Observation and action counts are mismatched
- An episode has fewer than 2 steps

### Extensible Schema Classification
Topic classification is keyword-based and pluggable. Adding a new signal class requires only appending hint tuples in `hf_loader.py`.

---

## Dependencies

```
mcap>=1.0.0             # MCAP binary format reader
mcap-ros2-support       # ROS2 message type support
huggingface_hub>=0.20   # Dataset download + listing
numpy                   # Numerical arrays
pandas + pyarrow        # Parquet IO
matplotlib + Pillow     # Visualization
scipy                   # KSG digamma function
PyYAML                  # Config loading
```

---

<div align="center">
<i>Built for rigorous, reproducible robot learning data evaluation.</i>
</div>
