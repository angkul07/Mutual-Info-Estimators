"""
extract_top50.py
────────────────
Reads easy_eval.json and hard_eval.json, sorts by mi_score,
and writes top50_easy.json (highest scores) and top50_hard.json (lowest scores).
"""

import json
from pathlib import Path

N = 50

easy = json.loads(Path("easy_eval.json").read_text())
hard = json.loads(Path("hard_eval.json").read_text())

top_easy = sorted(easy, key=lambda x: x["mi_score"], reverse=True)[:N]
top_hard = sorted(hard, key=lambda x: x["mi_score"])[:N]

Path("top50_seasy.json").write_text(json.dumps(top_easy, indent=2))
Path("top50_shard.json").write_text(json.dumps(top_hard, indent=2))

print(f"top50_seasy.json  →  score range [{top_easy[-1]['mi_score']:.3f}, {top_easy[0]['mi_score']:.3f}]")
print(f"top50_shard.json  →  score range [{top_hard[0]['mi_score']:.3f}, {top_hard[-1]['mi_score']:.3f}]")