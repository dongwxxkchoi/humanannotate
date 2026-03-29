"""
Rebuild `data/random_100_samples.json` with ~50/50 hazard_correct from aligned JSONL.

Run from repo root:
  python streamlit_hazard_correct_labeler/build_balanced_random100.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE = Path(__file__).resolve().parents[1]
SAMPLING_ROOT = WORKSPACE / "Sampling_aligned_triplets"
OUT_PATH = WORKSPACE / "streamlit_hazard_correct_labeler" / "data" / "random_100_samples.json"

SEED = 42
TARGET_TOTAL = 100
TARGET_PER_CLASS = 50


def _is_usable_row(obj: Dict[str, Any]) -> bool:
    if "idx" not in obj or "groundtruth_hazard" not in obj:
        return False
    v = obj.get("response_hazard")
    if v is None:
        return False
    if isinstance(v, str) and v.strip().lower() == "none":
        return False
    return True


def collect_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    true_rows: List[Dict[str, Any]] = []
    false_rows: List[Dict[str, Any]] = []
    for path in sorted(SAMPLING_ROOT.rglob("*.jsonl")):
        rel = path.relative_to(WORKSPACE)
        rel_s = str(rel).replace("\\", "/")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not _is_usable_row(obj):
                    continue
                row = {
                    "source_jsonl_path": rel_s,
                    "idx": int(obj["idx"]),
                    "groundtruth_hazard": str(obj["groundtruth_hazard"]),
                    "response_hazard": obj.get("response_hazard"),
                }
                if bool(obj.get("hazard_correct")):
                    true_rows.append(row)
                else:
                    false_rows.append(row)
    return true_rows, false_rows


def main() -> None:
    true_rows, false_rows = collect_rows()
    n_t, n_f = len(true_rows), len(false_rows)
    if n_t < TARGET_PER_CLASS or n_f < TARGET_PER_CLASS:
        raise SystemExit(
            f"Need at least {TARGET_PER_CLASS} true and {TARGET_PER_CLASS} false; got {n_t} true, {n_f} false."
        )

    rng = random.Random(SEED)
    picked = rng.sample(true_rows, TARGET_PER_CLASS) + rng.sample(false_rows, TARGET_PER_CLASS)
    picked.sort(key=lambda r: (r["source_jsonl_path"], r["idx"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {OUT_PATH.relative_to(WORKSPACE)}")
    print(f"Pool: true={n_t} false={n_f} | sample: true={TARGET_PER_CLASS} false={TARGET_PER_CLASS} seed={SEED}")


if __name__ == "__main__":
    main()
