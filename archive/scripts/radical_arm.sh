#!/usr/bin/env bash
# Radical prompt arm (2026-10-22): Luna on pool2's 657 held-out pairs under prompts.RADICAL_ARM, then the report.
# Run on the Mac from the repo root: bash scripts/radical_arm.sh   (~10.5k calls, about $1; the judge cache makes re-runs free)
set -euo pipefail
RUNS="${RUNS:-$HOME/Library/CloudStorage/GoogleDrive-zhangxufei05@gmail.com/My Drive/lenia-runs}"
uv run pytest -q
for p in weird_dual satisfying_dual violent_dual cute_dual deepsea_dual ugly_dual petrock_dual smaller_dual; do
  uv run lenia --runs "$RUNS" pool label --out "$RUNS/pool2" --judge luna --prompt "$p" --split heldout
done
uv run lenia --runs "$RUNS" pool arm --out "$RUNS/pool2"      # -> pool2/arm.md, arm.json, arm_top_/arm_bottom_ sheets
