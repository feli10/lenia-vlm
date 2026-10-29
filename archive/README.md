# archive

Earlier phases of lenia-vlm, kept for reference and moved here on 2026-10-28. Nothing in the live code imports
from this folder, and the files here are not on any import path; running them would need their old locations.

- `src_lenia/`: classic-Lenia pipeline (alive classifier rollouts, surveys, novelty search, Chan's catalogue and
  found/ creatures), VLM test and distillation (pool, student reward, rsearch, qd), operator pilot, rank,
  learnable-novelty estimator, the `lenia` CLI.
- `scripts/`: the MaCE hill climber (`mace_evo.py`), the surprise-detector experiments v1-v3 (`surprise_check.py`),
  goal matching v3 (`goal_groups.py`), `mace_qd_full.py` (with the retired surprise-weighted parent selection), calibrations, the cross-check against Chan's code, goal
  matching v1/v2, learnable-novelty check, GIFs for the student/features runs.
- `tests/`: tests of the modules above (`test_lenia.py`; its live tests moved to `tests/test_core.py`).
- `docs/lenia.md`: the results log up to MaCE MAP-Elites (superseded by the claude.ai project docs).
- `colab/`: the 19-section notebook.
- `src_lenia/`: also `alive.py` (classifier), `rle.py` + `donors.json` (seed bank), `prompts_classic.py`, and the full
  pre-trim `render_full.py`, `seeds_full.py`, `sim_full.py`.
- `data/`, `configs/`, `third_party/`: goal libraries v1/v2, the R = 26 and found/ configs, the full pre-trim
  `lenia_full.yaml`, learnable_novelty.
