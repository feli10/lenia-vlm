# lenia-vlm

Open-ended evolution of MaCE-Lenia worlds (a mass-conserving Lenia, 160 x 160, three channels), steered by
foundation models. **Jev** (TypeSafe), a cheap decision model that only chooses between two pictures, does
everything frequent: entry duels, surprise detection, steering. A **generative VLM** is called only for rare
decisions: naming what is new and reviewing where an expedition is heading. **Luna** (OpenAI Decisions via
OpenRouter) and blind human ratings are held-out judges and never drive search.

**Status (2026-10-28):** the base search (MAP-Elites with islands and a surprise archive) and the surprise
detector are validated. The next experiment, the stage-1 core test of the escalation loop, is planned, not yet
built. Plans and results live in the claude.ai project "Research At NYU Game Innovation Lab" (start with
"VLM in the loop - escalation plan (core first)"), not in this repo.

## Setup

```bash
uv sync                     # uv sync --extra gpu for PyTorch (needed for the DINOv2 form axis)
uv run pytest               # offline, a few seconds; tests/test_golden.py guards cache compatibility
cp .env.example .env        # then add OPENROUTER_API_KEY (Luna) and TYPESAFE_API_KEY (Jev)
```

GPU runs go on Colab: `colab/mace.ipynb`. Run folders live in Google Drive `lenia-runs/` and are
rsynced to `runs/` (git-ignored) for analysis on the Mac.

## Code

| File | What it is |
|---|---|
| `src/lenia/sim.py`, `backend.py` | the Lenia / MaCE update (numpy `Batch`, PyTorch `TorchBatch`), numpy or GPU |
| `src/lenia/mace.py` | MaCE worlds as the search sees them: `simulate`, the 2x2 `picture`, `founders`, `mutate`, `texture`, the complexity prompt |
| `src/lenia/judge.py`, `prompts.py` | pairwise and multiple-choice judges (Jev, Luna, fake) with a persistent cache |
| `src/lenia/embed.py` | DINOv2 embedder (the form axis) |
| `src/lenia/rating.py` | local page for rating pairs of pictures yourself |
| `src/lenia/render.py`, `seeds.py`, `sample.py`, `alive.py`, `rle.py`, `config.py` | colours, starting patterns, random rules, body counting, config |
| `scripts/mace_qd.py` | MAP-Elites base search (islands, form axis, surprise archive `--sarch`) |
| `scripts/surprise_check.py` | the surprise duel prompt; validation; blind "unusual?" rating page |
| `scripts/goal_session.py` | goal-steered sessions (`steer` / `attain` prompts), the expedition machinery |
| `scripts/goal_groups.py` | goal library v3 (`data/goal_library_v3.json`) menus and matching |
| `scripts/qd_compare.py` | compare final archives with held-out Luna and blind ratings |
| `scripts/res_check.py` | gate 0f: Jev on smaller pictures (done; keep for reference) |

Each script's docstring lists its commands, including an offline `--judge fake` dry run.

`archive/` holds earlier phases (classic Lenia, the alive classifier, VLM distillation, the hill climber,
learnable novelty) and their docs and notebook. It is not imported by anything live.
