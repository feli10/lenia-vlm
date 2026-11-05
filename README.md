# lenia-vlm

Open-ended evolution of MaCE-Lenia worlds (a mass-conserving Lenia, 160 x 160, three channels), steered by
foundation models. **Jev** (TypeSafe), a cheap decision model that only chooses between two pictures, does
everything frequent: entry duels, surprise detection, steering. A **generative VLM** is called only for rare
decisions: naming what is new and reviewing where an expedition is heading. **Luna** (OpenAI Decisions via
OpenRouter) and blind human ratings are held-out judges and never drive search.

**Status:** see [docs/STATUS.md](docs/STATUS.md): what works, the current stage, and the next directions.

## Setup

```bash
uv sync                     # uv sync --extra gpu for PyTorch (needed for the DINOv2 form axis)
uv run pytest               # offline, a few seconds; tests/test_golden.py guards cache compatibility
cp .env.example .env        # then add your keys: OPENROUTER_API_KEY (Luna, open-weights VLMs), TYPESAFE_API_KEY (Jev),
                            # GOOGLE_CLOUD_PROJECT (Vertex Gemini; after `gcloud auth application-default login`)
```

No keys needed to try it: an offline MAP-Elites run with fake judges (about 30 s on a CPU):

```bash
uv run python scripts/mace_qd.py run --out /tmp/qd-dry --judge fake --embed-model fake --iterations 3 --batch 8 \
    --init 8 --size 96 --warmup 200 --render 120 --gap 40
```

GPU runs go on Colab: `colab/mace.ipynb`. Run folders live in Google Drive `lenia-runs/` and are
rsynced to `runs/` (git-ignored) for analysis on the Mac. Most scripts other than `mace_qd.py run` read
earlier runs from there; ask Felix for the folders you need.

## Code

| File | What it is |
|---|---|
| `src/lenia/sim.py`, `backend.py` | the Lenia / MaCE update (numpy `Batch`, PyTorch `TorchBatch`), numpy or GPU |
| `src/lenia/mace.py` | MaCE worlds as the search sees them: `simulate`, the 2x2 `picture`, `founders`, `mutate`, `texture`, `bodies` |
| `src/lenia/judge.py` | pairwise and multiple-choice judges (Jev, Luna, fake) with a persistent cache; several questions per request (`MultiQuery`) |
| `src/lenia/describe.py` | the generative VLM (names, reviews): Vertex Gemini, OpenRouter, fake; cache and spend cap. Another provider is one method |
| `src/lenia/prompts.py` | **every prompt**: judge duels (complexity, surprise, interest, steer, attain, verify), goal menus, and the VLM's namer and review (`GEN`); texts are frozen by the cache |
| `src/lenia/embed.py` | DINOv2 embedder (the form axis); `ClipScorer` (CLIP text-target scores, `transformers`) |
| `src/lenia/rating.py` | local pages for rating yourself: pairs (left / right) and single worlds (unusual / typical / unsure) |
| `src/lenia/stats.py` | AUC with a permutation p-value, one-sided Fisher test |
| `src/lenia/render.py`, `seeds.py`, `sample.py`, `config.py` | colours, the smoothed-noise start, random rules, config (`configs/lenia.yaml`: the judges) |
| `scripts/mace_qd.py` | MAP-Elites base search (islands, form axis, surprise archive `--sarch`) |
| `scripts/goal_session.py` | goal-steered sessions with library v3 goals (`data/goal_library_v3.json`), the expedition machinery |
| `scripts/qd_compare.py` | compare archives with held-out Luna, then your blind "unusual?" ratings (`rate`) |
| `scripts/stage1.py` | Stage 1 of the escalation loop: `triggers` (re-score the surprise archive, best per lineage), `expedition --arm L/N/NR/NS` (guarded expeditions, simulation cache, one review for NR/NS), `check` (held-out Luna, end vs start; gate 0c) |
| `scripts/maptest.py` | map-prompt test: the base search with the user's aim as entry duel (`mace_qd.py run --like --aim`) vs "complex, life-like": map and texture `stats`, a blind matched-cell interest session (`build`, `rate`, `report`) |
| `scripts/gates.py` | Stage 0 gates: 0e (several questions per request), 0d (verify) |
| `scripts/clipbench.py` | CLIP targeted-search bench: which search settings steer toward a text target (inheritance vs regrowth, mutation strength, sep-CMA-ES, local control); no API calls |
| `scripts/bakeoff.py` | gate 0a: which VLM names the triggers (`ping`, `names`, `verify`, `rate`, `report`) |
| `scripts/res_check.py` | gate 0f: Jev on smaller pictures (done; keep for reference) |
| `scripts/pin_check.py` | confirms the pinned Jev version answers like our past results (~$0.015) |

Each script's docstring lists its commands, including an offline `--judge fake` dry run.

`archive/` holds earlier phases (classic Lenia, the alive classifier, VLM distillation, the hill climber,
learnable novelty, the surprise-detector and goal-matching experiments) and their docs and notebook. It is not imported by anything live.
