# lenia-vlm

Can a VLM's aesthetic taste, distilled into a reward model, guide search over Lenia creatures
without breaking what makes them creatures? Lenia (Chan 2019) is a continuous cellular automaton
whose rules can grow self-organising "creatures". Here a rule only counts if it grows one that is
**alive**: it doesn't fizzle, explode, fill the world, freeze, or exist only as pixel-scale speckle.
Aesthetics come on top of that hard constraint.

**Status:** Phase 0-1 done (simulation only); VLM judges tested and a prompt chosen; first
VLM-guided search ready to run. Results, calibrations and open questions:
[`docs/lenia.md`](docs/lenia.md).

## Setup

```bash
uv sync
mkdir -p data && curl -L -o data/animals.json \
  https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json   # Chan's catalogue (MIT)
uv run pytest                     # ~30 s, offline
uv run lenia orbium               # sanity check: Orbium glides
```

**GPU (Colab):** open `colab/lenia_colab.ipynb` in Colab (File -> Open notebook -> GitHub), pick a GPU
runtime and run the cells; results go to your Google Drive. Any command runs on a GPU with
`lenia --backend auto ...`; `lenia --backend auto bench` checks the GPU simulator against numpy and
times it. Locally, `uv sync --extra gpu` installs PyTorch (Apple GPUs work if PyTorch's FFT does).

API keys (only for the VLM judges): copy `.env.example` to `.env` and fill it in. Never put keys in
`.env.example`: it is committed.

## Main commands

```bash
uv run lenia survey --mode multikernel --n 240        # how many random rules are alive; contact sheets
uv run lenia explore --from runs/survey-multikernel   # novelty search over living creatures
uv run lenia show --run runs/survey-classic --alive   # GIFs
uv run lenia vlmtest build                            # creature set for the VLM test (no API calls)
uv run lenia vlmtest run --limit 3                    # smoke test: a few calls to Luna and Jev
uv run lenia vlmtest run                              # all pairs, both orders, the prompt sweep; then the report
uv run lenia vlmtest rate                             # rate 60 pairs yourself in the browser
uv run lenia guided run                               # VLM-guided search: Luna accepts mutations it prefers
uv run lenia guided check                             # held-out judges, drift, then `lenia guided rate`
```

Every command and what it writes: `docs/lenia.md` (Commands). Settings: `configs/lenia.yaml`.
Results go to `runs/` and data to `data/`; both are git-ignored.

## Layout

```
src/lenia/     sim.py (batched Lenia: numpy Batch, PyTorch TorchBatch), backend.py (numpy / GPU),
               bench.py (GPU check), alive.py (classifier), seeds.py, sample.py, survey.py,
               explore.py (novelty search + 200-unit and 2x-resolution checks), describe.py,
               catalogue.py, show.py, trace.py, render.py, cli.py,
               judge.py (Luna via OpenRouter, Jev via TypeSafe; cached), prompts.py, vlmtest.py,
               guided.py (VLM-guided search and its held-out check)
configs/       lenia.yaml; r26.yaml (double resolution: R = 26, 256x256)
colab/         lenia_colab.ipynb
scripts/       calibrations (growth, resolution) and the cross-check against Chan's code
tests/         test_lenia.py
docs/          lenia.md: results log
```

Split out of `vlm-pref-evo` on 2026-10-16; `judge.py` is adapted from that repo's judge.
