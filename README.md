# lenia-vlm

Can a VLM's aesthetic taste, distilled into a reward model, guide search over Lenia creatures
without breaking what makes them creatures? Lenia (Chan 2019) is a continuous cellular automaton
whose rules can grow self-organising "creatures". Here a rule only counts if it grows one that is
**alive**: it doesn't fizzle, explode, fill the world, freeze, or exist only as pixel-scale speckle.
Aesthetics come on top of that hard constraint.

**Status:** Phase 0-1 (simulation only, no VLM yet). Results, calibrations and open questions:
[`docs/lenia.md`](docs/lenia.md).

## Setup

```bash
uv sync
mkdir -p data && curl -L -o data/animals.json \
  https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json   # Chan's catalogue (MIT)
uv run pytest                     # ~30 s, offline
uv run lenia orbium               # sanity check: Orbium glides
```

API keys (only for the VLM phases): copy `.env.example` to `.env`.

## Main commands

```bash
uv run lenia survey --mode multikernel --n 240        # how many random rules are alive; contact sheets
uv run lenia explore --from runs/survey-multikernel   # novelty search over living creatures
uv run lenia show --run runs/survey-classic --alive   # GIFs
```

Every command and what it writes: `docs/lenia.md` (Commands). Settings: `configs/lenia.yaml`.
Results go to `runs/` and data to `data/`; both are git-ignored.

## Layout

```
src/lenia/     sim.py (batched Lenia), alive.py (classifier), seeds.py, sample.py, survey.py,
               explore.py (novelty search + 200-unit and 2x-resolution checks), describe.py,
               catalogue.py, show.py, trace.py, render.py, cli.py
configs/       lenia.yaml
scripts/       calibrations (growth, resolution) and the cross-check against Chan's code
tests/         test_lenia.py
docs/          lenia.md: results log
```

Split out of `vlm-pref-evo` on 2026-10-16. The VLM judge code (OpenRouter Decisions API, pairwise,
cached) will be brought over from there for the VLM phases.
