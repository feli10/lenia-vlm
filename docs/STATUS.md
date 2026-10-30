# Where lenia-vlm stands (2026-10-30)

Felix's plans and results live in a claude.ai project; this is the shareable summary. Numbers are from saved runs
in `runs/` (Felix's Drive), not in this repo.

## The setup
MaCE-Lenia worlds (mass-conserving Lenia, 160 x 160, three channels) evolve under MAP-Elites with 4 islands and a
surprise archive. Every world regrows from the same smoothed-noise start, so a genome is a rule set and the picture
is a 2x2 filmstrip of four snapshots (`lenia.mace.picture`, 644 px).

- **Jev** (TypeSafe's pairwise decision model, ~$0.0001 a duel) drives everything frequent: map entry (a
  complexity duel), surprise detection, goal steering.
- **A generative VLM** (Gemini, or open weights via OpenRouter) is called rarely: to name what is new in a world
  and to review an expedition halfway.
- **Luna** (OpenAI's decisions model) and Felix's blind ratings are held-out judges; they never drive search.
- A hand-coded texture rule (`lenia.mace.texture`) throws out grid-noise worlds, the main loophole.

## What works
- MAP-Elites with islands keeps many lineages alive (1 → 15 seed lineages vs a single map); held-out Luna
  preferred later elites over earlier ones (33 of 43 in v1).
- The Jev surprise duel agrees with Felix's "is something unusual here?" ratings well enough to fill an archive
  of stepping stones. It also works on 484-px pictures; the complexity duel only partly does, so it stays at 644.

## Stage 1 (current): goal-directed expeditions from surprising worlds
Triggers are the most surprising world per lineage. Each one gets a goal and a 10-step expedition (32 mutants a
step, Jev judges goal progress against the start, with a complexity floor). The arms differ only in who writes the
goal: **L** samples it from a fixed library, **N** has a VLM write it, **NR/NS** add one VLM review halfway.

- Arm L, three designs in a row: Luna's end-vs-start scores barely moved (attain ~0.57, complexity ~0.53).
- **Main finding: goal features are barely heritable.** Each step picks a child about 0.25 above the children's
  mean, but the next generation's mean gains about 0.006. Gentler mutation helps (scale 0.3 without ring edits
  keeps 36% of children within 0.1 of the parent's goal score, vs 14% at the base search's 3.0), but because
  every world regrows from noise, a tiny rule change still gives a different world. Expeditions act more like a
  local search around the trigger than a climb.
- Next: arm N v3, then Felix rates N vs L per trigger. A Stage 2 candidate: let children continue from their
  parent's final state ("world inheritance") instead of regrowing from noise.

## New directions (meeting 2026-10-30, Felix, Sam, Tim)
Simpler tests that give an objective signal and help tune hyperparameters (mutation scale, mutants per step,
number of steps, picture settings), alongside the full VLM run:

1. **CLIP-targeted search.** Sweep a set of text prompts; for each, evolve toward the prompt and check whether
   the result matches it in CLIP space. CLIP similarity is an objective score (as in ASAL's supervised target),
   unlike Jev's preferences. Not in the repo yet: `lenia.embed` has DINOv2 only.
2. **Larger structures.** Zoom out (a bigger world, possibly a longer run) and see whether larger structures
   appear, e.g. groups of bodies rather than single ones. The world size is a parameter (`--size` in
   `mace_qd.py`).
3. **A "which is noisier?" minimizer.** Add a Jev duel asking which world is noisier, minimize it, and see how
   it changes evolution (it might replace or back up the hand-coded texture rule). The prompt goes in
   `lenia/prompts.py`.

## Running things
- `scripts/mace_qd.py run` starts from scratch; its `--judge fake --embed-model fake` dry run takes about 30 s on a
  CPU. The real runs use a GPU (Colab: `colab/mace.ipynb`).
- Most other scripts read earlier runs from `runs/`; ask Felix for the folders you need.
- Using your own VLM: `src/lenia/describe.py` explains it (one method per provider). A new pairwise judge goes in
  `src/lenia/judge.py` (`_call` per backend) plus an entry under `judges:` in `configs/lenia.yaml`.
- Vertex Gemini is unavailable on Felix's Google project for now; use OpenRouter or your own provider.
