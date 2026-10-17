# lenia: is Lenia a good domain? (Phase 0-1)

Question for this phase: before any VLM call, does Lenia give us (1) a functional constraint that
binds without starving the search, and (2) enough visual range among the survivors for an aesthetic
reward to matter? Everything here is simulation only; no VLM, no API cost.

## What's built

| File | What it does |
|---|---|
| `src/lenia/sim.py` | Lenia in numpy, batched: classic (Chan 2019) and multi-kernel / multi-channel (Chan 2020). Torus, FFT convolution. Reads Chan's parameter format. |
| `src/lenia/rle.py` | Decoder for Chan's catalogue patterns (checked against his own decoder on 137 species, all identical). |
| `src/lenia/alive.py` | The "alive" classifier (below). `classify()` works from saved statistics, so thresholds can change without re-simulating (`lenia rejudge`). |
| `src/lenia/seeds.py` | The fixed bank of 8 starting patterns every rule is tried from. |
| `src/lenia/sample.py` | Random rules, using the ranges of Chan's own random search. |
| `src/lenia/survey.py` | Test 1 + 2: run rules x seeds, classify, write tables and contact sheets. |
| `src/lenia/catalogue.py` | Chan's catalogue (548 species, MIT licence) as the human reference set. |
| `src/lenia/show.py` | Animated GIFs of one rule or a grid of rules, camera following each creature (`lenia show`). |

Sanity check: Orbium from its catalogued pattern glides at constant mass (`lenia orbium`).

**Cross-check against Chan's own code (2026-10-14).** Chan's `Automaton` from `LeniaND.py` (GUI and GPU
stubbed out) and ours, run side by side from the same start on 30 random catalogue species for 300
steps: the largest cell difference is about 1e-7 after one step (float32 vs his float64), and stays
below 1e-3 after 300 steps for 27 of 30; three chaotic species (K4s, O4i, 3CtNa) drift to about 0.1 by
step 300, with total mass still agreeing. Same update rule; the catalogue species we call not-alive
are not a simulator bug. (`scripts/crosscheck_chan.py`; needs scipy and a checkout of Chan's repo.)

## The "alive" classifier

Measured on the second half of a 100-time-unit rollout (1,000 steps; 50 units before 2026-10-15), in
units of the kernel radius R. Rollouts that have clearly failed are dropped at checkpoints every 10
units (nothing left; more than half the world filled; or, from t = 20, more than 15% filled at two
checkpoints in a row), which makes the 100-unit horizon cheaper than the old 50-unit one.

| Class | Rule | Catches |
|---|---|---|
| fizzled | mass / R^2 drops below 0.02 | fades to nothing |
| exploded | occupies > 50% of the world | fills the world |
| spread | occupies > 15%, or radius of gyration > 2.5 R | space-filling stripes, dots, labyrinths, spirals |
| dying | end mass < 0.5x window peak | slow death |
| growing | end mass > 2x window start, or the fitted mass (or spread) trend over the window > 15% of its mean | slow explosion still under the size limits |
| multi | 2+ separate bodies at the end | fission, debris |
| frozen | speed < 0.02 R per time unit and shape change < 0.05 per time unit | static blobs and strings |
| **mover** | alive and travelling | gliders (Orbium: 0.48 R/time) |
| **pulser** | alive, in place, changing shape | rotating, breathing |

Shape change is the mass-normalised L1 difference between consecutive frames after re-centring
on the creature, per unit of time. `min_change` was first 0.01; at 0.01 to 0.05 the "pulsers" look
static in the filmstrips (re-centring jitter, not life), so it is now 0.05. That cut classic-mode alive
rules from 37 to 20.

**Check on real creatures:** run from their own catalogued pattern, 209 of Chan's 248 classic
species that fit a 128x128 world come out alive (84%; at min_change 0.01). Most misses are
species that are genuinely multi-body or space-filling.

## Seeds matter a lot

The same rule can give a creature or nothing depending on the start (your browser screenshots
showed this too). Calibrated on Chan's catalogue (rules known to have a creature):

| Starting pattern | Catalogue rules with a living creature |
|---|---|
| Uniform pixel noise (any size/density tried) | 0-3 of 24 |
| One soft blob | up to 5 of 24 |
| Smoothed noise, 1.5 R wide (2 seeds) | 9 of 24; 5 of 40 on held-out species |
| Five catalogue creatures' shapes, rescaled to the rule's R | 11 of 40 held-out |
| **Bank used: 5 creature shapes + 3 smoothed noise** | **36 of 80 held-out (45%)** |

So the bank finds a creature under about half of the rules that are known to have one: survey
numbers below are lower bounds. The shapes Scutium (S1s) and Gyropteron (PG1a) do most of the work.
Using creature shapes as starts biases towards known morphologies; the rule still decides what
grows.

## Test 1: how much does viability bind? (2026-10-14; classic and multikernel rerun at 100 units 2026-10-15)

**At 100 time units (current):** classic 17 of 400 rules alive (**4.3%**; 12 movers, 5 pulsers),
multikernel 13 of 240 (**5.4%**; 9 movers, 4 pulsers). Early stopping dropped 95% / 94% of rollouts
before the end (mean simulated time 23 units), so these took 4-5 minutes on 2 cores, less than the
50-unit runs below. Grid and multichannel were not rerun. The table below is the 50-unit version.

128x128 world, R = 13, T = 10, 8 seeds per rule, "alive" = any seed gives a mover or pulser
(min_change 0.05, via `lenia rejudge`; the contact sheets were drawn at 0.01, so they include some
near-static 'pulsers').

| Survey | Rules | Alive rules | Rollouts: fizzled / spread+exploded / frozen+multi | CPU time |
|---|---|---|---|---|
| grid: classic, one ring, 20x20 (m, s) | 400 | **3.0%** (12) | 78% / 21% / 0% | 20 min (1 core) |
| classic: 1-3 rings, random m, s | 400 | **5.0%** (20) | 53% / 42% / 4% | 9 min (2 cores) |
| multikernel: 1 channel, 3 kernels | 240 | **8.8%** (21) | 71% / 21% / 5% | 11 min (2 cores) |
| multichannel: 3 channels, 12 kernels | 100 | **1.0%** (1) | 97% / 2% / 0.4% | 21 min (2 cores) |

- **The constraint binds** in every mode: 91-99% of random rules fail. In classic and multikernel it
  leaves 3-9% alive, which is in the useful range (search can find survivors, the filter matters).
- **Two distinct failure modes, as seen in the browser:** fizzling (most common) and space-filling
  patterns (dots, labyrinths, stripes, spirals and target rings, `sheet_spread.png`). The
  space-filling ones are visually striking, which makes them the right distractors for the VLM gate.
- **The (m, s) map (`grid.png`)** shows creatures in two thin bands (Orbium's band near m = 0.12-0.18,
  s = 0.012-0.02, and a second near m = 0.26-0.36, s = 0.045), wedged between a large fizzle region
  and a large space-filling region. Classic Lenia's viable space is a sliver.
- **Movers are rare**: most survivors are pulsers (rotating or breathing in place). Movers: 10 of 400
  grid rules, 12 of 400 classic, 6 of 240 multikernel.
- **Multichannel starves random search**: 97% of rollouts are dead within the first half. Its
  creatures are colourful (RGB channels) but random sampling cannot reach them; that mode needs
  warm starts from known multichannel creatures (Chan's `Python/found/`, Hamon et al.'s published
  parameters, or Leniabreeder's ranges).

## Test 2: visual range among survivors (for Felix to judge)

`sheet_alive.png` in each survey folder; `catalogue_own.png` is the human-curated reference.

My read, to be checked by eye:
- Classic survivors: rings, gears, stars, Orbium-like gliders, rotating S-shapes, a striped disc.
  More variety than I expected, but mostly round and symmetric; several look like each other.
- Multikernel survivors are more irregular and organic (comets, sprouts, a ring with teeth, a
  morphing blob): probably the better aesthetic headroom.
- Chan's catalogue is clearly richer (long chains, multi-lobed swimmers, rotating star-fish), so there
  is a visible gap between random survivors and curated creatures for a reward to close.
- The colour map saturates: many creatures sit at value 1 and render as solid dark maroon, hiding
  inner structure. Worth trying another map before the VLM gate (colour map is a variable anyway).

## Mobility test: can search move through living-creature space? (2026-10-14)

`lenia explore` (`src/lenia/explore.py`, `src/lenia/describe.py`): novelty search over rules, no VLM.
Roots = the alive rules of a survey (classic: 19, multikernel: 16). A child is a mutated copy of an
archive member's rule (m, s, ring heights; plus r, h per kernel in multikernel), started from that
member's final creature, run for 30 time units; it joins the archive if alive and novel (mean
distance to its 5 nearest archive members, in 20 hand-made descriptors: speed, shape change, mass,
spread, a rotation-invariant angular spectrum, a radial profile). 3,200 children per run.

| | classic | multikernel |
|---|---|---|
| Children alive (at 30 time units) | 50% | 48% |
| Archive (roots + accepted children) | 276 | 285 |
| Lineages that produced descendants | 15 / 19 | 14 / 16 |
| Descriptor distance from own root, median (noise floor: a root vs itself 5 time units later) | 2.2 (0.35) | 6.6 (1.27) |
| Children farther from the nearest catalogue species than the catalogue's median species is from its nearest neighbour | 81% (roots: 32%) | 98% (roots: 94%) |
| Coverage cells (moving x size x symmetry order): roots / archive / catalogue / archive-only | 8 / 29 / 26 / 12 | 10 / 27 / 26 / 17 |
| **40 most novel children still alive at 100 time units** | **20 / 40** | **2 / 40** |
| 40 random children still alive at 100 time units | 26 / 40 | 18 / 40 |

**1. Search can move.** With the parent's creature carried over, half of all mutants stay alive (vs 3-9%
of random rules), and lineages travel many times the noise floor. Visible changes: an S-shaped
rotator becomes a chain of beads, rings become gears and five-pointed stars, a blob becomes a long
beaded worm (`sheet_lineages.png`, `lineages.gif`: each root next to its farthest descendant).

**2. Novelty on hand-made descriptors mostly rewards mess.** In classic Lenia the "most novel" are
ragged gears shedding debris (plus a creature that looks like a smiley face). In multikernel they
are long writhing worms and "colony discs" (a membrane around a dot or labyrinth pattern) that look
striking (`sheet_novel.png`).

**3. The most novel finds are mostly not alive: the search hacked the 30-time-unit check.** Rerun
for 100 time units, 34 of the 40 most novel multikernel creatures fill the world: they were slow
explosions still small enough at 30. The novelty objective selects exactly for creatures at the edge
of the viability filter, as we expected a VLM objective to do. (A first multikernel run showed the
same pressure through a descriptor-scaling bug, since fixed: novelty pushed creatures to grow right
up to the 2.5 R spread limit.) For "catalogue" comparisons, note the catalogue holds classic
single-kernel species, so multikernel creatures sit outside it almost by construction (94% of roots
already do).

**Implications**
- The viability check must cover a longer horizon (at least 100 time units) or test the growth
  rate, before any VLM run; otherwise any open-ended objective will find slow explosions. A cheap
  30-unit screen followed by a long confirmation for archive candidates keeps the cost down.
- Novelty alone gives mess, which is the argument for an aesthetic quality signal: a VLM reward
  (or a QD quality score) would have to prefer the beaded chain over the ragged gear.
- The multikernel space contains visually rich forms near the boundary (worms, colony discs). Whether
  durable versions exist is the next question, with the stricter viability check.
- Descriptors are hand-made; a learned image embedding (DINO/CLIP) would be a better novelty space
  and a better "is it new vs the catalogue" test. Not available in this environment (no model
  downloads), so it is for the lab machine.

## Longer horizon, growth check, confirmation stage (2026-10-15)

The first mobility test showed the 30-unit check was being gamed by slow explosions. Changes:
- Rollouts run 100 time units, judged on 50-100; failed rollouts are dropped at 10-unit checkpoints.
- **Growth check:** a rollout whose fitted mass (or spread) trend across the window exceeds 15% of
  its mean is "growing", i.e. not alive.
- Every rollout's statistics are kept every 0.5 units (`traces.npz`, a few KB each) and, for those
  we keep pictures of, a 32x32 thumbnail every 10 units (`thumbs.npz`); `lenia trace` plots them.
- Search: a child alive at 100 units and novel is a *candidate*; candidates are re-run to 200 units
  and join the archive only if still alive (and still novel) then.

**Calibration** (`scripts/calibrate_growth.py`, `runs/calibrate_growth.json`). Positives:
Chan's species from their own pattern (202 of 248 alive at 100 units). Negatives: 480 creatures that
the 30-unit searches had kept: only 254 of them are alive at 100 units, and of those, 49 (19%) are
dead by 200 ("late failures"). Most late failures are not steadily growing at 100 (median trend
0.001); they destabilise later. At 0.15 the growth check rejects 0% of catalogue species and 0.5%
of durable creatures, and catches 12% of late failures; the 200-unit confirmation catches the rest.

**Search rerun at 100 units with confirmation at 200** (roots: the new surveys' survivors):

| | classic | multikernel |
|---|---|---|
| Children alive at 100 units | 48% | 58% |
| Candidates / failed the 200-unit confirmation | 352 / 93 (26%) | 427 / 101 (24%) |
| Archive | 188 (17 roots) | 223 (13 roots) |
| Drift from root, median (noise floor) | 1.8 (0.38) | 2.7 (0.83) |
| Children farther from the catalogue than its median nearest-neighbour distance | 68% (roots 35%) | 95% (roots 92%) |
| Coverage cells: roots / archive / catalogue | 9 / 22 / 28 | 5 / 13 / 28 |
| CPU time (2 cores) | 23 min | 47 min |

- The worms and colony discs are gone: they were slow explosions. Durable lineages still change shape
  (an S-rotator becomes a beaded rod, a star becomes a ring), but drift is smaller than with the
  30-unit check (4.7x vs 6.4x the noise floor in classic): part of the earlier "movement" was the
  search drifting into transient forms.
- **A third failure mode: speckle.** What the multikernel search now finds most novel are small,
  fragmented, flickering blobs: bounded, alive by our rules, but chaotic pixel noise rather than a
  coherent body (the confetti problem again, at creature scale). A simple measure separates them:
  *roughness* (total edge length per unit mass). Chan's species: median 0.58, 99th percentile 1.07.
  Multikernel archive: median 1.38, and all 40 of its most novel members above 1.07 (59% of all its
  members also have >6% isolated cells, vs 1% for the catalogue). Classic archive: median 1.07; 72%
  of its 40 most novel above 1.07.
- **Open design decision:** add a coherence constraint (e.g. roughness below the catalogue's 99th
  percentile) to "alive", or leave speckle in as a test of the VLM's taste. Adding it is cheap but is
  arguably an aesthetic criterion smuggled into the functional constraint; leaving it in makes speckle
  a natural distractor for the VLM gate (does the VLM prefer a smooth Orbium to a flickering speck?).
  Possible middle ground: keep it out of the constraint for the gate, run the search both ways.
  **Resolved below: speckle turns out to be a grid artifact, so it is now excluded.**

## Coherence: the resolution check (2026-10-15)

**Speckle is a grid artifact, not a taste question.** A Lenia creature is a continuum object: run the same
rule with R doubled and the pattern upsampled 2x, and it should be the same creature at a finer
grid. Re-running the 10 most novel members of the multikernel archive this way: all 10 freeze. Of the
10 most novel classic members, 9 freeze. Those creatures' liveliness lived at the scale of single cells.

**The check** (`explore.rescale_check`): R -> 2R, final pattern upsampled 2x (bilinear), 256x256 world,
50 time units. Passing means alive there **and** the same creature: descriptor distance between the
creature and its 2x self <= `RESCALE_MAX_DIST`. Alive-only is not enough: several novel speckle
creatures pass it by relaxing, at 2x, into one shared smooth body (a ring next to a crescent), so
their 1x differences were pixel-level.

**Calibration** (`scripts/calibrate_rescale.py`, `runs/calibrate_rescale.json`). Of 202
catalogue species alive at 1x, 194 (96%) are alive at 2x; 2x horizons of 50 and 100 units give the
same 194. Descriptor distance to their 2x selves: median 0.68, p95 1.98, p99 2.59. The threshold is set
at the p95 (1.98), so about 9% of real species would fail the full check. For scale: neighbouring
catalogue species are a median 1.44 apart. The descriptor is coarse, so this is a filter, not proof.

**The check applied to existing archives after the fact** (p95 threshold):

| Archive | Pass | Of the 40 most novel |
|---|---|---|
| multikernel, no check | 93 / 210 | 0 |
| multikernel, alive-at-2x check only | 122 / 182 | 10 |
| classic, no check | 105 / 171 | 10 |

**Multikernel search rerun with the full check** (same budget; roots also checked: 9 of 13 pass;
the check's runs count toward the budget):

| | no check | alive-at-2x only | full check |
|---|---|---|---|
| Archive | 223 | 191 | 174 |
| Candidates failing the 2x check | - | 38 | 86 |
| Drift from root, median (noise floor) | 2.7 (0.83) | 2.3 (0.46) | 1.9 (0.45) |
| Children beyond the catalogue's p90 nearest-neighbour distance | 51% | 43% | 20% |
| Roughness, median (catalogue: 0.58, p99 1.07) | 1.38 | 1.29 | 1.05 |
| Of the 40 most novel, rougher than the catalogue's p99 | 100% | 95% | 72% |
| Movers among children | 49% | 58% | 72% |
| CPU time (2 cores) | 47 min | 84 min (1 worker) | 61 min |

Readings:
- **Much of what looked most novel was speckle.** Children far outside the catalogue fell from 51% to
  20%. Drift from roots is still 4x the noise floor, so the search still moves.
- **What's left are coherent blobs, with limited variety.** At 2x, the most novel members are smooth
  bodies, often a body plus an appendage or a faint ring around it (`novel-2x.gif`). At 1x they look
  pixelly: at R = 13 these multikernel creatures are only 10-15 cells across. **Render at 2x for
  anything a person or VLM looks at.** It costs only the archive members, not the search.
- **Goodhart, again.** Several of the most novel members sit just under the threshold (distance
  1.93-1.96 against 1.98), and #23 still turns into the ring-and-crescent body at 2x. The search leans
  toward the edge of any constraint we give it. A stricter threshold, or a third scale (3x), would
  tighten this; so would searching at R = 26 directly (4x the cost).
- The classic search in `runs/explore-classic` was run without the check; rerun it with the
  check before using it.

## Gates

- Test 1: **pass** for classic and multikernel; **fail** for random multichannel (warm start needed).
- Test 2: **Felix's call** from the sheets.
- Next (Phase 2, VLM gate) needs: survivors from classic + multikernel, catalogue creatures as the
  "clearly better" side of separated pairs, and spread/frozen patterns as distractors.

## Commands

```bash
uv run lenia orbium                                   # sanity check
curl -L -o data/animals.json https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json
uv run lenia catalogue --start own                    # classifier check + reference sheet
uv run lenia catalogue --start bank --n 80            # seed-bank check
uv run lenia survey --mode grid --n 400
uv run lenia survey --mode classic --n 400
uv run lenia survey --mode multikernel --n 240
uv run lenia survey --mode multichannel --n 100       # slow: ~12 kernels per step
uv run lenia rejudge --run runs/<survey>        # re-classify with the current thresholds
uv run lenia show --run runs/survey-classic --alive       # GIF grid of every alive rule (~20 s)
uv run lenia show --run runs/survey-classic --rules 27,141 --time 100
uv run lenia show --chan O2u,S1s,PG1a,H3s                        # catalogue species
uv run lenia show --rule '{"R":13,"T":10,"b":"1","m":0.15,"s":0.015}' --seed 0   # any rule (Chan's format)
uv run lenia explore --from runs/survey-classic --evals 3200            # mobility test (~8 min here)
uv run lenia explore --replot runs/explore-classic                     # redraw + lineage GIF
uv run lenia explore --confirm runs/explore-multikernel                # long re-runs: really alive?
uv run lenia trace --run runs/survey-classic --rules 18,141            # a rollout's future: mass, size, speed, thumbnails
uv run python scripts/calibrate_growth.py <explore runs...>           # growth-threshold calibration
uv run python scripts/calibrate_rescale.py                            # resolution-check calibration (catalogue)
uv run lenia explore --from ... --rescale-time 0                             # search without the resolution check
```

Runs above were made on a 2-core cloud machine; on a laptop with more cores they are proportionally
faster (`--workers`, default all cores but one).
