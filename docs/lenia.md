# lenia: is Lenia a good domain? (Phase 0-1)

Question for this phase: before any VLM call, does Lenia give us (1) a functional constraint that
binds without starving the search, and (2) enough visual range among the survivors for an aesthetic
reward to matter? Everything here is simulation only; no VLM, no API cost.

## Where things stand (2026-10-19): read this first

The doc is a lab notebook in time order; later sections supersede earlier ones. Current pipeline:
- **Substrate:** Chan's multi-channel found/ creatures at a common R = 20, world 256^2
  (`configs/found.yaml`), alive v2 (alive at 100 and 200 units, coherent at 2x: descriptor distance and
  texture loss, calibrated on the survey `found-survey-v2`).
- **Teacher:** Luna, prompt `curate_dual`, on two-view images (motion onion-skin + 2x2 close-ups), both
  orders averaged. Taste is lexicographic: size first, motion second; prompts barely steer it; agrees
  with Felix about 0.70. Jev is the held-out second judge.
- **Pool:** pool2, 433 creatures, 2,733 training / 657 held-out pairs.
- **Student:** frozen DINOv2 ViT-S/14 (768-d over the two views) + linear Bradley-Terry head, 5-member
  bootstrap ensemble (`distill_dinov2_vits14_curate_dual`). Baseline: the same fit on five statistics.
- **Search results:** greedy (Phase 4) and MAP-Elites over size x speed (Phase 5): the student stays
  calibrated to Luna, ranks unseen elites better than the five statistics (32 vs 14 discordant pairs),
  no over-optimisation found. But all found creatures are near neighbours of the pool (section "How far
  from the pool"), so these are results about the training neighbourhood.
- **Next:** the mutation-operator pilot (section "Next: mutation-operator pilot"), then MAP-Elites with the
  operators that pass and an embedding-novelty axis, then the teacher check again.
- **Data:** run folders in Google Drive `lenia-runs/` (found-survey-v2, expA2-explore, pool2, rsearch-*,
  qd-*, qd-gifs). Colab notebook `colab/lenia_colab.ipynb` (sections 5-9); judge calls run on the Mac
  (API keys in `.env`).

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

## Early VLM test (2026-10-17)

Tim's advice: bring VLMs in early. Before improving the substrate, ask two judges about the creatures
we already have: **Luna** (`openai/gpt-6-luna-decisions`, OpenRouter Decisions API, the judge used in
vlm-pref-evo) and **Jev** (TypeSafe System One; image input is experimental).

**Creature set** (`lenia vlmtest build`, `runs/vlmtest/`): 105 creatures in five groups, each
re-simulated and kept only if it still behaves as its group says:

| Group | n | What |
|---|---|---|
| catalogue | 29 | Chan's species from their own patterns (11 of 40 drawn were dropped: split, froze or spread) |
| coherent | 30 | multikernel search with the resolution check (15 most novel, 15 random), shown at 2x |
| classic | 18 | classic search, no resolution check |
| speckle | 13 | most novel finds of the multikernel search without the check, all rougher than any catalogue species (7 of 20 died or split on re-simulation) |
| failure | 15 | random rules that froze (5) or spread/exploded into world-filling textures (10) |

Each image is a 2x2 grid of four snapshots three time units apart (648x648 px), one colour map,
bilinear upscaling, cropped to the creature's own extent so every creature fills its frame (a first
version used a fixed 3R crop, where catalogue species looked bigger than ours: a size confound). 309
random pairs (each creature meets at least three others), asked in both orders, with two prompts:
*appeal* ("which creature is more visually appealing?") and *lifelike* ("which looks more like a
living organism?"). About 2,500 calls; Luna roughly $0.10-0.20, Jev a fraction of a cent.

**What it measures** (`lenia vlmtest report`): order consistency and first-slot bias; group-vs-group
win rates; Bradley-Terry scores per creature; agreement between judges and between prompts; and which
creature statistics the scores track (size, mass, speed, change, roughness).

**Questions:** do the judges prefer Chan's creatures to ours? Do they fall for speckle? Do the
world-filling textures win on *appeal* (they're striking) while losing on *lifelike*? Do Luna and Jev
agree? Caveats: stills, not video (frozen failures are only detectable as four identical snapshots);
speckle and classic creatures look slightly pixelated at their native resolution, which is part of
what they are.

**Results** (`runs/vlmtest/report.md`; Luna cost about $0.15, Jev effectively nothing; 0 failed calls).
Pair counts are order-averaged wins; p from a two-sided sign test.

| | Luna appeal | Jev appeal | Luna lifelike | Jev lifelike |
|---|---|---|---|---|
| catalogue beats coherent (ours) | 33/38 (p<1e-5) | 29/38 (p=0.002) | 28/38 (p=0.005) | 18/38 (n.s.) |
| coherent beats speckle | 22/24 (p<1e-4) | 21/24 (p<1e-3) | 17/24 (p=0.06) | 18/24 (p=0.02) |
| coherent beats failures | 18/21 (p=0.001) | 13/21 (n.s.) | 20/21 (p<1e-4) | 21/21 (p<1e-6) |
| order agreement / P(first slot wins) | 0.65 / 0.63 | 0.90 / 0.50 | 0.64 / 0.63 | 0.93 / 0.50 |

Judge vs judge, rank correlation of creature scores: appeal 0.65 (0.61 within our 48 creatures),
lifelike 0.78. Luna's appeal and lifelike scores correlate 0.77; Jev's only 0.39.

Readings:
- **The judges are not fooled by speckle.** It sits at the bottom on appeal for both. The resolution
  check and the VLMs' taste point the same way.
- **There is headroom.** Both judges prefer Chan's species to our search's creatures on appeal, by a
  wide margin. An aesthetic reward has a visible gap to close, which is what the project needs.
- **Appeal is not aliveness.** Jev rates world-filling textures as appealing as our creatures (one
  labyrinth texture is in its top 12), but on *lifelike* every judge puts failures last (our
  creatures beat them 21/21 for Jev). An aesthetic objective alone would drift toward textures; the
  hard constraint earns its place.
- **The judges agree** with each other substantially, including within our own creatures (0.61): a
  shared signal with real variance in exactly the space a reward model would be trained on.
- **Jev is the more reliable judge here:** 0.90-0.93 order agreement and no position bias, against
  Luna's 0.64-0.65 with a strong first-slot bias (in vscene Luna managed 0.86). Jev is also ~free.
  Caveat: its image mode is experimental and may change between releases.
- **What they like:** intricate, symmetric bodies with internal structure (4H3t, the radial species,
  our comb-shaped c056). Scores rise with creature size relative to the kernel (Spearman about +0.6)
  and liveliness (+0.4 to +0.6), and fall with roughness (-0.5 to -0.7 on appeal).
- **Caveat: resolution.** Speckle, classic creatures and frozen failures were simulated at R = 13 and
  look slightly blocky when upscaled; part of their penalty may be pixelation itself. A follow-up
  should render everything at the same effective resolution.

### Images v2: a fixed camera (2026-10-17)

Felix spotted while rating that the first images hid motion: the camera re-centred on the creature in
every snapshot, so a glider with a steady shape looked exactly like a frozen blob. Fast catalogue
creatures move about 0.4-0.5 kernel radii per time unit, i.e. several body lengths over the 9 units
shown. v2 images: one fixed camera per creature across its four snapshots, centred on its mean
position and sized to its whole path; a faint grid every kernel radius (drawn only on empty
background) to make displacement readable; snapshots 2 units apart instead of 3. Same 105
creatures and 309 pairs. Trade-off: fast movers take up less of the frame. The results above are
from v1 (`runs/vlmtest-v1/`) and stand as a first look; the prompt sweep and the human ratings use v2.

### Prompt sweep and human ratings (2026-10-17)

The guided search will optimise whatever the prompt rewards, quirks included, so the prompt is picked
first. Same 309 pairs, both judges, both orders, six prompts (`src/lenia/prompts.py`): `appeal`
(baseline), `appeal_bare` (same question, no Lenia framing), `beautiful`, `curate` ("you are curating a
gallery of the most remarkable creatures discovered in Lenia; which would you rather include?", an
expert framing that matches the catalogue, itself a curated collection), `interesting`, `lifelike`.
(A first draft had a video-game "creature design" framing; dropped, since no game uses these creatures
and it would steer judges toward a different notion, e.g. cute, readable mascots.) About $0.25 more of Luna.

Felix rates 60 of the pairs himself (`lenia vlmtest rate`: a local page, the same images the judges
saw, random left/right, keyboard or click; answers go straight to `runs/vlmtest/human.csv`).

Picking a prompt, from the comparison table at the top of `report.md`: self-consistency (same winner
in both orders), Luna-Jev agreement, separation (speckle at the bottom, catalogue above ours), low
correlation with nuisance statistics (size, roughness), and agreement with Felix's picks.

**Sweep results** (v2 images; `runs/vlmtest/report.md`; Felix rated 60 pairs, none skipped).
Human agreement has a 95% margin of about ±0.10 at n = 60.

| judge / prompt | order agr. | 1st slot | catalogue > ours | ours > speckle | ours > failure | Luna~Jev | Felix |
|---|---|---|---|---|---|---|---|
| luna / appeal | 0.62 | 0.67 | 0.70 | 0.82 | 0.73 | 0.69 | 0.75 |
| jev / appeal | 0.87 | 0.54 | 0.71 | 0.70 | 0.52 | 0.69 | 0.77 |
| luna / appeal_bare | 0.82 | 0.42 | 0.65 | 0.87 | 0.68 | 0.65 | 0.62 |
| jev / appeal_bare | 0.89 | 0.48 | 0.72 | 0.80 | 0.32 | 0.65 | 0.60 |
| luna / beautiful | 0.68 | 0.64 | 0.68 | 0.75 | 0.81 | 0.65 | 0.83 |
| jev / beautiful | 0.91 | 0.51 | 0.69 | 0.67 | 0.49 | 0.65 | 0.73 |
| **luna / curate** | **0.84** | **0.55** | **0.78** | **0.75** | **0.73** | **0.76** | **0.85** |
| jev / curate | 0.88 | 0.48 | 0.65 | 0.55 | 0.52 | 0.76 | 0.77 |
| luna / interesting | 0.86 | 0.50 | 0.78 | 0.59 | 0.75 | 0.85 | 0.78 |
| jev / interesting | 0.88 | 0.44 | 0.64 | 0.49 | 0.56 | 0.85 | 0.77 |
| luna / lifelike | 0.52 | 0.72 | 0.65 | 0.63 | 0.74 | 0.76 | 0.77 |
| jev / lifelike | 0.93 | 0.47 | 0.52 | 0.64 | 0.76 | 0.76 | 0.70 |

Readings:
- **Framing matters, and the right one helps.** Without the Lenia context (`appeal_bare`), agreement
  with Felix drops to 0.60-0.62, and Jev's top 12 includes four world-filling textures (it prefers
  failures to our creatures, 0.32). With context, every prompt reaches 0.70-0.85.
- **`curate` fixes Luna's position bias:** order agreement 0.84 (vs 0.62 for `appeal`), first slot
  0.55. It also has the best agreement with Felix (0.85), good separation, and Luna-Jev agreement 0.76.
  Its top 12 is all coherent creatures, mostly catalogue species.
- **`interesting` is risky as an objective:** the judges agree most (0.85), but speckle nearly ties
  our creatures (0.49-0.59). An "interestingness" reward would pull toward noise.
- **Felix's taste matches the judges' biases.** On his 60 pairs he picks the bigger creature
  (relative to the kernel) 62% of the time, the livelier 71%, the smoother 62%; Luna under `curate`:
  53%, 75%, 60%. The size and liveliness correlations look like shared taste, not judge artifacts.
- **Near-static creatures** (alive by our thresholds, change < 0.1 and speed < 0.03; 17 of them, 10
  from our coherent search) score low with every judge (e.g. Luna/curate -0.50 vs +0.33 for other living
  creatures) and lose to other living creatures in 82% of Felix's picks. The judge already penalises
  this loophole; raising the thresholds is optional.
- Averaging Luna and Jev under `appeal` reaches 0.88 agreement with Felix: two judges together beat
  either alone, a cheap option if a single judge proves too noisy.

**Decision:** `curate` is the prompt for the guided search; Luna drives (best match to Felix, now
self-consistent); Jev under `curate` is the held-out judge; Felix rates a final sample.

## VLM-guided search (2026-10-17)

First use of a VLM as the objective (`lenia guided`, `src/lenia/guided.py`). 16 lineages, founded by
diverse coherent creatures from the multikernel search with the resolution check (farthest-point
sampling in descriptor space, restricted to creatures whose 2x distance is at most 75% of the
threshold: the search had piled up right under it, and mutants of those edge cases mostly failed).
Each generation each lineage makes one child, a mutated rule started from its current creature, which
must pass, cheapest first: alive at 100 units; coherent at 2x (this run also makes the picture);
Luna under `curate` prefers it to its parent, P(child) averaged over both orders >= 0.6; alive at 200.
A lineage whose mutant dies or fails the 2x check tries another (up to 3 per generation; dry runs:
about half of mutants die, and ~70% of survivors pass the 2x check). One duel per lineage per
generation, so at most 1,600 Luna calls for 50 generations (roughly $0.10-0.20). State is saved every
generation; a stopped run resumes.

`lenia guided check`: Luna (driver) and Jev (held out, same prompt) judge each lineage's checkpoints
(generations 10, 25, 50) against its founder, and its founder and final creature against the same
four catalogue species (does the gap to Chan's creatures close?); plus how size, speed, liveliness
and mass drifted. The driver-vs-held-out gap is the first over-optimisation signal. Felix then rates
founder vs final and final vs a catalogue species for each changed lineage (`lenia guided rate`).

Pictures: `lineages.png` (rows = lineages, columns = checkpoints), `lineages.gif` (founders above
finals, animated at 2x).

**Results** (`runs/guided/`: 50 generations, 16 lineages, Luna about $0.05; 15 lineages changed,
44 accepted steps, 0-5 per lineage). P(final preferred to founder), both orders averaged:

| | final vs founder | lineages improved | founder vs catalogue | final vs catalogue |
|---|---|---|---|---|
| Luna (driver) | 0.80 | 14/15 | 0.27 | 0.41 |
| Jev (held out) | 0.55 | 9/15 | 0.34 | 0.39 |
| Felix (held out) | 7/15 | - | - | final beats catalogue 4/15 |

Luna's curve is flat after generation 10 (0.79, 0.77, 0.80 at generations 10, 25, 50); Jev's too
(0.53, 0.54, 0.55).

**This is over-optimisation of the judge, in its plainest form.** The driver says almost every lineage
improved and the gap to Chan's species closed by half; a held-out VLM sees a small effect, and Felix
sees none. What Luna pushed toward (from the images): rounder, more compact bodies (outline
lumpiness halved, 0.087 -> 0.043, rounder in 11/15 lineages), slower and less lively (median speed
0.037 -> 0.023, liveliness 0.22 -> 0.14, down in 8/15), several lineages converging on similar
discs and rings. Felix and the judges had all preferred *livelier* creatures on the fixed test set.
Not the rendering: frame fill rose only slightly and tracks Jev's verdicts more than Luna's.

Why it matters: Luna was the judge that best matched Felix on a fixed set of creatures (0.85). Under
selection that agreement broke: the search finds the directions where Luna's taste and Felix's part
ways (and Luna is deterministic, so its quirks are perfectly consistent targets). Agreement on a
fixed sample does not predict agreement under optimisation pressure. This is the phenomenon the
project set out to study, now visible in the direct-VLM case, before any distillation.

Caveats: 15 lineages and 30 human pairs (7/15 has a wide interval); one run, one seed; Jev's image
mode is experimental.

Natural next experiments: (1) a judge ensemble as the driver (Luna + Jev averaged; reached 0.88 with
Felix on the fixed set), held out against a third model and Felix: do ensembles resist
over-optimisation here, as reported for reward-model ensembles in RLHF? (2) a stricter acceptance
rule (e.g. both orders must prefer the child); (3) diversity pressure (QD) against the convergence
on discs; (4) then distillation: does a student reward model over-optimise faster than its teacher?

## GPU port and Colab (2026-10-17)

`TorchBatch` (`src/lenia/sim.py`) is the numpy update step line for line in PyTorch; `backend.py`
picks numpy or PyTorch (`lenia --backend auto`: CUDA, else Apple MPS if its FFT works, else numpy).
Only the step runs on the GPU: worlds are copied back when recorded (every 5th step) for the alive
classifier, descriptors and pictures, which stay numpy. On a GPU the CLI uses one process with
batches of 64. `lenia bench` checks PyTorch against numpy (one step to float32 rounding; Orbium
after 30 units) and times both. `--runs DIR` puts all results (and the judge cache) elsewhere, e.g.
Google Drive. `configs/r26.yaml`: double resolution for Experiment A. Notebook: `colab/lenia_colab.ipynb`.

Verified on Colab (Tesla T4, torch 2.11, 2026-10-17), `lenia --backend auto bench`: PyTorch vs
numpy one step max difference 2-5e-7 (float32 rounding) in all three setups; Orbium after 30 units
max cell difference 4.5e-3, mass difference 1.5e-4. ALL CHECKS PASSED.

| ms per creature per step | numpy (1 CPU process) | T4, batch 64 | speedup |
|---|---|---|---|
| classic 128x128, R=13 | 0.53 | 0.016 | 33x |
| multikernel 256x256, R=26 | 6.1 | 0.13 | 47x |
| multichannel 256x256, R=26 | 30.2 | 0.51 | 60x |

End to end, a 64-rule x 8-seed multikernel survey at R=26 took 38 s on the T4, about 10x the
2-core build sandbox: the alive classifier and recording still run on the CPU every 5th step, so the
GPU idles part of the time. Moving those statistics to the GPU is the next speed-up if needed.

**Chan's `found/` collections** (github.com/Chakazul/Lenia, Python/found/, not in animals.json):
about 3,000 2D creatures with multiple kernels and channels, e.g. 954 three-channel 15-kernel
creatures (233.json), 752 more (231b.json), 642 (233c.json), plus 2-, 4-, 6- and 7-channel sets.
Records are concatenated JSON objects in Chan's parameter format, which our simulator reads: a sample
from 213, 221 and 233 ran correctly, most alive after 30 units. These are the warm starts for
Experiment A, and a much larger reference set than the 548 classic species.

## Experiment A: multi-channel creatures at higher resolution (2026-10-17, set up)

Question: does a richer substrate (several channels and kernels, larger kernel radius) give more
varied, more complex coherent creatures than the single-channel R = 13 searches above?

- `lenia found download | survey` (`src/lenia/found.py`): Chan's `found/` files (10 by default: 212,
  213, 221, 223, 231, 231b, 232, 233, 233c, 233s; 2,589 creatures after removing duplicates; 1,559 of
  them 3-channel / 15-kernel, 774 3-channel / 9-kernel, plus 1- and 2-channel sets; R from 8 to 41).
  The survey runs each from its own pattern for 100 units in a 256x256 world (`configs/r26.yaml`),
  classifies it, and keeps the living ones' final patterns and descriptors (`alive.npz`).
- `lenia explore --from-found <survey> --roots 64`: novelty search as before (alive at 100, alive
  at 200, coherent at 2x), started from 64 diverse living found creatures (farthest-point sampling on
  descriptors), with every living found creature as the reference set ("beyond the reference" now
  means beyond Chan's multi-channel finds, a much higher bar than the classic catalogue).
- The search is now resumable: a checkpoint (`checkpoint.pkl`) every 2 minutes; re-running the same
  command with the same `--out` continues, and a higher `--evals` extends a finished run.
- Multi-channel creatures render with one channel per RGB primary on black (Chan's convention).
  Note for the VLM phase: colour is then a confound against single-channel creatures.
- Caveats: the 2x-coherence threshold (1.98) was calibrated on classic species; recalibrate on
  found creatures before trusting it there. CPU dry runs (small subset, 128x128) passed; the real
  runs are on Colab.

### Results (Colab T4, 2026-10-17)

Survey (100 per file, 620 run): 192 alive by our checks (31%; classic catalogue 84%). By file:
212 19/44, 213 50/88, 221 23/40, 223 9/21, 231 3/23, 231b 8/100, 232 1/63, 233 35/100, 233c 27/100,
233s 27/41. The rest: frozen 120, spread 104, multi 72, fizzled 70. Some of these are our narrower
definition (a record in 231 is named "stable stationary"; multi may include splitting creatures).
Visually the living ones are far richer than the classic searches: internal structure, several
coloured parts, rings, lobes, chains (`sheet_233.png`, `sheet_233s.png`). The substrate passes.

Search (38 roots: 64 picked, 50 alive, 38 coherent at 2x; 3,222 evaluations, 61 generations, 33 min):
archive 204 and still growing linearly; children alive 51%; drift from root median 2.6 (noise floor
0.63), max 13.2; 27% of children beyond the reference set's p90 nearest-neighbour distance.

But the novelty is concentrated. 166 children, of which 26 from one root (231b:LQRNVH) and 20 from
221:#9. Of the 40 children farthest from the reference set, 16 are that one 231b lineage: a creature
whose green channel forms a labyrinth of stripes 1 to 2 cells wide. That root is the only one at
R = 12 (roots keep Chan's own R: 14 of 38 at R = 20, 1 at 12), and it and its descendants are the 23
highest-frequency patterns in the archive (power at wavelengths under 4 cells: 0.20 at the root, up to
0.32 in descendants; archive median 0.03). Another 14 of the 40 are single-channel 212 lineages
(rendered on white, so they also look different for a trivial reason). Other clearly new forms: the
spiky star from 221:#9 (#173), a four-lobed 212 (#90), the 233:M3R6NK lineage.

**Is the labyrinth a grid artifact? No (checked).** At first it looked like speckle again. Re-run at
R = 26 (rule scaled, pattern resampled, 30 units), #146 and its root keep the labyrinth, and the
stripes widen with R, from about 2 cells to 4-5. Measured in units of R the fine structure is
unchanged: the share of power at wavelengths under R/3 is 0.309 at R = 12 and 0.322 at R = 26 for
#146 (root: 0.204 and 0.208; the spiky star #173, R = 38 to 26: 0.044 and 0.031). Structure that
scales with the kernel is real Lenia structure, not grid noise. So the family is a genuine
striped-skin creature that the descriptors happen to reward heavily (stripes light up the angular
orders). That makes it a novelty-measure problem (one family dominating), not a validity problem.

Changes for the second run (alive v2, built 2026-10-17; `configs/found.yaml`):
1. Every found creature at a common R = 20 (`lenia found survey` brings creatures to the config's
   world.R; pattern resampled, rule R changed). Fine structure is then resolved by 3+ cells, and every
   creature is rendered at the same scale. Why 20 and not 26: 20 is Chan's own most common R for these
   files, and at R = 26 17% of the creatures would not fit a 256x256 world (8% at R = 20); a 384x384
   world would cost 2.3x.
2. 'Spread' judged by shape, not world size. The first survey called most large creatures spread:
   64 of 125 creatures 90-128 cells wide, against 0 of 342 under 60. The cause is the occupancy cap
   (15% of the world, set for small classic creatures), which at R = 20 in 256x256 binds before the
   R-relative radius-of-gyration cap (2.5 R). `configs/found.yaml` raises it to 25%, just above a disc
   at the rg cap, so the shape cap decides.
3. Checks calibrated on the creatures being searched: the survey runs every living creature at 2x
   and writes `calibration.json` (2x descriptor distance: p95 of those alive at both scales, as for
   the classic catalogue; texture loss: p99). `explore --from-found` uses it.
4. A texture-loss term in the 2x check (`explore.texture_loss`): the share of power at wavelengths
   under 4 cells at R, minus the share under 8 cells at 2R (the same wavelengths in units of R). Real
   structure keeps its share (the measurement above); grid noise that relaxes at the finer grid loses
   it. This catches noise that relaxes into a body with the same mass profile, which the
   descriptor-based check cannot see, without punishing real fine structure the way an absolute
   texture cap would. Threshold: p99 of Chan's creatures, but never below 0.03 (about the whole
   fine structure of a median creature): real creatures vary by a few hundredths with phase, grid
   noise loses 0.1-0.3. `sheet_texture.png` shows the creatures with the most fine structure and
   those losing the most at 2x.
5. Roots are drawn only from creatures that pass the calibrated checks (farthest-point sampling
   otherwise favours outliers, which are the ones that fail: 12 of 50 in the first run).
6. Multi-channel files only (221, 223, 231, 231b, 232, 233, 233c, 233s): every creature renders as
   colour on black.

Deferred (novelty, not validity): one family dominating the top of the archive is a property of the
hand-made descriptors; Experiment B (embedding novelty) or a per-lineage cap would address it. For
building a labelling pool, sampling stratified by lineage is enough.

### Run 2 results (alive v2, Colab T4, 2026-10-17)

Survey (all 2,265 multi-channel creatures that fit, at R = 20): 644 alive (28%): 233 338/826,
233c 254/637, 231b 64/626, 221 20/40, 233s 24/41, 223 9/21, 231 3/23, 232 2/51. The largest
failure classes are frozen (511) and spread (316). Calibration: 553 of 644 alive at 2x; 2x
distance threshold 2.95 (p95; classic catalogue 1.98; p99 6.6); texture-loss threshold 0.037 (p99;
median loss 0.000). Chan's most textured creatures (`sheet_texture.png`) are real fine structure
(dotted halos, fringes, rings) and lose almost none of it at 2x, as predicted.

Search: 45 roots (64 picked, 56 alive, 45 coherent at 2x); 3,202 evaluations, 63 generations,
42 min; archive 224 (179 descendants, 38 of 45 lineages); children alive 50%; 2x failures 45 (run 1:
80), texture-loss failures 2 (the check rarely binds, as intended). Drift from root: median 3.3
(noise floor 0.56), max 10.5. 47% of descendants lie beyond the reference set's p90
nearest-neighbour distance (run 1: 27%, against a smaller reference set).

What the pictures show:
- The top of `sheet_novel` is varied: 13 lineages among the 40 farthest (run 1: one lineage had 16
  of 40). New forms: linked double rings ("dumbbells", 233:Z18A9R), a spiral that grew out of a
  blob (233c:G26RKS, #215), a hollowed green disc with a striped fringe (231b:E45DZL), a ring that
  loses its arms (233c:WWS78E).
- One new loophole, on the novelty side: very light creatures. Eight of the top 40 (#150, #154,
  #156, #179 from 221:#14; #216, #210, #133, #145) are faint, small single-channel crescents or
  outlines, lighter than 99% of Chan's living creatures. They are alive (peak values 0.8-1.0, they
  move and persist), but mass is one of the 20 descriptors, so being lighter than anything Chan
  has counts as novel. A validity question only if we want a minimum size; otherwise one for the
  novelty measure (Experiment B) and for the VLM, which should rate them low.
- No grid-scale texture anywhere in the archive (largest texture loss 0.031).

Verdict: the substrate and the alive check pass. The check is frozen as alive v2 (`configs/found.yaml`
plus the survey's `calibration.json`) for the labelling phase.

## Phase 2: labelling pool (built 2026-10-17)

Order of work from here: check the teacher (Luna, `curate`) on the new kind of creature, distil it
(Phase 3), then search against the student (Phase 4). The prompt sweep and the 85% agreement with
Felix were measured on classic and multikernel creatures at R = 13; multi-channel creatures at R = 20
look different, so the check is repeated on this pool before any label is used for training.

`src/lenia/pool.py`, `lenia pool build | label | rate | report`:
- **build** (Colab): about 300 of Chan's living creatures that pass the calibrated checks (at most 40%
  from one file) plus up to 10 descendants per lineage from the run-2 archive; each re-simulated
  from its stored pattern and rendered as the fixed-camera 2x2 image (faint grid now drawn on black
  backgrounds too). 20% of the creatures are held out (stratified chan/search). Pairs: each training
  creature in about 16 random training pairs, each held-out creature in about 16 held-out pairs. Held-out pairs
  are only between held-out creatures, so the student is tested on creatures it never saw.
- **label** (your Mac): Luna on every pair (training labels; its held-out labels are the teacher's
  answers the student is scored against), Jev on held-out pairs only (a second opinion).
  `labels_<judge>_<prompt>.csv`, one row per pair with both orders and the order-averaged p_a.
- **rate**: 60 held-out pairs for you, with the judge's own question.
- **report**: order agreement, first-slot rate, confidence, share of decisive pairs, Luna vs Jev,
  agreement with you, chan vs search win rates, what the scores track (size, mass, speed, change,
  texture), top and bottom 12 per judge.

Gate before Phase 3: on this pool Luna should keep order agreement around 0.8 or better and agree with
Felix about as well as before (0.85 on the old pool), and Luna and Jev should agree clearly above
chance. If not, fix the judge (prompt, image) before distilling.

On the Mac (Drive for desktop makes the Colab folder a local path):

```bash
RUNS="$HOME/Library/CloudStorage/GoogleDrive-zhangxufei05@gmail.com/My Drive/lenia-runs"
uv run lenia --runs "$RUNS" pool label --judge luna --limit 20        # smoke test, a few cents
uv run lenia --runs "$RUNS" pool label --judge luna                   # all pairs (about 7,000 calls)
uv run lenia --runs "$RUNS" pool label --judge jev --split heldout
uv run lenia --runs "$RUNS" pool rate                                 # 60 held-out pairs for you
uv run lenia --runs "$RUNS" pool report
```

### Judge check on the pool (2026-10-17)

Pool: 433 creatures (283 chan, 150 search; 87 held out), 2,733 training and 657 held-out pairs; 11
of Chan's creatures dropped as not alive in the 30-unit render run.

| | order agr. | 1st slot | decisive | chan > search | agrees with Felix (60) | picks larger size / mass | picks faster / more change |
|---|---|---|---|---|---|---|---|
| Luna / curate | 0.79 | 0.54 | 0.47 | 0.49 | 0.72 (decisive pairs 0.77, close 0.65) | 0.68 / 0.71 | 0.50 / 0.52 |
| Jev / curate | 0.87 | 0.44 | 0.33 | 0.53 | 0.58 | (r = +0.70 with size) | |
| Felix | | | | 0.56 (25) | | 0.57 / 0.52 | 0.68 / 0.63 |

Luna and Jev pick the same winner on 76% of held-out pairs, and on 99% of the pairs where both are
decisive. Felix also prefers smoother creatures (picks the less textured one 65% of the time); Luna
is nearly neutral there (53%).

Reading:
- Luna is consistent enough to distil (order agreement about 0.8), and it ranks the faint
  light-creature loophole at the very bottom, as hoped. Its top 12 are large, structured creatures:
  dumbbells, fringed discs, spirals.
- Search descendants are judged as good as Chan's own creatures (0.49 by Luna, 0.53 by Jev).
- Agreement with Felix drops from 0.85 (classic pool) to 0.72, and the difference is specific:
  Felix rewards motion and change, Luna does not respond to them at all (0.50, 0.52) but rewards
  size and mass. Felix saw motion in the same images (the faint grid shows a shift); Luna
  apparently does not, and little motion is visible to begin with: the median creature moves 0.05 R
  per time unit, so about 0.1 R between panels 2 units apart (0.75 R across all four panels even at
  the 90th percentile).
- Jev agrees with Felix barely above chance (0.58) and is the most size-driven: it stays a second
  opinion, not a teacher.

### Motion test (2026-10-17)

The 87 held-out creatures re-rendered with 8 time units between snapshots instead of 2 (`pool rerender
--snap 8`, folder `pool-snap8`); Luna re-labelled the 657 held-out pairs, and Felix re-rated the same
60 pairs on the new images.

| | Luna order agr. | Luna agrees with Felix | Luna picks faster | Luna picks larger (size / mass) | Felix picks faster / more change |
|---|---|---|---|---|---|
| 2 units (pool) | 0.80 | 0.72 | 0.54 | 0.71 / 0.73 | 0.68 / 0.63 |
| 8 units (pool-snap8) | 0.81 | 0.55 | 0.50 | 0.72 / 0.74 | 0.77 / 0.72 |

Read naively, wider gaps made Luna worse. Two things explain it:
- **A framing confound.** The fixed camera fits each creature's whole path, so fast creatures are
  drawn smaller: with 8-unit gaps, on-screen size relative to the 2-unit images falls with speed
  (Spearman -0.46; median creature 72% of its old size). Luna picks the creature that is larger on
  screen 67% of the time; Felix picks the smaller one 72% of the time (on screen, smaller now means
  faster).
- **Luna does value motion when it is not confounded with size.** In pairs where one creature is at
  least twice as fast and is not smaller on screen, Luna picks the faster one 86% of the time (64
  pairs).

Noise ceiling: Felix's own picks on the same 60 pairs agree across the two image versions 78% of the
time, so the 0.72 agreement on the original images is close to his own test-retest consistency.

Tried while analysing (not labelled): a fixed camera with one scale for all (a window of at least
10 R) makes slow, detailed creatures tiny, because a fast glider needs 12 R; a tracking camera at one
scale with a world-fixed grid hides motion again (the sliding grid is too weak a cue); tracking at
one scale with the creature's recent path drawn as a thin line shows motion and keeps true size
(`vlmtest.tracking_image`; the trail is a prototype).

### Two-view images (2026-10-17)

Felix's proposal after the motion test: give each creature a zoomed-out view where motion is clear
and zoomed-in views where shape is clear (`vlmtest.dual_image`, `pool rerender --camera dual`).
One composite image per creature (so the judges still compare image A with image B, and the student
later gets one input), 1032x512:
- left, motion: a 12 R window fixed in the world and centred on the creature's mean position, the
  same scale for every creature; the creature's positions over the last 16 time units overlaid as
  multiple exposures (every 4 units, earlier = fainter; chronophotography); faint world grid every R.
  Distance travelled is comparable across creatures, and nothing is zoomed to fit its path.
- right, close-ups: four panels 2 time units apart in reading order, each centred on the creature in
  a 7 R window, again the same scale for every creature: shape and shape change, at true size.
- prompt `curate_dual`: the same curator question with a description of the two views;
  the rating page shows a matching description.

`pool compare` now also reports `weights`: a logistic regression of the choices on the
differences in speed, change, size, mass and texture (standardised, log for speed, change and mass),
so each feature's effect is read with the others held fixed. On the existing labels:

| | speed | change | size | mass | texture |
|---|---|---|---|---|---|
| Luna, 2-unit images | +0.02 | +0.22 | -0.69 | +1.33 | +0.12 |
| Luna, 8-unit images | -0.11 | +0.33 | -0.46 | +1.36 | +0.25 |
| Felix, 2-unit images (60) | +0.73 | +0.03 | +0.51 | -0.14 | -0.61 |
| Felix, 8-unit images (60) | +0.79 | +0.37 | +0.06 | +0.11 | -0.71 |

Held fixed against each other, Luna's choices are driven by mass (how much material, i.e. how much
bright colour) and not by speed; Felix's by speed and smoothness (60 pairs, so his weights are
rough). Size and mass are correlated, so their separate signs matter less than their sum.

Result (Luna `curate_dual` on the 657 held-out pairs; Felix re-rated the same 60 pairs):

| | Luna order agr. | Luna agrees with Felix | Luna picks faster / more change / larger | Luna weights: speed, change, size, mass, texture |
|---|---|---|---|---|
| 2-unit images | 0.80 | 0.72 | 0.54 / 0.52 / 0.71 | +0.02, +0.22, -0.69, +1.33, +0.12 |
| two-view images | **0.89** | 0.70 | **0.65** / 0.58 / 0.85 | **+0.25**, +0.14, +0.56, +0.48, -0.06 |
| Felix, two-view (60) | | | 0.75 / 0.70 / 0.60 | +0.78, +0.13, +0.39, +0.22, -0.70 |

- The format works: Luna is markedly more self-consistent (0.89 order agreement), and motion now
  registers in its choices (speed weight +0.25 with size and mass held fixed, from about 0). Felix
  finds it the most intuitive of the formats tried, and his own picks agree 83-85% with his picks on
  the earlier formats.
- Agreement with Felix stays at 0.70. The remaining gap is taste, not representation: Luna weights
  size and mass about twice as much as Felix (true size is now salient in the close-ups) and
  ignores fine texture, which Felix dislikes (-0.70).
- Decision: the two-view image is the format for the pool and for the reward model's input; Luna
  with `curate_dual` is the teacher. The Luna-human gap (size up, texture-blind, motion under-weighted)
  is a measured property of the teacher, to report and to watch in Phase 4 (the student should inherit
  it; search against it should push toward big creatures).
- Full pool re-rendered in this format: `pool2` (all 433 creatures, same ids, splits and pairs; Felix's
  60 two-view ratings copied in).

### Full pool in the two-view format (pool2, 2026-10-18)

Luna `curate_dual` on all 3,390 pairs, Jev on the 657 held-out pairs:

| | order agr. | agrees with Felix (60) | chan > search | r(score, size) | r(score, speed) |
|---|---|---|---|---|---|
| Luna | 0.86 | 0.70 | 0.40 | +0.90 | +0.27 |
| Jev | 0.83 | 0.72 | 0.47 | +0.83 | +0.40 |

Luna and Jev now pick the same winner on 88% of held-out pairs (100% where both are decisive); Jev
went from 0.58 to 0.72 agreement with Felix with the new images.

**The teacher is mostly a size detector.** "The larger creature wins" reproduces 86% of Luna's
labels and 98% of its decisive ones (Jev 80%, Felix 60%). Broken down by how different the two sizes
are (radius of gyration ratio):

| size ratio | pairs | Luna order agr. | confidence | picks larger | picks faster | picks more change |
|---|---|---|---|---|---|---|
| 1.0-1.1 | 515 | 0.68 | 0.31 | 0.57 | 0.67 | 0.60 |
| 1.1-1.25 | 657 | 0.72 | 0.34 | 0.75 | 0.62 | 0.60 |
| 1.25-1.5 | 785 | 0.89 | 0.51 | 0.92 | 0.61 | 0.58 |
| > 1.5 | 1,433 | 0.98 | 0.72 | 0.99 | 0.56 | 0.55 |

Luna behaves lexicographically: if one creature is clearly bigger it wins, confidently and
consistently; only among similar sizes do motion and change decide, and there Luna leans the same
way as Felix (faster 0.67). Search descendants are larger than Chan's creatures (median radius of
gyration 0.99 vs 0.66 R), which is why they now win 60% of mixed pairs.

Consequence for the study: distilled as is, the student will mostly learn size, a feature one line
of code computes, and search against it will push creatures to the size limits of the alive check
(radius of gyration 2.5 R, occupancy 25%). That is a clean composition result, but a thin aesthetic
one. Options: distil as is and report it; remove size from the image (each creature drawn relative
to its own size: close-ups fit the creature, motion in body lengths); label size-matched pairs;
or tell the judge to set size aside (a prompt arm).

### Size-removed test (2026-10-18): not adopted

Each creature drawn relative to its own size (`--camera dual_body`, prompt `curate_dual_body`), 87 held-out
creatures, Luna on the 657 held-out pairs, Felix on the same 60 pairs. Decision rule fixed beforehand:
adopt if Luna's "picks larger" < 0.75, order agreement >= 0.80 and agreement with Felix >= 0.70.

| | Luna order agr. | confidence | decisive | agrees with Felix | picks faster / larger / heavier | Luna weights: speed, change, size, mass, texture |
|---|---|---|---|---|---|---|
| two-view, true size | 0.89 | 0.53 | 0.54 | 0.70 | 0.65 / 0.85 / 0.84 | +0.32, +0.15, +0.68, +0.45, -0.06 |
| two-view, size removed | 0.76 | 0.28 | 0.16 | 0.75 | 0.62 / 0.65 / 0.66 | +0.17, +0.10, -0.18, +0.41, +0.03 |

Size preference drops as intended and agreement with Felix rises a little, but Luna becomes much less
sure (only 16% of pairs decisive) and less self-consistent (0.76, below the 0.80 bar). With size taken
away, Luna has weak preferences among these creatures; size was most of what it judged confidently.
Felix's own picks agree 88% across the two versions. Decision: keep the true-size two-view format
(pool2, `curate_dual`) and distil as is.

### Resolution test (2026-10-18): resolution is not the confound

Felix's hypothesis: small creatures look blurry when zoomed (too few cells upscaled), so "Luna prefers
big creatures" might be "Luna prefers creatures it can see clearly". Test: the 87 held-out creatures
simulated at 2R (R = 40; rule rescaled, pattern upsampled; all passed the 2x check, so the same
creature) purely for the pictures (`pool rerender --upscale 2`), in both formats. Rule fixed beforehand:
resolution is the confound if the sharp size-removed images bring Luna back to >= 0.80 order agreement
without a strong size preference, or if the sharp true-size images clearly lower the size preference.

| held-out, Luna | order agr. | confidence | decisive | picks larger / heavier / faster | weights: speed, size, mass |
|---|---|---|---|---|---|
| two-view (pool2) | 0.89 | 0.53 | 0.54 | 0.85 / 0.84 / 0.65 | +0.32, +0.68, +0.45 |
| two-view, sharp (pool-2x) | 0.86 | 0.55 | 0.58 | 0.87 / 0.82 / 0.63 | +0.26, +0.56, +0.54 |
| size removed (pool-body) | 0.76 | 0.28 | 0.16 | 0.65 / 0.66 / 0.62 | +0.17, -0.18, +0.41 |
| size removed, sharp (pool-body-2x) | 0.77 | 0.30 | 0.20 | 0.64 / 0.63 / 0.60 | +0.17, -0.22, +0.48 |

Sharpness changes almost nothing: the size preference is about size itself, and without it Luna stays
unsure whether or not small creatures are sharp. Felix on the sharp size-removed images: weights speed
+0.81, size +0.54, texture -0.91; agreement with Luna 0.62. Decision: keep pool2 (R = 20, true-size
two-view, `curate_dual`). The image-format question is closed.

Next: prompt arm on pool2's held-out pairs (`curate_dual_nuanced`, `lifelike_dual`, `elegant_dual`)
and the first student.

## Phase 3: the student (built 2026-10-18)

`src/lenia/distill.py`, `lenia distill embed | train`:
- **embed** (PyTorch; Colab section 7 or `uv run --extra gpu` on the Mac): a frozen DINOv2 ViT-S/14
  over every pool image; two-view images are split into the motion view and the close-up block,
  each embedded on its own (384 + 384 dims), so the student sees motion and shape separately.
- **train** (numpy, anywhere): Bradley-Terry reward on the standardised embedding (linear by default,
  small MLP optional), soft targets = the teacher's order-averaged p_a on training pairs, L2 chosen on a
  creature-level validation split, 5-member bootstrap ensemble (mean reward and spread, for
  uncertainty-aware search later).
- **evaluation**, held-out pairs only (creatures never seen in training): agreement with the teacher
  (all, decisive, size-matched), with Jev, with Felix; against three baselines fitted or applied the
  same way: "larger wins", "heavier wins", and a Bradley-Terry reward on the five hand-made statistics
  (speed, change, size, mass, texture).

**The bar is high.** Dry run with stand-in embeddings: the five-statistic baseline already agrees with
Luna on 89% of held-out pairs (99% of decisive ones, 71% of size-matched ones), above Luna's own
order agreement (86%), which is roughly the ceiling for any scorer of a noisy teacher. "Larger wins"
alone reaches 85%. So the image student can only show it learned something beyond the statistics on
the size-matched pairs, and in the search phase: a feature reward and an image reward can be
exploited in different ways, which is itself an RQ1 comparison.

### Prompt arm, first look (2026-10-18)

Luna on pool2's 657 held-out pairs under four prompts (all on the two-view images). "same as curate" =
same winner as `curate_dual`; rank corr. = Spearman of the Bradley-Terry scores of the 87 held-out
creatures with `curate_dual`'s.

| prompt | order agr. | confidence | decisive | agrees with Felix | picks larger / faster / smoother | same as curate | rank corr. | size-matched: picks faster |
|---|---|---|---|---|---|---|---|---|
| curate_dual | 0.89 | 0.53 | 0.54 | 0.70 | 0.85 / 0.65 / 0.57 | 1.00 | 1.00 | 0.77 |
| curate_dual_nuanced ("not by how large it appears") | 0.85 | 0.53 | 0.55 | 0.72 | 0.85 / 0.62 / 0.54 | 0.94 | 0.99 | 0.70 |
| lifelike_dual | 0.84 | 0.36 | 0.29 | 0.70 | 0.85 / 0.62 / 0.55 | 0.91 | 0.97 | 0.61 |
| elegant_dual | 0.78 | 0.27 | 0.12 | 0.62 | 0.70 / 0.55 / 0.60 | 0.75 | 0.65 | 0.50 |

- Telling the curator to set size aside changes nothing: same ranking (0.99), same size preference.
  The instruction is accepted in words and ignored in the choices.
- "Lifelike" gives the same ranking with less confidence. Only "elegant" ranks differently (0.65),
  with a weaker size pull, but Luna is unsure under it (12% decisive) and it agrees less with Felix.
- So far, on this pool, the wording barely steers the teacher: one dominant axis (size, then motion)
  under every prompt that keeps Luna confident. Whether a more diverse pool gives the prompts more to
  work with is open (Felix's argument).

### First student (2026-10-18)

DINOv2 ViT-S/14 embeddings of pool2 (motion view and close-ups, 768 dims), Bradley-Terry head on
Luna `curate_dual`, 2,733 training pairs over 346 creatures, tested on 657 pairs between 87 unseen
creatures:

| scorer | Luna (all) | Luna (decisive) | Luna (size-matched, 100) | Jev | Felix (60) |
|---|---|---|---|---|---|
| student, linear | 0.92 | 1.00 | **0.84** | 0.86 | 0.63 |
| student, MLP | 0.92 | 1.00 | 0.82 | 0.87 | 0.70 |
| five statistics | 0.89 | 0.99 | 0.71 | 0.81 | 0.62 |
| larger wins | 0.85 | 0.97 | 0.58 | 0.80 | 0.60 |
| heavier wins | 0.84 | 0.96 | 0.61 | 0.75 | 0.55 |

- The student reproduces the teacher on unseen creatures about as well as the teacher reproduces itself
  across the two orders (0.89), and agrees with Jev as often as Luna does (0.88).
- It beats the five statistics where it matters: on size-matched pairs 0.84 vs 0.71 (100 pairs, so about
  +/- 0.09): the image model picked up part of what Luna judges beyond size.
- Its held-out top 12 are large structured creatures (fringed discs, dumbbells, jellyfish-like forms);
  its bottom 12 are the smallest creatures, including the faint crescents.

## Phase 4: search against the student (built 2026-10-18)

`src/lenia/rsearch.py`, `lenia rsearch run | check | rate | draw`:
- **run** (Colab, GPU; no API calls): 16 lineages founded by pool2 creatures spread evenly across the
  student's score range (the same founders for every reward). Each generation every lineage tries 3
  mutants started from its current creature; a mutant must be alive at 100 units and coherent at 2x
  (descriptor distance and texture loss, the survey's calibrated thresholds), is then rendered in
  the pool's image format (two-view) and scored. The best-scoring viable mutant replaces its parent
  if it scores higher and is still alive at 200 units (greedy hill climbing). Rewards: `student` (the
  distilled image reward, ensemble mean) or `features` (the five-statistic reward fitted on the same
  Luna labels). Every accepted creature is stored with its image, rule, pattern, both reward scores,
  the student's ensemble spread and its statistics. Resumable.
- **check** (Mac): Luna (`curate_dual`) and Jev on founder vs the lineage's creature at checkpoints
  (generations 5, 10, 20, 40, 60), both orders. The curve per checkpoint: mean student gain, mean
  feature gain, the teacher's P(new beats founder), and the median size, mass and speed. This is the
  RQ1 measurement: does the teacher agree with the reward as search pressure grows?
- **rate**: you rate founder vs final per lineage.

Predictions written down before the runs: (1) under the student, size and mass climb toward the
alive check's limits (radius of gyration 2.5 R, occupancy 25%); (2) the teacher agrees with early
gains and less with later ones (over-optimisation); (3) the student and the feature reward drift in
different directions once size saturates, since the student also learned something beyond the
statistics.

Dry run on real creatures (CPU, 4 lineages, 2 generations, features reward): works end to end.
Estimated cost on a T4: about an hour per 60-generation run.

### Phase 4 results (2026-10-18)

Both runs on Colab (about 1.8 h each), same 16 founders; teacher check on the Mac (`rsearch check`),
Felix rated founder vs final for the student run. Curve: `docs/img/phase4_curve.png`.

| gen | student run: gain | Luna p | Jev p | features run: gain | Luna p | Jev p |
|---|---|---|---|---|---|---|
| 5 | +0.43 | 0.56 | 0.61 | +0.21 | 0.57 | 0.58 |
| 10 | +0.64 | 0.61 | 0.61 | +0.42 | 0.63 | 0.66 |
| 20 | +0.93 | 0.66 | 0.69 | +0.69 | 0.66 | 0.67 |
| 40 | +1.12 | 0.69 | 0.73 | +0.94 | 0.71 | 0.70 |
| 60 | +1.22 | 0.72 | 0.76 | +1.12 | 0.76 | 0.76 |

(gain = the run's own reward; p = P(new beats founder), unchanged lineages count as 0.5.)

Against the predictions:
1. **Size and mass climb to the alive limits: no (student), partly (features).** Median ratio to
   founder at gen 60: student run size x1.11, mass x1.12, speed x1.68, change x1.63; features run size
   x1.35, mass x1.45, speed x1.47, change x1.99. Neither run got near the limits, but the features
   run's viable mutants fell from about 24 to 10 of 48 per generation while the student run's stayed
   at 23-27: the features reward pushes toward the constraint edge, the student does not.
2. **The teacher agrees less with later gains: no, at this pressure.** Agreement rises monotonically
   in both runs. The student is a calibrated proxy: since its scores are Bradley-Terry logits,
   sigmoid(gain) predicts Luna's p, and it overestimates by a steady 0.03-0.05 (0.63 vs 0.57 at gen 5,
   0.76 vs 0.72 at gen 60); the gap does not grow. Jev, which the student never saw, agrees as much
   (0.76) and tracks the student's per-lineage gains (r = 0.81 at gen 60), so the gains are not Luna
   quirks. 60 generations of greedy climbing (student gain +1.2 logits) is mild pressure: this is a
   lower bound, not evidence that the student cannot be over-optimised.
3. **The two rewards drift differently: yes, physically; the teacher cannot tell them apart.** The
   features reward bought its approval mostly with size and mass; the student with motion and
   texture at modest size. Luna and Jev end at the same p for both (0.72-0.76). Where size grew more
   than 5% Luna's mean p is 0.76-0.79; where it did not (4 student lineages, 3 features), 0.61-0.62.
   One features lineage went backwards for the teacher (L03, p = 0.30: mass x0.68, size x1.25, change
   x9, the features reward accepted a thinner, spread-out body); the student run's worst was 0.49.
   The student also predicts 0.70 for the features run's final creatures, against Luna's 0.76: it
   works as an evaluator for a search it did not drive.

**Felix vs the teacher (student run, founder vs final, 16 pairs):** Felix preferred the final
creature in 10/16, Luna in 15/16 (consistent with the 0.63-0.70 Felix-Luna agreement on pool2). The 6
Felix rejected include the two biggest size gains (L06 x1.73, L07 x1.69) and creatures that converged
on a round spiral/ring disc (L13; L08's ring texture). Luna's p does not predict Felix's choice (mean
0.74 on the ones he rejected, 0.71 on the ones he kept). One rater, one pass, 16 pairs: anecdotal,
but the visible divergence is teacher vs human, not student vs teacher.

**Visual:** several lineages in both runs converge on the same attractor, a round disc with spiral or
ring texture (student L09, L13; features L09, L12, L13, where jellyfish body plans dissolve into
speckled discs). Greedy search loses body-plan diversity; that is the case for MAP-Elites.

Implications: (a) to measure RQ1 divergence the search needs more pressure (MAP-Elites with the
student as quality, thousands of evaluations, longer runs) and then the same teacher check; (b) the
image student is not better than the five-statistic reward at pleasing Luna, but it is safer (no
viability collapse, no backwards lineage) and transfers to Jev; (c) the more interesting gap is Luna
vs a human, which points back at the teacher's size preference and the prompt arm.

### Is the ensemble spread an early warning? (2026-10-18)

No, not as built. On the 657 held-out pool pairs, the spread of the 5 members' predicted differences does
not track the student's error against Luna (Spearman 0.02); the members barely differ (spread about
0.08 against a typical margin of 1.1): bootstrapped linear heads on frozen features are near-copies.
The 6% of pairs where members disagree on the sign are near chance (0.53 agreement vs 0.94), but those
are just the close calls. In the searches, spread does not predict where the student overestimates
Luna either (Spearman 0.04). What spread does show: the features-reward search moved creatures away from
the student's training data (median spread 0.066 -> 0.102) while the student-reward search did not
(0.066 -> 0.055). So uncertainty-aware search (LCB) is parked; Phase 5 logs a more direct measure,
the distance to the nearest pool creature in embedding space.

## Phase 5: MAP-Elites over size x speed (built 2026-10-18)

`src/lenia/qd.py`, `lenia qd run | check | rate | draw`. Why: the greedy search let both rewards win mostly
by growing (the teacher's first preference), the five-statistic reward pleased Luna as well as the
student did, and lineages collapsed onto one round disc. Here the map's axes are size (radius of
gyration) and speed, the teacher's first and second preferences, 10 x 10 log-spaced cells (size
0.35-2.5 R, speed 0.003-0.5 R per unit; bins about 1.2x wide in size). A creature can only take a cell
by being better at the same size and speed, so search pressure goes into whatever else the reward
learned, and every within-cell comparison is size-matched, where the student is weakest (0.84 held out).
Relation to ASAL (Kumar et al. 2024): their illumination fills a foundation-model embedding space with
diverse creatures and their targets are prompts ("a caterpillar"); here diversity comes from interpretable
axes and quality from a learned aesthetic judge, which is the part that is hard to write down.

- **run** (Colab): seeded with all 433 pool creatures (64 of 100 cells). Each iteration: 64 parents drawn
  uniformly from occupied cells, mutated, frozen alive check (as rsearch), rendered and scored; a viable
  mutant takes its cell if empty or better, after the alive-at-200 confirmation. Default 80 iterations
  (5,120 mutants, about 3 h on a T4). Logs both rewards, ensemble spread and nearest-pool distance for
  every insertion. Resumable.
- **check** (Mac): (1) within cells, first occupant vs the elite at checkpoints, and at the end binned by
  student gain; (2) 150 random pairs of final elites across the map (all, and same size row): does the
  student still rank creatures it never saw as the teacher does? Pairs for you: first occupant vs elite.
- **draw**: map.png (elite score per cell, seed vs final), elites.png (each cell's elite in place),
  progress.png (cells, mean elite, QD score).

Predictions written down before the runs:
1. Within cells, Luna's p falls below what the student predicts, more so for high-gain cells. Decision
   rule: a gap of 0.10 or more in the top third of gains = over-optimisation found (RQ1 positive); within
   0.05 everywhere = the student holds even at matched size.
2. Within cells, the features reward does worse with Luna than the student (it can only move mass,
   change and texture there). If it does as well, the student has not learned anything the five
   statistics miss, at least nothing Luna rewards.
3. Across the map, the student agrees with Luna on most random elite pairs (size still separates them),
   less on same-size-row pairs.
4. Coverage grows from 64 cells, but not into the large, fast corner (the alive check's limits).
5. Distance from the pool grows with iterations; if Luna's disagreement grows with it, that distance is
   the early-warning signal the ensemble spread is not.

### Phase 5 results (2026-10-18)

Both runs on Colab (80 iterations x 64 mutants, about 2.5 h each); teacher check on the Mac; Felix rated
24 first-vs-elite pairs from the student run (decided 15, skipped 9 as too close). GIFs of eight
representative cells: `runs/qd-gifs/` (`scripts/qd_gifs.py`).

| | student run | features run |
|---|---|---|
| cells filled | 64 -> 85 | 64 -> 90 |
| improved cells (replacements per cell, mean) | 75 (2.9) | 82 (3.4) |
| within-cell size / speed change (median) | x1.03 / x0.99 | x1.03 / x1.01 |
| within-cell mass / change (median) | x1.03 / x1.10 | x1.13 / x1.24 |
| other reward went down in | 37% of cells | 27% of cells |
| Luna p (elite beats first occupant), Jev p | 0.614, 0.601 | 0.571, 0.556 |
| student predicts | 0.612 | 0.545 |
| across the map: student / features agree with Luna (150 pairs) | 0.91 / 0.86 | 0.94 / 0.87 |

Against the predictions:
1. **Over-optimisation within cells: not found.** The student predicts Luna's within-cell win rate almost
   exactly at every checkpoint (0.612 vs 0.614 at the end) and in every gain tercile (top third, gains
   +0.61 to +1.45: 0.692 vs 0.690). Jev agrees (0.68 in the top third). The decision rule (gap of 0.10 in
   the top third) is far from met. Caveat: within-cell pressure stayed mild (mean gain +0.47, about 3
   replacements per cell); QD spreads its effort over the map.
2. **Features reward worse than the student within cells: in the predicted direction, not decisive.**
   Luna 0.614 vs 0.571 (difference 0.043, bootstrap 95% CI -0.003 to 0.088), Jev 0.601 vs 0.556 (CI
   -0.003 to 0.095). Where the two rewards point opposite ways (28 cells in the student run, 22 in the
   features run) Luna is near a coin flip either way (0.54 and 0.54).
3. **Across the map: yes, and the student beats the five statistics there.** On random pairs of final
   elites the student agrees with Luna 0.91 / 0.94, the features reward 0.86 / 0.87. Pooled over both runs,
   pairs only the student gets right vs only the features reward: 32 vs 14 for Luna (p about 0.01), 30 vs
   16 for Jev (p about 0.05). These are creatures neither reward was trained on, including the features
   run's own elites. Same-size-row pairs: too few (15-17) to read.
4. **Coverage stops short of the large, fast corner: wrong.** Both runs filled it (a fast, curved,
   holed band; the onion-skin image makes it look like a lattice).
5. **Distance from the pool as an early warning: no.** Found elites sit at cosine distance about 0.35
   from the nearest pool creature (seeds 0), but the distance does not predict where the student
   overestimates Luna (Spearman 0.02 and -0.03).

**Felix:** elite beats first occupant in 10 of 15 decided pairs; agreement with Luna, Jev and the student
0.67 each; Luna's mean p is 0.65 on the pairs he accepted and 0.55 on the ones he rejected. 9 of 24 skipped:
many within-cell changes are too small to call by eye.

**In motion (GIFs):** each reward has one visible failure. The student's small slow elite (cell 2,3) sheds
loose debris specks (the Phase 1 speckle failure creeping back; Luna still gives it 0.62). The features
reward's huge chain (cell 9,5) breaks into separate bodies within 50 units of replay after passing the
200-unit alive check (mass pressure at the constraint's edge). Onion-skin stills make single fast creatures
look like segmented chains, for the judges as well as for us.

Reading: the distilled student is a faithful proxy for its teacher under both greedy and QD search at the
pressures tried, and it transfers to unseen creatures better than the five-statistic reward; what it adds
beyond the statistics is real but modest. Over-optimisation has not been reached; the stress test needs
much more pressure on fewer targets (long single-cell climbs or adversarial search against the student).

### How far from the pool did the search get? (2026-10-19)

Not far. Distances are cosine distances in the student's standardised DINOv2 embedding:

| | median distance to nearest neighbour |
|---|---|
| pool creature to its nearest other pool creature (leave one out) | 0.31 (p90 0.53; Chan 0.34, search 0.26) |
| found elites, student run / features run | 0.35 / 0.36 (p90 0.53) |
| two random pool creatures | 1.02 |

Found creatures sit about as far from the pool as pool creatures sit from each other: siblings, not new
species. About 10% land beyond the pool's own 90th percentile. Ancestry: final elites are a median of 4
mutations (max 12-15) from a pool creature; distance grows with depth only slowly (r about 0.4). The 85-90
final elites descend from just 35 (student) / 28 (features) pool creatures; one two-armed spinner
ancestor alone fills 13 (p0313, student run) / 19 (p0305, features run) cells. The farthest elites
(`farthest.png` in the chat, cells 8,1 7,2 5,0 6,0 / 2,9 2,6 2,5 7,1) are visibly the same species as their
ancestor (bigger, extra rings, rims). So "no over-optimisation" and "transfers to unseen creatures" are
results about near neighbours of the training set.

Why: a mutation (`explore.mutate`) nudges m (sigma 0.01), s (x e^0.08), r, h of about half the kernels and
occasionally one ring, and starts the child from the parent's pattern. It never changes the number of
kernels, the channel wiring (c0 -> c1), or combines two creatures. All creatures in the maps are Chan
multi-channel types: (15 kernels, 3 channels), (9, 3), (4, 2). Even these small steps kill half the
children (alive at 100 units 0.49, fully viable 0.40). Selection (alive check + reward + an incumbent
elite per cell) pulls survivors back toward known good shapes.

### Next: mutation-operator pilot (planned 2026-10-19; built, not yet run)

Question: which variation operators reach genuinely new creatures while staying viable? Turning up the
current noise alone is expected to mostly kill (Lenia creatures live on narrow parameter islands).
Pilot (Colab, about 15-20 min, no API calls): 64 parents from the QD maps x each operator, record the
viable fraction and the embedding distance of survivors to their parent and to the pool.
- current mutation at scale 1x, 2x, 4x
- pattern mutation: same rule, new start (reseed from the seed bank, flip/cut, or merge two creatures'
  patterns); one rule can host several creatures, and the rule stays viable
- structural: add / remove / duplicate a kernel, rewire c0 -> c1; followed by a short local re-tune of
  m and s (a few small-mutation steps) before judging
- crossover: swap kernel subsets between two parents with the same (kernels, channels) layout
Decision rule (written before running): an operator is worth using if at least 10% of children are viable
and the median viable child is beyond 0.5 from the pool (about the pool's p90). Then: a MAP-Elites run
with those operators and an embedding-novelty axis (e.g. size x a DINOv2 principal direction, or a
novelty bonus), seeded from the current maps, followed by the same teacher check: the real RQ1 stress test
on creatures unlike the training set.

**Built** (`src/lenia/operators.py`, `src/lenia/pilot.py`, `lenia pilot run | report`, Colab section 10).
Choices made while building, so the plan above stays the pre-registration:
- Parents: 64 final elites, 32 from each map (qd-student, qd-features); all 12 operators apply to all 64
  (35 have 15 kernels, 20 have 9, 9 have 4; all multi-channel). One child per parent per operator.
- A `copy` row (the parent unchanged, through the same checks) as the baseline: noise floor of the distances,
  and how often an elite passes the checks a second time. Its verdict is "baseline", not pass/fail.
- Pattern: `reseed` (a seed-bank start at R = 20, random channel strengths), `cut` (half-plane through the
  centre of mass wiped, soft edge), `merge` (partner with the same channel count, turned/mirrored, overlaid by
  the cell-wise maximum). Flips were left out: Lenia is rotation- and mirror-equivariant up to the grid, so a
  flipped start is the same creature.
- Structural: `add` (a mutated copy of a random kernel with a random source channel, at a quarter of the
  template's weight), `remove` (only if the target channel keeps another writer), `duplicate` (split a kernel's
  weight between it and a copy, then mutate the copy at 2x; the split alone is the same rule, tested), `rewire`
  (one kernel's c0, or c1 if the old target keeps another writer). No channel is ever left without a writer
  (it would freeze). Re-tune: up to 2 rounds x 3 tries (the unchanged edit first, then m/s nudges at half the
  usual step); the first alive at 100 units goes on to the full check.
- Crossover: same layout (structure and T); per target channel, all its kernels from one parent (at least one
  channel from each), since a channel's update depends only on the kernels that write to it.
- Viable = the full frozen check, including alive at 200 units (as an elite must be). Distances as qd's `nn`:
  cosine in the student's standardised DINOv2 embedding; "to parent" is to the nearer of two parents for merge
  and crossover. The report also gives the pool's leave-one-out distribution, the stage at which children die,
  re-tune cost, and the student's score change. Viable children's patterns are kept (`patterns.npz`) so the
  next MAP-Elites run can be seeded from them.

## Catalogue rankings: student vs teacher (built 2026-10-20, not yet run)

`src/lenia/rank.py`, `lenia rank build | label | report`, Colab section 11. What does each consider most worth
including, and, with no search involved, does the student rank creatures as the teacher does?
- Sets, each rendered as the pool (two-view, R = 20, 256 x 256), ranked separately (pairs only within a set):
  `found` = all 644 living creatures of found-survey-v2 (257 in pool2: 208 train, 49 held-out; 387 never in
  the pool; 521 pass the calibrated 2x/texture checks, flagged); `classic` = Chan's animals.json, one channel and
  one kernel (548 species, 487 fit at R = 20), run 100 units from their own pattern, the living ones kept,
  drawn grey on black (`--classic-cmap`). Classic vs found differs in image style as well as creature type,
  so a gap there cannot be pinned on either alone.
- Teacher: 8 partners per creature (about 16 comparisons each), both orders, Bradley-Terry. Split-half
  reliability (Spearman of rankings from two random halves of the pairs, Spearman-Brown corrected) is the
  ceiling for any scorer's agreement with it.
- Report: per group (found: pool train / pool held-out / never in pool; classic): student, five-statistic
  reward and "larger wins" against the teacher, as rank correlation over creatures and winner agreement on
  decisive pairs (|p - 0.5| > 0.25), plus rank correlation with size taken out (residuals after regressing each
  score on log size). Sheets: top and bottom 12 per scorer, raw and size-adjusted; `leaderboard.csv`.
- Cost: about 9,000 pairs, 18,000 Luna calls; at the vlmtest rate (2,500 calls about $0.15) roughly $1-3,
  above the judge's default $2 budget guard, so `--yes`. Smoke test first with `--limit 20`.

## Open issues (as of 2026-10-17; see "Where things stand" at the top for the current state)

**Viability (the 'alive' judge)**
- Near-static creatures pass as alive (motion/change thresholds 0.02 / 0.05 R per unit); the VLMs and
  Felix penalise them. Decision (2026-10-17): leave them legal. "Alive" is physical validity (exists,
  one body, persists, coherent across resolutions), not taste; whether the aesthetic reward pushes
  toward motion is then a measurable trade-off rather than a rule hidden in the constraint.
- Splitting creatures (classed "multi") stay excluded for now: one creature per image keeps pairwise
  judging clean. A possible later extension.
- The search piles up just under the 2x-check threshold; a few novel creatures still collapse into one
  shared smooth body at 2x (#23, #144). Options: stricter threshold, a 3x check, or search at higher R.
- The 2x check rests on a coarse descriptor (about 9% of catalogue species fail it): a filter, not proof.
- Root cause is resolution: at R = 13 multikernel creatures are 10-15 cells wide. Experiment A
  (R = 26, multi-channel from known creatures) fixes it instead of filtering it.

**Novelty and aesthetics (the other judges)**
- The 20 hand-made descriptors miss internal structure and behaviour; novelty search maximised what
  they can see (size, roughness), hence speckle. Experiment B: image-embedding novelty (CLIP / DINO).
- The `interesting` prompt nearly ties speckle with real creatures: a novelty-seeking VLM judge needs
  the hard constraints to stay strict.
- Luna's position bias is mostly fixed by `curate`; averaging Luna and Jev is a cheap fallback (0.88
  agreement with Felix under `appeal`).
- Part of the judges' penalty on speckle and R = 13 creatures may be pixelation from upscaling; a fair
  comparison renders everything at the same effective resolution (Experiment A again).
- Judges see stills: the fixed camera shows motion over a 6-unit window, nothing faster or slower.

**Search and substrate**
- Flow-Lenia probe (Tim): a better Lenia? how much data? what results?
- Experiment A runs on Colab (GPU port done; about 23x the CPU compute per step). The novelty search
  (`lenia explore`) is not resumable yet; needed before long Colab runs.
- The guided search is a greedy hill climber: it refines, it cannot find radically new body plans. For
  that: quality-diversity (MAP-Elites niches, VLM or distilled reward as quality within each niche),
  neutral drift (accept near-ties), structured mutations (add/remove kernels, crossover, multi-channel).
  QD is also where distillation pays: thousands of quality judgments per generation.
- Framing (Tim's pointer to mechanistic interpretability): search against a reward model is feature
  visualisation; speckle is its adversarial-noise failure, the resolution check its regulariser.

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
uv run lenia vlmtest build                                                  # VLM test: creatures, images, pairs
uv run lenia vlmtest run --limit 3                                          # smoke test both judges
uv run lenia vlmtest run [--judges luna,jev] [--prompts appeal,beautiful]   # all calls (cached), then the report; default: the prompt sweep
uv run lenia vlmtest rate [--n 60]                                          # rate pairs yourself in the browser (resumable)
uv run lenia vlmtest report                                                 # redo the report (e.g. after rating)
uv run lenia guided run [--generations 50] [--lineages 16]                 # VLM-guided search (resumable); then:
uv run lenia guided check                                                   # held-out judges + drift report
uv run lenia guided rate                                                    # rate founder vs final yourself
uv run lenia found download                                                 # Chan's found/ collections -> data/found/
uv run lenia --backend auto --config configs/found.yaml found survey [--n-per-file 100]   # Experiment A: survey + calibration at R = 20
uv run lenia --backend auto --config configs/found.yaml explore --from-found runs/found-survey --roots 64 --out runs/expA   # step 2 (resumable)
uv run lenia --backend auto --config configs/found.yaml pool build                     # Phase 2 pool (then label/rate/report, see above)
uv run --extra gpu lenia distill embed --pool runs/pool2 && uv run lenia distill train --pool runs/pool2   # Phase 3 student
uv run lenia --backend auto --config configs/found.yaml rsearch run --reward student   # Phase 4 (Colab), also --reward features
uv run lenia rsearch check --out runs/rsearch-student                                  # Phase 4 teacher check (API keys)
uv run lenia --backend auto --config configs/found.yaml qd run --reward student       # Phase 5 MAP-Elites (Colab), also --reward features
uv run lenia qd check --reward student                                                   # Phase 5 teacher check (API keys); qd rate, qd draw
uv run lenia --backend auto --config configs/found.yaml pilot run                       # mutation-operator pilot (Colab); pilot report anywhere
uv run lenia --runs "$RUNS" rank label --judges luna --limit 20                         # catalogue rankings: build on Colab, label (API keys), report
uv run lenia trace --run runs/survey-classic --rules 18,141            # a rollout's future: mass, size, speed, thumbnails
uv run python scripts/calibrate_growth.py <explore runs...>           # growth-threshold calibration
uv run python scripts/calibrate_rescale.py                            # resolution-check calibration (catalogue)
uv run lenia explore --from ... --rescale-time 0                             # search without the resolution check
```

Runs above were made on a 2-core cloud machine; on a laptop with more cores they are proportionally
faster (`--workers`, default all cores but one).
