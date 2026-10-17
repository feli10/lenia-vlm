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

Verification so far: PyTorch could not be installed in the build sandbox, so the PyTorch code path
was tested against a numpy-backed stand-in for the PyTorch API (identical results; whole test suite
passes through it), which checks the logic, not PyTorch itself. The real check is `lenia bench` on
Colab. CPU reference timings (ms per creature per step, one process): classic 128x128 0.7,
multikernel 256x256 R=26 5.8, multichannel 256x256 R=26 29.6.

## Open issues (as of 2026-10-17)

**Viability (the 'alive' judge)**
- Near-static creatures pass as alive (motion/change thresholds 0.02 / 0.05 R per unit); the VLMs and
  Felix penalise them, but raising the thresholds would close the loophole outright.
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
uv run lenia trace --run runs/survey-classic --rules 18,141            # a rollout's future: mass, size, speed, thumbnails
uv run python scripts/calibrate_growth.py <explore runs...>           # growth-threshold calibration
uv run python scripts/calibrate_rescale.py                            # resolution-check calibration (catalogue)
uv run lenia explore --from ... --rescale-time 0                             # search without the resolution check
```

Runs above were made on a 2-core cloud machine; on a laptop with more cores they are proportionally
faster (`--workers`, default all cores but one).
