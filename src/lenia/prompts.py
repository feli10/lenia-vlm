"""Every judge prompt of the live pipeline, in one place.

Pairwise duels: {A} and {B} name the two images ("the first image" / "the second image" for the OpenRouter
Decisions API, the state keys `first` / `second` for Jev). Multiple choice: {W} names the image, {ABOUT} the
aspect asked about. Ids are stable and the full text goes into the judge cache key, so editing a prompt
invalidates its cached answers; tests/test_golden.py fails on any change to these texts.

The four "You are shown ... worlds from a mass-conserving variant of Lenia" preambles differ slightly in wording.
That is historical, and kept on purpose: unifying them would change the text and so every cache key.

Gathered here verbatim on 2026-10-28 from lenia/mace.py (complexity), scripts/surprise_check.py (surprise),
scripts/goal_session.py (interest, steer, attain, session menus). verify_prompt added 2026-10-29 (gate 0d).
The library-v3 menus of the goal-matching experiment are in archive/scripts/goal_groups.py.
The classic-Lenia creature prompts (appeal, curate, dual views, radical arm) are in archive/src_lenia/prompts_classic.py.
"""
from __future__ import annotations

PROMPTS: dict[str, dict[str, str]] = {}


# ---------- pairwise duels ----------

# Map entry (complexity duel vs the cell's incumbent) and the expedition floor.
COMPLEXITY = "evolve_complexity_world"
PROMPTS.setdefault(COMPLEXITY, {
    "instructions": ("You are shown two worlds from a mass-conserving variant of Lenia, a continuous cellular automaton "
                     "used in artificial-life research. Each image shows one whole world as four snapshots taken a few "
                     "time apart, long enough to show movement and change, in reading order (top-left, top-right, "
                     "bottom-left, bottom-right); the colours are "
                     "three interacting chemical channels. The two images are {A} and {B}. We are running an "
                     "evolutionary process whose goal is to reach more complex, life-like structure over many "
                     "generations. Which world would you choose as the parent for the next generation?"),
    "first": "I would choose the world in {A} as the parent.",
    "second": "I would choose the world in {B} as the parent.",
})


# Aim-conditioned map entry (map-prompt test, 11-04; handoff "Stage 1 aim-conditioned goals", road map step 2): the
# entry duel with its goal sentence replaced by the user's aim. Every other word is COMPLEXITY's except its typo "a few
# time apart", which becomes "some time apart" (the surprise duel's wording; Felix 11-04: this prompt goes on into the
# full run, so the typo isn't carried forward; COMPLEXITY keeps it, since its caches and runs depend on its text).
# Registered per aim; the id hashes the full instruction text. Felix 11-03: "complex" may reward texture.
def map_prompt(aim: str) -> str:
    import hashlib
    head = PROMPTS[COMPLEXITY]["instructions"].replace("taken a few time apart", "taken some time apart")
    head = head[:head.index("The two images are {A} and {B}.") + len("The two images are {A} and {B}.")]
    text = (head + " We are running an evolutionary process over many generations, aimed at this, in the user's words: "
            f"\"{aim}\". Which world would you choose as the parent for the next generation?")
    pid = "map_aim_" + hashlib.sha1(text.encode()).hexdigest()[:12]
    PROMPTS.setdefault(pid, {"instructions": text, "first": PROMPTS[COMPLEXITY]["first"],
                             "second": PROMPTS[COMPLEXITY]["second"]})
    return pid

# The surprise detector (validated: archive/scripts/surprise_check.py validate, runs/surprise-val).
SURPRISE = "surprise_unusual_interesting"
PROMPTS.setdefault(SURPRISE, {
    "instructions": ("You are shown two worlds from a mass-conserving variant of Lenia, a continuous cellular automaton "
                     "used in artificial-life research. Each image shows one whole world as four snapshots taken some time "
                     "apart, in reading order (top-left, top-right, bottom-left, bottom-right); the colours are three "
                     "interacting chemical channels. The two images are {A} and {B}. Which world shows something more "
                     "unusual and interesting: a distinct feature or behaviour that stands out, rather than just more or "
                     "less of the same, a different size, or a different colour?"),
    "first": "The world in {A} shows something more unusual and interesting.",
    "second": "The world in {B} shows something more unusual and interesting."})


# Goal-session interest check (held-out comparisons of goal vs free branches).
WORLD = ("You are shown {N} from a mass-conserving variant of Lenia, a continuous cellular automaton used in "
         "artificial-life research. Each image shows one whole world as four snapshots taken some time apart, long "
         "enough to show movement and change, in reading order (top-left, top-right, bottom-left, bottom-right); the "
         "colours are three interacting chemical channels.")
INTEREST = "goal_session_interest"
PROMPTS.setdefault(INTEREST, {
    "instructions": WORLD.replace("{N}", "two worlds") + " The two images are {A} and {B}. Which world is more "
                    "interesting as artificial life: richer structure or behaviour, more like something alive, more "
                    "worth exploring further?",
    "first": "The world in {A} is more interesting.", "second": "The world in {B} is more interesting."})


# Goal prompts, registered per goal: steer drives an expedition; attain is the held-out check (Luna).
def steer_prompt(g: dict) -> str:
    pid = f"goal_steer_{g['id']}"
    PROMPTS.setdefault(pid, {
        "instructions": WORLD.replace("{N}", "two worlds") + " The two images are {A} and {B}. We are steering an "
                        f"evolutionary process toward a goal. The world it started from showed this: {g['sees']}. "
                        f"The goal: {g['push']}. Which world would you choose as the parent for the next generation, "
                        "to move toward this goal?",
        "first": "I would choose the world in {A} as the parent.",
        "second": "I would choose the world in {B} as the parent."})
    return pid


def attain_prompt(g: dict) -> str:
    pid = f"goal_attain_{g['id']}"
    PROMPTS.setdefault(pid, {
        "instructions": WORLD.replace("{N}", "two worlds") + " The two images are {A} and {B}. Both descend from a "
                        f"world that showed this: {g['sees']}. We wanted this change: {g['push']}. Which world shows "
                        "this change more clearly?",
        "first": "The world in {A} shows it more clearly.", "second": "The world in {B} shows it more clearly."})
    return pid


# Verify (gate 0d, Stage 1 logs it): given a description of one world, which of two worlds does it describe? The true
# world meets a near decoy (same lineage) and two random map worlds, both orders. Registered per description; the id
# hashes the text, so library entries and VLM-written names share one scheme.
def verify_prompt(sees: str) -> str:
    import hashlib
    pid = "verify_" + hashlib.sha1(sees.encode()).hexdigest()[:12]
    PROMPTS.setdefault(pid, {
        "instructions": WORLD.replace("{N}", "two worlds") + " The two images are {A} and {B}. Someone looked at one "
                        f"of these two worlds and described what it shows: \"{sees}\". Which world were they describing?",
        "first": "They were describing the world in {A}.",
        "second": "They were describing the world in {B}."})
    return pid


# ---------- multiple choice: matching a world to a goal-library entry (goal sessions, arm L) ----------

NONE = "none"


# The five groups a goal session steers by, and its matching instructions (goal_session.py).
STEER = {"interior": "the inside of the main body", "shape": "the overall shape and size of the main body",
         "surroundings": "what is around the main body",
         "bodies": "how many bodies there are and how they relate to each other",
         "change": "how the world changes between the snapshots"}
SESSION_MATCH = (WORLD.replace("{N}", "one world").replace("Each image shows", "The image {W} shows") +
         " Looking only at {ABOUT}, which description best matches this world? If none of them fits, choose "
         "\"None of these fits.\"")


# ---------- generative VLM (lenia.describe): system text, user template with {fields}, JSON fields returned ----------

GEN: dict[str, dict] = {}

_GEN_SYSTEM = ("You help run an artificial-life experiment. We evolve worlds of a mass-conserving variant of Lenia, a "
               "continuous cellular automaton: each world is a 160 x 160 grid with three interacting chemical channels, "
               "shown as colours, and every world grows from the same small square of noise. A cheap judge model, which "
               "can only choose between two pictures, does the frequent work; you are asked only for rare decisions "
               "that need words. Answer with one JSON object and nothing else.")
_FRAMES = ("Each image shows one whole world as four snapshots taken some time apart, long enough to show movement and "
           "change, in reading order (top-left, top-right, bottom-left, bottom-right).")

# Arm N: name what is new about a trigger world, as a goal the cheap judge can verify and steer by (sees -> verify and
# the steer/attain prompts' "the world it started from showed this"; push -> "the goal").
NAMER = "namer_v1"
GEN.setdefault(NAMER, {
    "system": _GEN_SYSTEM + " The cheap judge has flagged one world as surprising compared with what the run usually "
              "produces. Your description will be used by the cheap judge to pick this world out from others, and to "
              "steer its descendants.",
    "user": ("Image 1 is the flagged world. Images 2-5 are typical worlds from the same run, for contrast. " + _FRAMES +
             "\n\nReturn {\"sees\": ..., \"push\": ...}:\n"
             "- sees: what Image 1 shows that the typical worlds do not: one concrete, visible feature or behaviour that "
             "sets it apart, described so that someone could point to it in the picture (one sentence, at most 25 words). "
             "Not just a different colour, size or amount of the same thing.\n"
             "- push: one change to its descendants that would develop that feature further, as an instruction starting "
             "with \"make\" (one sentence, at most 20 words). It must be visible in snapshots like these.\n\n"
             "Descriptions already given to other worlds in this run (do not reuse them; if Image 1's most distinctive "
             "feature is one of these, describe a different one):\n{names}"),
    "fields": ["sees", "push"]})

# Aim-conditioned namer (Stage 1, 11-04; project doc "Handoff - Stage 1 aim-conditioned goals"). The user's open-ended
# aim becomes an input: the push is the change to the "sees" feature that would most move the world toward the aim.
# What changed from namer_v1 and why (v4 ratings, 11-03: N's goals on the floor triggers lost 0-6 on interest to the
# library's, and all asked for "more of it": spawn, bud, split, a larger colony):
# - v1's push had to "develop that feature further", which reads as "more of it"; it now serves the aim.
# - v1's "It must be visible in snapshots like these" read as "static"; it is now stated as a fact about the system
#   (the judge compares four snapshots, which show shapes and how they change over time), not as a rule.
# - No list of banned goal types (Felix 11-03: a change of approach, not patches). Kept from v1: the contrast with
#   typical worlds, the concrete "sees", the "make ..." push and its length (the attain prompt and the captions read
#   it), and the run's earlier names (so triggers get different features).
AIM_ALIFE = "more interesting as artificial life: richer structure or behaviour, more like something alive"
NAMER_AIM_V1 = "namer_aim_v1"
GEN.setdefault(NAMER_AIM_V1, {
    "system": _GEN_SYSTEM + " The cheap judge has flagged one world as surprising compared with what the run usually "
              "produces. Your description will be used by the cheap judge to pick this world out from others, and to "
              "steer its descendants toward the aim of the experiment.",
    "user": ("The aim of this experiment, in the user's words: \"{aim}\"\n\n"
             "Image 1 is the flagged world. Images 2-5 are typical worlds from the same run, for contrast. " + _FRAMES +
             "\n\nReturn {\"sees\": ..., \"push\": ...}:\n"
             "- sees: what Image 1 shows that the typical worlds do not: one concrete, visible feature or behaviour that "
             "sets it apart, described so that someone could point to it in the picture (one sentence, at most 25 words). "
             "Not just a different colour, size or amount of the same thing.\n"
             "- push: the change to that feature that would most move this world toward the aim, as an instruction "
             "starting with \"make\" (one sentence, at most 20 words).\n\n"
             "The cheap judge will compare descendants with Image 1 using four snapshots like these, which show both the "
             "shapes in each snapshot and how they move or change from one snapshot to the next; it can follow a change "
             "only if that change shows in them.\n\n"
             "Descriptions already given to other worlds in this run (do not reuse them; if Image 1's most distinctive "
             "feature is one of these, describe a different one):\n{names}"),
    "fields": ["sees", "push"]})

# v2 (11-04, after v1's 12 goals): 9 of 12 pushes asked for buds, eggs or split-off pieces to grow into new organisms,
# swim away or form a colony. In MaCE-Lenia that cannot happen: mass is conserved, so a piece takes its mass from the
# body and stays small unless pieces merge again (Felix 11-04). The model was never told this. v2 adds the substrate's
# mass budget as a fact (_MASS, to be shared with the aim-aware review); nothing else changes, and no goal type is banned.
_MASS = ("How these worlds work: each chemical channel's total mass is fixed. A world never gains or loses mass; it only "
         "moves it around, and every world starts with the same amount. So when a body splits or something buds off, the "
         "pieces share the mass the body had. A piece cannot grow by itself: it grows only by merging with other pieces "
         "or drawing mass from them, and pieces that drift apart stay small.")
NAMER_AIM_V2 = "namer_aim_v2"
_u = GEN[NAMER_AIM_V1]["user"]
_cut = "\n\nReturn {"
assert _cut in _u
GEN.setdefault(NAMER_AIM_V2, {"system": GEN[NAMER_AIM_V1]["system"], "user": _u.replace(_cut, "\n\n" + _MASS + _cut, 1),
                              "fields": ["sees", "push"]})

# v3 (11-04, after v2): the "grow into new organisms" pushes were gone, but 8 of 12 goals still asked for division or
# detaching. Their "sees" was a division event: the triggers are surprise-flagged, and in these worlds a splitting
# event is often what stands out, so "sees = what is most distinctive" + "push develops sees" yields division. Where
# "sees" was a body (L55, L19, L46, L22), v2's goals were good. v3 separates what is distinctive from what to develop,
# as a Picbreeder user does: list 2-3 distinctive features, then build on the one with the most potential toward the
# aim. Everything else is v2's text.
NAMER_AIM = "namer_aim_v3"
_old_spec = GEN[NAMER_AIM_V2]["user"][GEN[NAMER_AIM_V2]["user"].index("Return {"):GEN[NAMER_AIM_V2]["user"].index("- push:")]
_new_spec = ("Return {\"features\": [...], \"sees\": ..., \"push\": ...}:\n"
             "- features: two or three things Image 1 shows that the typical worlds do not, each one concrete, visible "
             "feature or behaviour, described so that someone could point to it in the picture (at most 25 words each). "
             "Not just a different colour, size or amount of the same thing.\n"
             "- sees: the one of those features with the most potential toward the aim, given how these worlds work: "
             "the starting point the descendants will build on (copy or shorten it; one sentence, at most 25 words).\n")
assert _old_spec.startswith("Return {") and "- sees:" in _old_spec
GEN.setdefault(NAMER_AIM, {"system": GEN[NAMER_AIM_V2]["system"],
                           "user": GEN[NAMER_AIM_V2]["user"].replace(_old_spec, _new_spec, 1),
                           "fields": ["sees", "push"]})      # "features" (a list) passes through describe.parse as an extra key

# Arm N+R: one scheduled review of an expedition (continue / refine only in Stage 1).
REVIEW = "review_v1"
GEN.setdefault(REVIEW, {
    "system": _GEN_SYSTEM + " Here you review an expedition: the descendants of one world, steered toward a goal by "
              "the cheap judge.",
    "user": ("We are steering the descendants of a world toward a goal, one generation at a time. Each generation, "
             "{mutants} mutants of the current parent are made, and the cheap judge accepts the one that best moves "
             "toward the goal, if any. " + _FRAMES + "\n\nImage 1 is the world the expedition started from. Image 2 is "
             "the current parent, after {gens} of {total} generations.\nThe world it started from showed this: {sees}\n"
             "The goal: {push}\nSo far: {progress}\n\nDecide for the remaining generations:\n"
             "- \"continue\": keep the goal as it is.\n"
             "- \"refine\": keep what the start showed, but change the goal's direction.\n"
             "Return {\"decision\": \"continue\" or \"refine\", \"push\": the goal for the remaining generations "
             "(the same text if you continue; an instruction starting with \"make\", at most 20 words), "
             "\"why\": one short sentence}."),
    "fields": ["decision", "push", "why"]})


# Stage 1 review prompts (10-29, after gate 0b): review_v1 let "refine" drift into switching to a new feature about half
# the time. Refine and switch are now separate options, so arms differ by one component:
#   N+R: continue / refine (REVIEW_R);  N+S: continue / refine / switch (REVIEW_S). One review per expedition in Stage 1.
_REVIEW_HEAD = ("We are steering the descendants of a world toward a goal, one generation at a time. Each generation, "
                "{mutants} mutants of the current parent are made, and the cheap judge accepts the one that best moves "
                "toward the goal, if any. " + _FRAMES + "\n\nImage 1 is the world the expedition started from. Image 2 is "
                "the current parent, after {gens} of {total} generations.\nThe world it started from showed this: {sees}\n"
                "The goal: {push}\nSo far: {progress}\n\nDecide for the remaining {left} generations (the goal will not be "
                "reviewed again):\n"
                "- \"continue\": keep the goal as it is.\n"
                "- \"refine\": keep the same target, the feature the start showed, and rewrite the goal's direction so the "
                "judge can make progress on it. Do not change to a different feature, even one that has appeared since.\n")
_PUSH_SPEC = "an instruction starting with \"make\", at most 20 words"

REVIEW_R = "review_refine_v2"
GEN.setdefault(REVIEW_R, {
    "system": GEN[REVIEW]["system"],
    "user": (_REVIEW_HEAD + "Return {\"decision\": \"continue\" or \"refine\", \"push\": the goal for the remaining "
             f"generations (the same text if you continue; {_PUSH_SPEC}), \"why\": one short sentence}}."),
    "fields": ["decision", "push", "why"]})

REVIEW_S = "review_switch_v1"
GEN.setdefault(REVIEW_S, {
    "system": GEN[REVIEW]["system"],
    "user": (_REVIEW_HEAD + "- \"switch\": drop the target and pursue a different feature that has appeared in the "
             "current parent. Reaching anything takes commitment: switching throws away the progress made on the current "
             "goal, and only the remaining generations are left for the new one. Switch only if the new feature is clearly "
             "more promising than the current goal and can be developed within them.\n"
             "Return {\"decision\": \"continue\", \"refine\" or \"switch\", \"sees\": what the rest of the expedition "
             "builds on (the start's description unchanged, unless you switch; if you switch, the new feature in Image 2: "
             "one concrete, visible feature, at most 25 words), \"push\": the goal for the remaining generations (the same "
             f"text if you continue; {_PUSH_SPEC}), \"why\": one short sentence}}."),
    "fields": ["decision", "sees", "push", "why"]})


# Aim-aware review for expedition v4 (arm NS-aim, 11-04; handoff "Stage 1 aim-conditioned goals"): one review at
# step 5 of 10. Differs from review_switch_v1 in what it sees and what it judges by: the user's aim; the mass budget
# (_MASS); v4's step (children, shortlist, comma selection); the per-generation goal scores; and three other recent
# descendants (the most surprising unchosen judged children of the last three generations, by Jev's surprise score),
# so "switch" can move to a new stepping stone from the population instead of only re-reading the current parent.
# Kept from review_switch_v1: continue / refine / switch, one review per expedition, and switching stated as costly
# (Felix 10-29: commit to a goal for a while). The judge fact is the aim namer's.
REVIEW_AIM = "review_aim_v1"
GEN.setdefault(REVIEW_AIM, {
    "system": _GEN_SYSTEM + " Here you review an expedition: the descendants of one world, steered toward a goal by "
              "the cheap judge in service of the experiment's aim.",
    "user": ("The aim of this experiment, in the user's words: \"{aim}\"\n\n"
             "We steer the descendants of a world toward a goal, one generation at a time. Each generation, {children} "
             "mutants of the current parent are made, {judged} of them are judged against the expedition's reference "
             "world, and the one that best shows the goal becomes the next parent. " + _FRAMES + "\n\n" + _MASS + "\n\n"
             "Image 1 is the world the expedition started from. Image 2 is the current parent, after {gens} of {total} "
             "generations. Images 3, 4 and 5 are other recent descendants, not chosen, that the cheap judge found the "
             "most surprising.\nThe world it started from showed this: {sees}\nThe goal: {push}\n"
             "The cheap judge's goal score for each generation's parent against the start (0.5 = no visible change, "
             "1 = clearly further): {progress}\n\n"
             "Decide how to spend the remaining {left} generations (there is no later review). Judge by the aim, not "
             "only by the goal's score:\n"
             "- \"continue\": keep the goal.\n"
             "- \"refine\": keep building on the current parent's feature, with a new goal for it.\n"
             "- \"switch\": build on Image 3, 4 or 5 instead: that world becomes the parent, with its own feature and "
             "goal. Switching drops the progress made on the current goal and leaves only the remaining generations for "
             "the new one, so switch only if that world is clearly the more promising start toward the aim.\n\n"
             "The cheap judge compares descendants with the reference world using four snapshots like these, which show "
             "both the shapes in each snapshot and how they move or change from one snapshot to the next; it can follow "
             "a change only if that change shows in them.\n\n"
             "Return {\"decision\": \"continue\", \"refine\" or \"switch\", \"image\": 3, 4 or 5 if you switch "
             "(otherwise null), \"sees\": the feature the remaining generations build on (the description above, "
             "unchanged, unless you switch; if you switch, one concrete, visible feature of that image, at most 25 "
             "words), \"push\": the goal for the remaining generations (the same text if you continue; otherwise an "
             "instruction starting with \"make\", at most 20 words), \"why\": one short sentence}."),
    "fields": ["decision", "sees", "push", "why"]})      # "image" passes through describe.parse as an extra key

# Goal-progress judge for the generative VLM (goal check after arm L, 10-29): the attain question, asked of Gemini as a
# third judge beside Jev and Luna, to see which agrees with Felix's blind ratings.
GOAL_JUDGE = "goal_judge_v1"
GEN.setdefault(GOAL_JUDGE, {
    "system": _GEN_SYSTEM + " Here you compare two worlds of one lineage against a goal.",
    "user": ("Image 1 and Image 2 are two worlds; one descends from the other. " + _FRAMES + "\n\nThe lineage started "
             "from a world that showed this: {sees}\nThis change was wanted: {push}\n\nWhich image shows this change "
             "more clearly? Judge only the change asked for, not which world is nicer.\n"
             "Return {\"choice\": \"1\" or \"2\", \"why\": one short sentence}."),
    "fields": ["choice", "why"]})


# Goal captions (Stage 1 v4, 11-02): the CLIP shortlist (48 children -> CLIP keeps the 16 closest to a caption that pass
# the motion floor -> Jev picks) is used only for goals whose end state can be pictured. CLIP scores one still, the
# bottom-right snapshot, so the caption describes that still; "none" sends the goal to the motion floor + a random 16.
# "start" captions the same aspect of the start world, for directional (change) scoring: CLIP's image change start ->
# child against the text change start -> end (Gal et al. 2021, StyleGAN-NADA). Arm L's library goals get hand-written
# captions in the same form (data/goal_library_v3_captions.json). No colours: a colour caption would let the shortlist
# select channel colour, not structure.
CAPTION = "goal_caption_v1"
GEN.setdefault(CAPTION, {
    "system": _GEN_SYSTEM + " Here you turn one goal into short picture captions for CLIP, an image-text model that "
              "helps shortlist the descendants that look most like the goal's end state.",
    "user": ("Image 1 is the world an expedition starts from. " + _FRAMES + "\nWhat it shows: {sees}\n"
             "The goal for its descendants: {push}\n\nCLIP sees only the last snapshot (bottom-right) of each world, as a single "
             "still picture, and scores how well it matches a caption. Return {\"end\": ..., \"start\": ...}:\n"
             "- end: what the last snapshot would show once the goal is reached, as a plain caption of 3 to 8 words, a "
             "noun phrase like the caption under a photo. Describe shapes, parts and how they are arranged, in everyday "
             "words; no colours, no words about movement, speed or timing, no Lenia or chemistry terms. If what the goal "
             "asks for is movement, rhythm or a change between snapshots, so that a world that reached it would look no "
             "different in one still, answer \"none\".\n"
             "- start: a caption of the same kind for the last snapshot of Image 1, describing the same aspect the goal "
             "changes, so the two captions differ mainly in what the goal changes; \"none\" if end is \"none\"."),
    "fields": ["end", "start"]})


# Gallery pick (full run, 11-05; project doc "Handoff - full run build (seed 1)"): the last step of a run. Scores pick the
# shortlist (Bradley-Terry ratings from every aim entry duel, a Swiss tournament among the cell champions, then at most
# two per lineage and one per cell); the generative VLM chooses the final set from it, judging the set as a whole. The
# shortlist is shown in random order. In the spirit of Sam's Picbreeder score tracking; the VLM choosing a set is ours.
GALLERY = "gallery_pick_v1"
GEN.setdefault(GALLERY, {
    "system": _GEN_SYSTEM + " Here you choose the gallery of a finished run: the worlds shown to the user as its result.",
    "user": ("The aim of this experiment, in the user's words: \"{aim}\"\n\n"
             "Images 1 to {n} are the run's best worlds by the cheap judge's scores, at most two from any one family of "
             "descendants, in no particular order. " + _FRAMES + "\n\n"
             "Choose the {k} worlds that together make the best gallery for the aim: each should be worth showing as a "
             "result for the aim on its own, and the set should be as varied as possible, so that no two show essentially "
             "the same thing. Judge by the structure and behaviour the worlds show, not by their colours.\n\n"
             "Return {\"picks\": [the {k} image numbers, best first], \"why\": one short reason per pick, in the same "
             "order, separated by semicolons}."),
    "fields": ["why"]})                                   # "picks" (a list) passes through describe.parse as an extra key

# Pairwise judge for the generative VLM (11-06; project doc "Full run (seed 1) - diagnosis"): the judge test asks Gemini
# the same pairs Felix rated blind, beside Jev, to see which agrees with him. One prompt for both questions and both
# inputs; the fields carry the layout (the four stills Jev sees, or eight frames of the film Felix sees), the question
# and, for goal pairs, the goal. Ours. (GOAL_JUDGE, goal_judge_v1, is the 10-29 goal-only version; kept for its cache.)
PAIR_JUDGE = "pair_judge_v1"
GEN.setdefault(PAIR_JUDGE, {
    "system": _GEN_SYSTEM + " Here you compare two worlds and answer one question about them.",
    "user": ("Image 1 and Image 2 are two worlds. {layout}\n\n{context}Question: {question}\n\n"
             "Answer from what the images show, not from what such worlds usually do. "
             "Return {\"choice\": \"1\" or \"2\", \"why\": one short sentence}."),
    "fields": ["choice", "why"]})
PAIR_LAYOUT = {
    "stills": _FRAMES,
    "strip": ("Each image shows one whole world as eight frames of a film, evenly spaced over the same stretch of time, "
              "in reading order (two rows of four), so you can see how it moves and changes."),
}


# ---------- motion benchmark, Stage A (11-07; project doc "Jev motion benchmark - spec") ----------
# Can Jev see how much a world changes, and does the picture fix it? One question for every configuration; the
# description of how to read the picture goes with the picture. R0 = the pipeline's grid and its own preamble (WORLD);
# the other configurations describe the time order and interval. Ours.
BENCH_FORMATS = ("grid", "line", "ghosts", "labels")
_ORDER = "The world wraps around at its edges. Compare the snapshots to see how the world changes over time"
BENCH_LAYOUT = {
    "R0": ("four snapshots taken some time apart, long enough to show movement and change, in reading order "
           "(top-left, top-right, bottom-left, bottom-right)"),
    "grid": ("four snapshots of the same world in time order, 400 simulation steps apart: top-left is the earliest, then "
             "top-right, then bottom-left, and bottom-right is the latest. " + _ORDER),
    "line": ("four snapshots of the same world in time order, 400 simulation steps apart, from the earliest on the left "
             "to the latest on the right. " + _ORDER),
    "ghosts": ("seven snapshots of the same world overlaid in one picture, 200 simulation steps apart: the latest is "
               "drawn brightest and each earlier one fainter, so faint trails show where things were. The world wraps "
               "around at its edges. Compare the bright and faint copies to see how the world changes over time"),
    "labels": ("four snapshots of the same world in time order, 400 simulation steps apart, from the earliest on the "
               "left to the latest on the right; each is labelled with its time step and drawn over the same coordinate "
               "grid, so you can compare positions. " + _ORDER),
    "grid9": ("nine snapshots of the same world in time order, 150 simulation steps apart, in reading order (three rows "
              "of three): top-left is the earliest and bottom-right is the latest. " + _ORDER),
}
assert ("as " + BENCH_LAYOUT["R0"] + ";") in WORLD
BENCH_QUESTION = ("Which world changes more over time: more of it moves, reshapes or rearranges from one snapshot to "
                  "the next?")


def bench_prompt(layout: str) -> str:
    """layout: R0 or a format. WORLD with its layout clause swapped, then the one question."""
    import hashlib
    text = (WORLD.replace(BENCH_LAYOUT["R0"], BENCH_LAYOUT[layout]).replace("{N}", "two worlds") +
            " The two images are {A} and {B}. " + BENCH_QUESTION)
    pid = "bench_a_" + hashlib.sha1(text.encode()).hexdigest()[:12]
    PROMPTS.setdefault(pid, {"instructions": text, "first": "The world in {A} changes more.",
                             "second": "The world in {B} changes more."})
    return pid



# ---------- rating benchmark (11-10; project doc "Jev motion benchmark - spec", Stage B) ----------
# Jev must make the calls Felix makes from films while seeing only stills. Two dimensions on his rated seed-1 pairs:
# the picture (its layout clause in WORLD is swapped for the format's) and the question (the pipeline's own, or the
# same with one sentence asking Jev to judge the worlds as films). R0 with the pipeline's question returns the
# pipeline's own prompt id, so that cell reproduces seed 1 exactly. Ours.
BENCH_MOTION = ("Judge each world as it would look as a film playing from its earliest snapshot to its latest: how it "
                "moves, which parts move and in which direction, and how it changes over time, not only how any single "
                "snapshot looks.")


def rate_prompt(fmt: str, base_pid: str, motion: bool = False) -> str:
    """base_pid: INTEREST or an attain_prompt id. fmt: a BENCH_LAYOUT key (R0 = the pipeline's grid)."""
    import hashlib
    p = PROMPTS[base_pid]
    text = p["instructions"]
    assert BENCH_LAYOUT["R0"] in text, base_pid
    text = text.replace(BENCH_LAYOUT["R0"], BENCH_LAYOUT[fmt])
    if motion:
        k = text.index("The two images are {A} and {B}.") + len("The two images are {A} and {B}.")
        text = text[:k] + " " + BENCH_MOTION + text[k:]
    if text == p["instructions"]:
        return base_pid
    pid = "rate_" + hashlib.sha1((text + p["first"] + p["second"]).encode()).hexdigest()[:12]
    PROMPTS.setdefault(pid, {"instructions": text, "first": p["first"], "second": p["second"]})
    return pid


# ---------- lookup ----------

def get_prompt(prompt_id: str) -> dict[str, str]:
    if prompt_id not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt_id!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[prompt_id]


def render_prompt(prompt_id: str, a: str, b: str) -> dict[str, str]:
    out = {k: v.replace("{A}", a).replace("{B}", b) for k, v in get_prompt(prompt_id).items()}
    return {k: v[:1].upper() + v[1:] if k != "instructions" else v for k, v in out.items()}
