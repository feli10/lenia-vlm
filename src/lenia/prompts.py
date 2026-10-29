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


# ---------- lookup ----------

def get_prompt(prompt_id: str) -> dict[str, str]:
    if prompt_id not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt_id!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[prompt_id]


def render_prompt(prompt_id: str, a: str, b: str) -> dict[str, str]:
    out = {k: v.replace("{A}", a).replace("{B}", b) for k, v in get_prompt(prompt_id).items()}
    return {k: v[:1].upper() + v[1:] if k != "instructions" else v for k, v in out.items()}
