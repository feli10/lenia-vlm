"""Pairwise prompts for judging Lenia creatures.

{A} and {B} name the two images: "the first image" / "the second image" for the OpenRouter
Decisions API, and the state keys `first` / `second` for Jev. The ids are stable and the full text
goes into the judge cache key, so editing a prompt invalidates its cached answers.
"""
from __future__ import annotations

CONTEXT = ("You are shown two creatures from Lenia, a continuous cellular automaton used in artificial-life "
           "research. Each image shows one creature as four snapshots taken a few moments apart, in reading "
           "order (top-left, top-right, bottom-left, bottom-right). The two images are {A} and {B}. ")

CONTEXT_DUAL = ("You are shown two creatures from Lenia, a continuous cellular automaton used in artificial-life "
                "research. Each image shows one creature in two views, both drawn at the same scale for every "
                "creature. Left: a wide view of the creature's recent movement, with its positions over the last "
                "16 time units overlaid as multiple exposures (fainter copies are earlier, the brightest is the "
                "latest), so you can see how far, how fast and along what path it moves; a creature that stays in "
                "place shows a single shape. Right: four close-ups of the creature, centred on it, 2 time units "
                "apart in reading order (top-left, top-right, bottom-left, bottom-right), showing its shape and "
                "how it changes. The two images are {A} and {B}. ")

CONTEXT_DUAL_BODY = ("You are shown two creatures from Lenia, a continuous cellular automaton used in artificial-life "
                     "research. Each image shows one creature in two views, each drawn relative to the creature's own "
                     "size, so the images do not show how big the creatures are compared with each other. Left: the "
                     "creature's recent movement, in a window four body lengths wide, with its positions over the last "
                     "16 time units overlaid as multiple exposures (fainter copies are earlier, the brightest is the "
                     "latest), so you can see how many body lengths it travels and along what path; a creature that "
                     "stays in place shows a single shape. Right: four close-ups filling the frame, 2 time units apart in "
                     "reading order (top-left, top-right, bottom-left, bottom-right), showing its shape and how it "
                     "changes. The two images are {A} and {B}. ")

PROMPTS: dict[str, dict[str, str]] = {
    "appeal": {
        "instructions": CONTEXT + "Which creature is more visually appealing?",
        "first": "The creature in {A} is more visually appealing.",
        "second": "The creature in {B} is more visually appealing.",
    },
    "lifelike": {
        "instructions": CONTEXT + "Which creature looks more like a living organism?",
        "first": "The creature in {A} looks more like a living organism.",
        "second": "The creature in {B} looks more like a living organism.",
    },
    "appeal_bare": {                   # the same question with no Lenia / artificial-life framing
        "instructions": "You are shown two images: {A} and {B}. Which image is more visually appealing?",
        "first": "{A} is more visually appealing.",
        "second": "{B} is more visually appealing.",
    },
    "beautiful": {
        "instructions": CONTEXT + "Which creature is more beautiful?",
        "first": "The creature in {A} is more beautiful.",
        "second": "The creature in {B} is more beautiful.",
    },
    "curate": {                        # expert-curator framing: Chan's catalogue is itself a curated collection
        "instructions": CONTEXT + "You are curating a gallery of the most remarkable creatures discovered in Lenia. "
                                  "Which creature would you rather include?",
        "first": "I would rather include the creature in {A}.",
        "second": "I would rather include the creature in {B}.",
    },
    "curate_dual": {                   # `curate` for the two-view images (pool --camera dual)
        "instructions": CONTEXT_DUAL + "You are curating a gallery of the most remarkable creatures discovered in Lenia. "
                                       "Which creature would you rather include?",
        "first": "I would rather include the creature in {A}.",
        "second": "I would rather include the creature in {B}.",
    },
    "curate_dual_body": {              # `curate` for the size-removed two-view images (pool --camera dual_body)
        "instructions": CONTEXT_DUAL_BODY + "You are curating a gallery of the most remarkable creatures discovered in "
                                            "Lenia. Which creature would you rather include?",
        "first": "I would rather include the creature in {A}.",
        "second": "I would rather include the creature in {B}.",
    },
    # prompt arm (RQ3) on the two-view images: does the wording move the teacher off "bigger wins"?
    "curate_dual_nuanced": {
        "instructions": CONTEXT_DUAL + "You are curating a gallery of the most remarkable creatures discovered in Lenia. "
                                       "Judge each creature by its form, internal structure and behaviour, not by how "
                                       "large it appears. Which creature would you rather include?",
        "first": "I would rather include the creature in {A}.",
        "second": "I would rather include the creature in {B}.",
    },
    "lifelike_dual": {
        "instructions": CONTEXT_DUAL + "Which creature behaves more like a living organism?",
        "first": "The creature in {A} behaves more like a living organism.",
        "second": "The creature in {B} behaves more like a living organism.",
    },
    "elegant_dual": {
        "instructions": CONTEXT_DUAL + "Which creature is more elegant?",
        "first": "The creature in {A} is more elegant.",
        "second": "The creature in {B} is more elegant.",
    },
    "interesting": {
        "instructions": CONTEXT + "Which creature is more interesting to look at?",
        "first": "The creature in {A} is more interesting to look at.",
        "second": "The creature in {B} is more interesting to look at.",
    },
}


def get_prompt(prompt_id: str) -> dict[str, str]:
    if prompt_id not in PROMPTS:
        raise ValueError(f"unknown prompt {prompt_id!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[prompt_id]


def render_prompt(prompt_id: str, a: str, b: str) -> dict[str, str]:
    out = {k: v.replace("{A}", a).replace("{B}", b) for k, v in get_prompt(prompt_id).items()}
    return {k: v[:1].upper() + v[1:] if k != "instructions" else v for k, v in out.items()}
