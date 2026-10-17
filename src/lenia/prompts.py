"""Pairwise prompts for judging Lenia creatures.

{A} and {B} name the two images: "the first image" / "the second image" for the OpenRouter
Decisions API, and the state keys `first` / `second` for Jev. The ids are stable and the full text
goes into the judge cache key, so editing a prompt invalidates its cached answers.
"""
from __future__ import annotations

CONTEXT = ("You are shown two creatures from Lenia, a continuous cellular automaton used in artificial-life "
           "research. Each image shows one creature as four snapshots taken a few moments apart, in reading "
           "order (top-left, top-right, bottom-left, bottom-right). The two images are {A} and {B}. ")

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
