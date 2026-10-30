# Working on lenia-vlm

Read README.md first (it maps the live code), then docs/STATUS.md (where the project stands and what's next).
**Do not read `archive/` unless asked**: it holds retired designs (distillation, the alive classifier, the hill
climber) that no longer describe this project.

## Working rules (everyone)
- Two people work here (Felix and Sam). Work on your own branch (e.g. `sam/<topic>`); don't push to `main`.
- API keys live only in `.env`; never print, log or commit them.
- Try every new command offline first: each script's docstring has a `--judge fake` dry run. Paid scripts take
  `--max-usd` (below); keep it small.
- Use no more judge or VLM calls than a result needs.
- New experiments go in new scripts or new commands. Don't change what an existing command does: saved runs and
  their caches depend on it.
- Say what comes from papers and what is ours.

## Do not break the caches
All judge prompts live in `src/lenia/prompts.py`; add new ones there, never in scripts. Judge caches are keyed on
the prompt text and the picture bytes, and saved runs depend on `mutate()`, the
founders and the simulator. `tests/test_golden.py` checks all of these; it must pass before and after any
change. Changing a prompt or `lenia.mace.picture` on purpose means starting fresh caches.

## Jev and Luna, briefly
- Pairwise decision models: they choose, they cannot write. Every duel is asked in both orders (Jev favours
  the second image, P(first) about 0.41; Luna the first, about 0.58).
- Jev bills only input tokens, about 2,640 per duel of two 644-px pictures ($0.00011). Smaller pictures:
  484 px is validated for the surprise duel only (gate 0f).
- Jev is pinned (`configs/lenia.yaml`, `jev-1.14.0`; `scripts/pin_check.py` confirms it). Every cached answer records
  the provider's snapshot (`snapshots` table); a change mid-run logs a warning.
- Every script that calls a judge takes `--max-usd`: a hard cap on the run's spend, counted from its own cache, so
  it holds across resumes and is not overridden by `--yes`. A batch in which every call fails stops the run.

## Data you may not have
`runs/` is git-ignored (Felix's run folders live in his Google Drive). `mace_qd.py run` starts from scratch;
`stage1.py`, `gates.py`, `bakeoff.py`, `qd_compare.py`, `res_check.py`, `pin_check.py` and `goal_session.py` read
earlier runs (see each docstring). If one is missing, say so and ask for it rather than faking it.
