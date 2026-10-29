# Working on lenia-vlm

Read README.md first: it maps the live code. **Do not read `archive/` unless asked**: it holds retired
designs (distillation, the alive classifier, the hill climber) that no longer describe this project.
Plans, results and decisions live in the claude.ai project "Research At NYU Game Innovation Lab", not here;
the current plan is "VLM in the loop - escalation plan (core first)".

## Working rules (Felix)
- Felix commits and pushes himself: end with a one-line commit message.
- API keys live only in `.env`; never print or log them.
- GPU runs on Colab; judge calls on the Mac or on Colab. Run folders are in Google Drive `lenia-runs/`,
  rsynced to `runs/`.
- In the device shell: no `uv sync` / `uv run` (it rebuilds .venv); use `PYTHONPATH=src:scripts python3`.
  It cannot delete files: use `mv`. No pytest there: run `python3 tests/test_golden.py` directly.
- Use no more judge or VLM calls than a result needs. Review inherited components before reusing them in a
  new experiment: keep them only if they pull their weight.
- Say what comes from papers and what is ours.

## Do not break the caches
Judge caches are keyed on the prompt text and the picture bytes, and saved runs depend on `mutate()`, the
founders and the simulator. `tests/test_golden.py` checks all of these; it must pass before and after any
change. Changing a prompt or `lenia.mace.picture` on purpose means starting fresh caches.

## Jev and Luna, briefly
- Pairwise decision models: they choose, they cannot write. Every duel is asked in both orders (Jev favours
  the second image, P(first) about 0.41; Luna the first, about 0.58).
- Jev bills only input tokens, about 2,640 per duel of two 644-px pictures ($0.00011). Smaller pictures:
  484 px is validated for the surprise duel only (gate 0f, `runs/res-check`).
- The config calls `jev-latest`, an alias; pin a version before the next long programme.
