# NOTICE: Upstream Attribution

This folder is a **copy of an open-source project**. It is not original work by the owner of this repository.

| Item | Value |
|---|---|
| Upstream project | learn-claude-code |
| Upstream author | shareAI Lab |
| Upstream repository | https://github.com/shareAI-lab/learn-claude-code |
| Upstream license | MIT (see `LICENSE`, unchanged, "Copyright (c) 2024 shareAI Lab") |
| Copied from commit | `7226f555d48900076eba126acf37c9ed5f002b5d` |
| Upstream commit date | 2026-10-27 |
| Copied on | 2026-11-08 |

The original `LICENSE` file is kept in full, as the MIT License requires. This copy is not affiliated with, endorsed by, or sponsored by shareAI Lab or Anthropic.

## What was copied (unchanged)

- `s01_agent_loop/` to `s17_goal_loop/`: all 17 lessons (each has `code.py`, `README.md`, `README.zh.md`, `README.ja.md`, and `images/`). Lesson code is byte-identical to upstream at the commit above.
- `skills/`: sample skills. Lessons s07 and s15 load skills from this folder at runtime.
- `README.md`, `README-zh.md`, `README-ja.md`, `requirements.txt`, `.env.example`, `.gitignore`: unchanged.
- `tests/`: 13 of upstream's 21 test files, the ones that do not depend on the legacy folders left out below.

## What was left out (on purpose)

- `agents/`, `docs/`: the older 12-lesson track that upstream keeps only for old links.
- `web/`: the web app that renders the older track.
- `.github/` (upstream CI workflows) and `CONTRIBUTING.md`.
- 8 test files that import `agents/`, `web/`, `docs/` or `s_full`.

To restore any of these, copy them from the upstream repository at the commit above.

## Local changes

None to upstream files. This `NOTICE.md` is the only file added in this folder. If you change lesson code later, record each change below so it stays clear what is upstream and what is yours.

| Date | File | Change |
|---|---|---|
| | | |

## Setup

```
pip install -r requirements.txt
copy .env.example .env      (Windows)   |   cp .env.example .env   (macOS/Linux)
```

Then edit `.env` and set `ANTHROPIC_API_KEY` (and optionally `MODEL_ID`). **Never commit `.env`**: it is listed in `.gitignore`.

Run the first lesson:

```
python s01_agent_loop/code.py
```

Lessons are meant to be read in order, s01 to s17. Only use the upstream repository named above as the source for updates; look-alike mirror sites may harvest API keys.
