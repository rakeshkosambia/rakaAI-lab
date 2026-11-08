# b07: Skill loading

> *Load knowledge on demand, not upfront.*

Equivalent in the Claude Code course: **s07 Skill Loading**. AWS twist: skills can live in S3, so a team shares one set.

## The problem

Ten playbooks of 2,000 words each, pasted into every system prompt, cost money on **every** request, including the ones that need none of them.

## The idea

A library with a card catalogue. The catalogue (name + one line) is always on the desk. The book is fetched only when someone needs it.

- **Layer 1, always present**: a list of `name: description`, about 450 characters for two skills.
- **Layer 2, on demand**: the model calls `load_skill("change-request-impact")` and gets the full body back as a tool result.

A skill is a folder with a `SKILL.md`:

```
---
name: change-request-impact
description: A checklist for judging which parts of a system a change request touches. Use when asked to assess impact.
---
# Body: steps, rules, output format ...
```

The description is what decides whether the model loads the skill, so write it as "what it is, and when to use it".

## Sources

| Source | Where skills live | When to use it |
|---|---|---|
| `LocalSkillSource` | the `skills/` folder in this repo | learning, a single developer |
| `S3SkillSource` | `s3://BUCKET/PREFIX/<name>/SKILL.md` | a team sharing one set of skills |

Switch with `SKILLS_SOURCE=s3`, `SKILLS_S3_BUCKET`, `SKILLS_S3_PREFIX` in `.env`. Give the role only the access in `iam/s3-read-prefix.json` (list and get on that prefix, nothing else).

Both sources expose `names()` and `read(name)`; adding another (for example Parameter Store) means writing those two methods.

## Run it

```
python b07_skill_loading/code.py --offline
run.bat b07_skill_loading --prompt "Plan an upgrade of our build tooling. Use the right skill."
```

## Try this

1. Add a third skill folder, run again, and check that the catalogue lists it with no code change.
2. Write a vague description, then a precise one. Which does the model load correctly?
3. Ask for a skill that does not exist and read the error that lists the valid names.

## Security note

**A skill is instructions the model will follow.** Whoever can write to the skills folder or bucket can steer your agent. Treat write access to a shared skills bucket as a privileged permission, review changes, and keep the bucket read-only for the agent's role.

## Pitfalls

- **Descriptions that overlap** make the model pick the wrong skill.
- **Huge skill bodies** defeat the purpose. If a skill is very long, split it.
- **Reading every skill at start-up** is fine locally but slow on S3; the catalogue is cached per run.
