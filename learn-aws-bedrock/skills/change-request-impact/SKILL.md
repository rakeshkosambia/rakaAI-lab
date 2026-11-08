---
name: change-request-impact
description: A checklist for judging which parts of a system a change request touches. Use when asked to assess the impact or risk of a requested change.
---
# Change request impact assessment

Work through these steps in order and keep notes short.

1. Restate the change in one sentence: what behaviour changes, for whom.
2. Find the entry points the change touches (screens, endpoints, jobs, messages).
3. Trace each entry point to the data it reads or writes (tables, files, queues).
4. List other callers of anything you would modify. A shared helper counts as a wide impact.
5. Classify the impact: LOCAL (one module), SHARED (several modules), or CROSS-CUTTING (config, schema, security).
6. Name the tests that must run and any test that is missing.
7. Report: summary, impact class, touched areas, risks, open questions.

Rules:
- Say what you verified and what you assumed. Never present a guess as a finding.
- If a file you need is not available, say so instead of inventing its contents.
