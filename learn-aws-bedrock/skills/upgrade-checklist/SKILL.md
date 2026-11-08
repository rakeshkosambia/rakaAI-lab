---
name: upgrade-checklist
description: A generic checklist for planning a runtime or framework upgrade. Use when asked to plan, review or de-risk a version upgrade.
---
# Upgrade checklist

1. Record the current and target versions of the runtime, framework and key libraries.
2. List dependencies that have no release supporting the target version.
3. Find removed or renamed APIs that the code uses (compiler warnings and release notes are the source).
4. Decide the order: build tooling first, then libraries, then application code.
5. Define a rollback step for every stage before starting it.
6. Run the full test suite before and after; compare, do not just look for green.
7. Report: blockers, effort per area (small, medium, large), and a proposed order.

Rules:
- Cite the release note or file that supports each claim.
- Separate "will break" from "might break".
