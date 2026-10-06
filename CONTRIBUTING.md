# Contributing

## Workflow

1. Open or pick an issue. Use the templates (feature, bug, research note).
2. Branch from `main`: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>` or `exp/<short-name>` for experiments.
3. Keep PRs small and focused. One reviewer from the team approves before merge.
4. CI (lint + tests) must pass.

## Commit messages

Short imperative summary, for example:

```
Add packet-drop model to comms graph
Fix belief renormalization when map is all zeros
```

## Code

- Run `ruff check src tests` and `pytest` before pushing.
- New behavior gets a test and, if user-facing, a config option.
- Agents only use local information and messages received over the comms graph.

## Documentation

- Meeting notes: `docs/meetings/YYYY-MM-DD.md` (copy `TEMPLATE.md`).
- Design decisions: `docs/decisions/NNNN-title.md` (copy `0000-template.md`).
- Papers: add an entry to `docs/literature.md` with a 2-3 line takeaway.
