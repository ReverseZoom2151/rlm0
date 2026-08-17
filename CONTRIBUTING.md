# Contributing

Keep changes small, typed, and independently testable. Run the complete local
check before opening a pull request:

```bash
python -m pytest
ruff check .
mypy src tests
```

Do not add a benchmark score without its depth-0 row, model identifier, budget,
wall-clock measurement, and evidence-grade result. Do not weaken the sandbox
defaults to make a local demo convenient. New provider integrations must report
provider usage, preserve unpriced cost as `None`, and include fixture tests.

For a behavior change, add the regression test first when practical. Keep API
and security changes in separate commits from prose-only changes.
