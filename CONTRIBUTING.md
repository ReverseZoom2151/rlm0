# Contributing

Keep changes small, typed, and independently testable. Run the complete local
check before opening a pull request:

```bash
python -m pytest
ruff check .
mypy src tests
```

Do not add a benchmark score without the depth-zero row, a strong
nonrecursive baseline, model identifier, budget, wall-clock measurement,
evidence-grade result, and raw per-sample records. Do not weaken the sandbox
defaults to make a local demo convenient. New provider integrations must report
provider usage, preserve unpriced cost as `None`, and include fixture tests.

Do not label a backend isolated because it accepts container flags. A new
backend needs live tests for its claimed kernel, network, filesystem, user,
capability, and process boundaries. If a feature is prepared but not connected
to a public execution path, document it as such.

For a behavior change, add the regression test first when practical. Keep API
and security changes in separate commits from prose-only changes.
