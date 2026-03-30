# Merge Request Description

## Summary

This MR strengthens the calibration workflow, cleans up supporting quality
issues, introduces a cleaner public IR namespace, and refreshes the project
documentation so it matches the current repository state.

## What Changed

### Calibration

- Added representative dataset generation utilities under
  `src/edgelstm/calibration/`
- Added multiple sampling strategies:
  `uniform`, `stratified_temporal`, `regime_aware`, and `tail_aware`
- Added calibration statistics helpers and distribution comparison utilities
- Added representative-dataset test coverage in
  `tests/unit/test_representative_dataset.py`

### Public API

- Added the preferred public namespace `edge_lstm.ir`
- Kept `edge_lstm.ir_graph` as a compatibility alias
- Updated IR-focused tests to exercise `edge_lstm.ir`

### Quality And Maintenance

- Cleaned up repo lint issues that were blocking `ruff`
- Fixed the representative-dataset test typing issue that was failing
  `pyrefly check`
- Kept the branch green under lint and test verification

### Documentation

- Rewrote the top-level `README.md`
- Added structured docs for architecture, calibration, development, and roadmap
- Expanded environment setup documentation

## Why

These changes make the repo more usable in three ways:

1. Calibration work now has a real implementation and validation coverage.
2. Consumers get a cleaner public import path with `edge_lstm.ir`.
3. The docs now reflect what the project actually does today versus what is
   still planned.

## Verification

The branch was verified with:

```bash
ruff check src tests
pyrefly check
pytest -q
```

## Notes

- `edge_lstm.ir` is now the preferred public IR namespace.
- `edge_lstm.ir_graph` remains available as a compatibility layer for now.
