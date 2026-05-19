# Tests

```bash
pytest -m unit -v
pytest -m slow -v
pytest -m reproducibility -v
```

- `unit`: no network or transformer downloads.
- `slow`: transformer feature extraction and MoE smoke tests.
- `reproducibility`: deterministic MoE checks on fixed synthetic 4D data.

The production path is MoE-only. Old bagged tests and scripts are intentionally removed.
