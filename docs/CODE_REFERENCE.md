# Code and artifact reference

[Overview](../README.md) · [Walkthrough](PROJECT_GUIDE.md) · [Development guide](DEVELOPMENT.md)

This reference describes the files on this branch. Function names are navigation aids; the walkthrough explains how their inputs, state, and outputs fit together.

## Python modules

### [pr4.py](../pr4.py)

Validate the heart-data schema, create leakage-aware partitions, and train/evaluate two dense classifiers.

- `load_data(path)`
- `split_data(data, features, seed=42)`
- `build_model(feature_count)`
- `main()`

## Checks and setup

[Tests](../tests/) contains the focused checks described in the [development guide](DEVELOPMENT.md). Consult that guide for dependencies, commands, and the boundary between automated validation and untested integrations.
