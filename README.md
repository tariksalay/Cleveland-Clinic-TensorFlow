# Heart-Data Classification Experiment

**Python · pandas · scikit-learn · TensorFlow/Keras**

An educational comparison of two neural-network inputs: all available feature columns versus four selected columns. The script validates a bundled tabular dataset, creates reproducible partitions, and reports final test metrics for each experiment.

## What this project demonstrates

- Separating training, validation, and final testing.
- Fitting preprocessing on training rows and reusing it on held-out data.
- Comparing feature sets with a consistent model architecture.

## Start here

| Read | Use it for |
| --- | --- |
| [Project walkthrough](docs/PROJECT_GUIDE.md) | Purpose, workflow, repository map, and a concrete example |
| [Code and artifact reference](docs/CODE_REFERENCE.md) | Important files, functions, classes, and their responsibilities |
| [Development guide](docs/DEVELOPMENT.md) | Setup, configuration, commands, tests, and remaining limitations |

## Explore locally

Use Python 3.11 or 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python pr4.py --epochs 20
python -m unittest discover -s tests -v
```

## Validation and scope

Two data/split tests pass, and the Keras model completed one small training batch. Full training quality has not been benchmarked. This is coursework involving medical data, not a clinically validated diagnostic or screening model.



---

[GitHub profile](https://github.com/tariksalay) · [Portfolio](https://tariksalay.github.io/) · [Earlier README](README_HISTORY.md)

The earlier README is preserved as historical context; the guides above describe this branch's implementation.
