# Heart Disease Dataset — TensorFlow / Keras Experiment

**Python · pandas · scikit-learn · TensorFlow · Keras**

An early machine-learning experiment that trains feed-forward neural networks against a tabular heart-disease dataset and compares using all available features versus a selected four-feature subset.

> **Status:** Historical coursework experiment. It is not a clinically validated prediction model and must not be used for medical decisions.

## Workflow

1. Read the repository's [heart.csv](heart.csv) dataset with pandas.
2. Inspect the table for missing values.
3. Separate input features from the `target` column.
4. Split records into **80% training and 20% testing** in the current code, using `random_state=42`.
5. Train a two-hidden-layer Keras Sequential network with 64 ReLU units in each layer.
6. Repeat using four named features: `cp`, `slope`, `exang`, and `thal`.

The implementation uses a single dense output and mean squared error loss with an `accuracy` metric. This is preserved as written; it should not be confused with a tuned, validated binary-classification pipeline.

## Files

- [pr4.py](pr4.py) — feature selection, train/test splitting, model creation, training, evaluation.
- [heart.csv](heart.csv) — tabular dataset used by the example.
- [heart.txt](heart.txt) — supplemental dataset/project notes.

## Reproducibility and limitations

The current script relies on absolute Windows paths such as `C:/Users/tarik/PycharmProjects/pr4`, does not pin packages, and trains for 1,000 epochs without a separate validation workflow. The original README described a 70/15/15 split, but the source code actually uses an **80/20 split**.

Before rerunning it, update file paths and validate TensorFlow/Keras compatibility in an isolated environment. No accuracy score or real-world predictive capability is claimed here.

## Disclaimer

A learning exercise involving medical data, not an operational diagnostic model. It should not be used to diagnose, screen, or treat disease.

[GitHub profile](https://github.com/tariksalay)
