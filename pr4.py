"""Reproducible coursework comparison; not a clinically validated model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SELECTED_FEATURES = ['cp', 'slope', 'exang', 'thal']


def load_data(path):
    data = pd.read_csv(path)
    required = {'target', *SELECTED_FEATURES}
    if not required.issubset(data.columns):
        raise ValueError(f'Dataset must include {sorted(required)}')
    if data.isna().any().any():
        raise ValueError('Dataset contains missing values; choose an explicit imputation strategy.')
    if set(data['target'].unique()) != {0, 1}:
        raise ValueError('Expected a binary target containing 0 and 1.')
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError('All input columns must be numeric.')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Dataset contains non-finite values.')
    return data


def split_data(data, features, seed=42):
    x = data[features]
    y = data['target']
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y)
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.20, random_state=seed, stratify=train_y)
    scaler = StandardScaler().fit(train_x)
    return ((scaler.transform(train_x).astype('float32'), train_y.to_numpy()),
            (scaler.transform(valid_x).astype('float32'), valid_y.to_numpy()),
            (scaler.transform(test_x).astype('float32'), test_y.to_numpy()))


def build_model(feature_count):
    from tensorflow import keras
    model = keras.Sequential([
        keras.Input(shape=(feature_count,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('heart.csv'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    data = load_data(args.data)
    from tensorflow import keras
    keras.utils.set_random_seed(args.seed)
    for name, features in [('all features', list(data.drop(columns='target').columns)),
                           ('four features', SELECTED_FEATURES)]:
        train, valid, test = split_data(data, features, args.seed)
        keras.backend.clear_session()
        model = build_model(len(features))
        model.fit(*train, validation_data=valid, epochs=args.epochs, batch_size=32,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                          restore_best_weights=True)], verbose=0)
        print(name, model.evaluate(*test, verbose=0, return_dict=True))


if __name__ == '__main__':
    main()
