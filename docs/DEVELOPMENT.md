# Development and validation

This modernization supplements the historical README. `pr4.py` now reads the adjacent `heart.csv` by default, validates numeric finite inputs, uses a sigmoid output with binary cross-entropy, and fits scaling on training rows only. It retains the all-features versus four-features comparison.

```sh
python -m pip install -r requirements.txt
python pr4.py --epochs 20
python pr4.py --data path/to/heart.csv --epochs 100 --seed 42
python -m unittest discover -s tests -v
```

Use Python 3.11 or 3.12. The seeded, stratified partitions are approximately 64% training, 16% validation, and 20% final testing. Early stopping monitors validation loss and restores the best weights. The final test partition is not used for model fitting or scaling.

Data validation/split tests and a single synthetic-size training batch can establish basic compatibility; they do not establish predictive quality. Full training results have not been benchmarked against the old experiment. This remains coursework and is not a clinical model.
