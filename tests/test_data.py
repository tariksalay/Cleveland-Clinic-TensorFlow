from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr4 import load_data, split_data, SELECTED_FEATURES


class DataTests(unittest.TestCase):
    def test_repository_dataset_and_disjoint_reproducible_splits(self):
        data = load_data(Path(__file__).resolve().parents[1] / 'heart.csv')
        first = split_data(data, SELECTED_FEATURES)
        second = split_data(data, SELECTED_FEATURES)
        self.assertEqual(sum(len(y) for x, y in first), len(data))
        for (x, y), (xx, yy) in zip(first, second):
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)
            self.assertEqual(x.shape[1], 4)
        np.testing.assert_allclose(first[0][0].mean(axis=0), 0, atol=1e-6)

    def test_invalid_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.csv'
            path.write_text('target,cp,slope,exang,thal\n2,0,0,0,0\n')
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == '__main__':
    unittest.main()
