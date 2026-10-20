"""Tests use generated synthetic data only, never OEM recordings."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from icing_demo.schema import FEATURES, validate_dataframe
from icing_demo.synthetic import generate_synthetic_recordings
from icing_demo.training import split_by_recording, train_and_evaluate
from icing_demo.predict import predict_file


class IcingDemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_recordings(per_class=5, samples_per_recording=40)

    def test_three_inputs_and_binary_labels(self):
        self.assertTrue(set(FEATURES).issubset(self.data.columns))
        self.assertEqual(set(self.data["icing_label"]), {0, 1})

    def test_reproducible_data(self):
        other = generate_synthetic_recordings(per_class=5, samples_per_recording=40)
        pd.testing.assert_frame_equal(self.data, other)

    def test_recording_class_consistency(self):
        self.assertEqual(self.data.groupby("recording_id").icing_label.nunique().max(), 1)

    def test_no_recording_leakage(self):
        train, test = split_by_recording(self.data)
        self.assertFalse(set(train.recording_id) & set(test.recording_id))

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataframe(self.data.drop(columns="high_pressure_bar"))

    def test_nan_rejected(self):
        broken = self.data.copy()
        broken.loc[0, "suction_pressure_bar"] = np.nan
        with self.assertRaises(ValueError):
            validate_dataframe(broken)

    def test_conflicting_recording_label_rejected(self):
        broken = self.data.copy()
        broken.loc[0, "icing_label"] = 1 - broken.loc[0, "icing_label"]
        with self.assertRaises(ValueError):
            validate_dataframe(broken)

    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metrics = train_and_evaluate(self.data, root / "model")
            self.assertIn("SYNTHETIC", metrics["dataset"])
            self.assertTrue(0 <= metrics["accuracy"] <= 1)
            self.assertEqual(len(metrics["confusion_matrix_true_rows_predicted_columns"]), 2)
            csv_file = root / "data.csv"
            self.data.to_csv(csv_file, index=False)
            result = predict_file(csv_file, root / "model", root / "pred.csv")
            self.assertEqual(len(result), len(self.data))
            self.assertTrue(result.icing_probability.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
