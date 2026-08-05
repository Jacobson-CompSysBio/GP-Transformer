import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.fit_decomposition import load_training_frame
from utils.data_integrity import (
    align_xy_by_physical_row,
    extract_hybrid_name,
    extract_hybrid_series,
    split_hybrid_parents,
    validate_xy_physical_alignment,
)


class CanonicalIdentityTests(unittest.TestCase):
    def test_exact_env_prefix_handles_hyphenated_environment(self):
        self.assertEqual(
            extract_hybrid_name("TXH1-Dry_2017-P001/PHP02", "TXH1-Dry_2017"),
            "P001/PHP02",
        )
        hybrid = extract_hybrid_series(
            pd.Series(["TXH1-Early_2018-P002/LH287"]),
            pd.Series(["TXH1-Early_2018"]),
        )
        self.assertEqual(hybrid.tolist(), ["P002/LH287"])

    def test_malformed_id_or_parent_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exact Env prefix"):
            extract_hybrid_name("OTHER_2017-P001/PHP02", "TXH1-Dry_2017")
        with self.assertRaisesRegex(ValueError, "exactly one slash"):
            split_hybrid_parents(pd.Series(["P001-PHP02"]))
        with self.assertRaisesRegex(ValueError, "two non-empty"):
            split_hybrid_parents(pd.Series(["P001/"]))

    def test_physical_alignment_preserves_duplicate_ids(self):
        x = pd.DataFrame(
            {
                "id": ["E_2020-A/T", "E_2020-A/T", "E_2020-B/T"],
                "Env": ["E_2020"] * 3,
            }
        )
        y = pd.DataFrame(
            {
                "id": x["id"],
                "Yield_Mg_ha": [1.0, 1.2, 2.0],
            }
        )
        aligned = align_xy_by_physical_row(
            x, y, y_columns=["Yield_Mg_ha"], source="test rows"
        )
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned["Yield_Mg_ha"].tolist(), [1.0, 1.2, 2.0])

        swapped = y.iloc[[1, 0, 2]].reset_index(drop=True)
        # The duplicate ids alone cannot reveal a swap, but a distinct row must.
        swapped.loc[0, "id"] = "E_2020-C/T"
        with self.assertRaisesRegex(ValueError, "physical row 0"):
            align_xy_by_physical_row(x, swapped, y_columns=["Yield_Mg_ha"])

    def test_paired_missing_identifiers_are_rejected(self):
        x = pd.DataFrame({"id": [None], "Env": ["E_2020"]})
        y = pd.DataFrame({"id": [None], "Yield_Mg_ha": [1.0]})
        with self.assertRaisesRegex(ValueError, "missing/empty id"):
            validate_xy_physical_alignment(x, y)

    def test_paired_empty_environments_are_rejected(self):
        x = pd.DataFrame({"id": ["E_2020-A/T"], "Env": [""]})
        y = pd.DataFrame({"id": ["E_2020-A/T"], "Env": [""]})
        with self.assertRaisesRegex(ValueError, "missing/empty Env"):
            validate_xy_physical_alignment(x, y)

    def test_decomposition_loader_never_cartesian_expands_replicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [
                "TXH1-Dry_2017-A/T",
                "TXH1-Dry_2017-A/T",
                "E_2018-B/T",
            ]
            pd.DataFrame(
                {"id": ids, "Env": ["TXH1-Dry_2017"] * 2 + ["E_2018"]}
            ).to_csv(root / "X_train.csv", index=False)
            pd.DataFrame(
                {"id": ids, "Yield_Mg_ha": [1.0, 1.5, 2.0]}
            ).to_csv(root / "y_train.csv", index=False)

            frame = load_training_frame(f"{root}/")
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame["Hybrid"].tolist(), ["A/T", "A/T", "B/T"])
            self.assertEqual(frame["Yield_Mg_ha"].tolist(), [1.0, 1.5, 2.0])


if __name__ == "__main__":
    unittest.main()
