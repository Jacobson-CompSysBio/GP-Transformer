import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.checkpoints import resolve_checkpoint_path


def _has_modules(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


HAS_DATASET_RUNTIME = _has_modules("torch", "transformers")
HAS_TORCH = _has_modules("torch")


def _write_synthetic_gxe_data(root: Path) -> None:
    rng = np.random.default_rng(7)
    marker_cols = ["m0", "m1", "m2"]
    env_cols = [f"env_{i:03d}" for i in range(705)]
    rows = []
    y_rows = []

    envs = [f"LOC{i}_{year}" for i, year in enumerate(range(2019, 2024), start=1)]
    parent1s = ["P001", "P002", "P003", "P004"]
    testers = ["PHP02", "CHK01"]
    for env_idx, env in enumerate(envs):
        year = int(env.rsplit("_", 1)[1])
        for p_idx, parent1 in enumerate(parent1s):
            for tester in testers:
                hybrid = f"{parent1}/{tester}"
                markers = ((rng.integers(0, 3, size=len(marker_cols))) / 2.0).tolist()
                env_values = np.linspace(0, 1, len(env_cols), dtype=float) + env_idx
                sample_id = f"{env}-{hybrid}"
                rows.append([sample_id, env, *markers, *env_values.tolist()])
                y_rows.append({"Yield_Mg_ha": 2.5 + 0.1 * (year - 2019) + 0.05 * p_idx})

    test_rows = []
    test_y = []
    for hybrid in ["LH287/PHP02", "P001/PHP02"]:
        env = "TST1_2024"
        markers = ((rng.integers(0, 3, size=len(marker_cols))) / 2.0).tolist()
        env_values = np.linspace(1, 2, len(env_cols), dtype=float)
        test_rows.append([f"{env}-{hybrid}", env, *markers, *env_values.tolist()])
        test_y.append({"Yield_Mg_ha": 3.0})

    cols = ["id", "Env", *marker_cols, *env_cols]
    pd.DataFrame(rows, columns=cols).to_csv(root / "X_train.csv", index=False)
    pd.DataFrame(y_rows).to_csv(root / "y_train.csv", index=False)
    pd.DataFrame(test_rows, columns=cols).to_csv(root / "X_test.csv", index=False)
    pd.DataFrame(test_y).to_csv(root / "y_test.csv", index=False)


class CheckpointResolutionTests(unittest.TestCase):
    def test_checkpoint_tag_resolution_prefers_manifest_and_latest_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint_0001.pt").write_text("old")
            (root / "checkpoint_0002.pt").write_text("new")
            alias = root / "best_leo.pt"
            alias.write_text("alias")
            (root / "checkpoint_manifest.json").write_text(json.dumps({
                "aliases": {"best_leo": {"path": str(alias.resolve()), "epoch": 2}}
            }))

            self.assertEqual(resolve_checkpoint_path(root, "best_leo"), alias.resolve())
            self.assertEqual(resolve_checkpoint_path(root), root / "checkpoint_0002.pt")

            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_path(root, "missing")

    def test_checkpoint_tag_resolution_falls_back_to_latest_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest.pt"
            latest.write_text("latest")
            self.assertEqual(resolve_checkpoint_path(root), latest)


@unittest.skipIf(not HAS_DATASET_RUNTIME, "torch/transformers are required for dataset tests")
class DatasetSplitTests(unittest.TestCase):
    def test_proxy_same_tester_validation_is_group_disjoint(self):
        from utils.dataset import GxE_Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_gxe_data(root)
            data_path = f"{root}/"

            train = GxE_Dataset(
                split="train",
                data_path=data_path,
                scale_targets=False,
                val_scheme="proxy_same_tester",
                proxy_tester="PHP02",
                proxy_holdout_frac=0.25,
                proxy_seed=3,
            )
            val = GxE_Dataset(
                split="val",
                data_path=data_path,
                scaler=train.scaler,
                scale_targets=False,
                val_scheme="proxy_same_tester",
                proxy_tester="PHP02",
                proxy_holdout_frac=0.25,
                proxy_seed=3,
                proxy_val_parent1s=train.proxy_val_parent1s,
                parent_vocab=train.parent_vocab,
            )

            self.assertGreater(len(val), 0)
            self.assertFalse(set(train.meta["id"]) & set(val.meta["id"]))
            self.assertEqual(set(val.meta["parent2"]), {"PHP02"})
            self.assertTrue(set(val.meta["parent1"]).issubset(train.proxy_val_parent1s))
            self.assertEqual(train.proxy_info["proxy_row_count"], len(val))

    def test_leo_validation_has_no_environment_overlap(self):
        from utils.dataset import GxE_Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_gxe_data(root)
            data_path = f"{root}/"

            train = GxE_Dataset(
                split="train",
                data_path=data_path,
                scale_targets=False,
                val_scheme="leo",
                leo_val_fraction=0.40,
                leo_seed=11,
            )
            val = GxE_Dataset(
                split="val",
                data_path=data_path,
                scaler=train.scaler,
                scale_targets=False,
                val_scheme="leo",
                leo_val_envs=train.leo_val_envs,
                parent_vocab=train.parent_vocab,
            )

            self.assertGreater(len(val), 0)
            self.assertFalse(set(train.meta["Env"]) & set(val.meta["Env"]))
            self.assertTrue((train.meta["Year"] < 2024).all())
            self.assertTrue((val.meta["Year"] < 2024).all())

    def test_unseen_parent_maps_to_unk(self):
        from utils.dataset import GxE_Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_gxe_data(root)
            data_path = f"{root}/"

            train = GxE_Dataset(
                split="train",
                data_path=data_path,
                scale_targets=False,
                val_scheme="year",
            )
            test = GxE_Dataset(
                split="test",
                data_path=data_path,
                scaler=train.scaler,
                scale_targets=False,
                parent_vocab=train.parent_vocab,
            )

            unseen_idx = test.meta.index[test.meta["parent1"] == "LH287"][0]
            self.assertEqual(int(test.parent_id_tensor[unseen_idx, 0]), 0)
            self.assertGreater(int(test.parent_id_tensor[unseen_idx, 1]), 0)


@unittest.skipIf(not HAS_TORCH, "torch is required for model tests")
class EnvAffineCalibrationTests(unittest.TestCase):
    def test_env_affine_preserves_within_environment_rank_order(self):
        import torch
        from models.config import Config
        from models.model import FullTransformer

        torch.manual_seed(0)
        config = Config(
            block_size=3,
            n_env_fts=5,
            n_embd=16,
            n_gxe_layer=1,
            n_head=4,
            dropout=0.0,
            calibration_mode="env_affine",
        )
        model = FullTransformer(config, mlp_type="dense")
        model.eval()
        x = {
            "g_data": torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
            "e_data": torch.ones(2, 5),
            "parent_ids": torch.zeros(2, 2, dtype=torch.long),
            "g_additive": torch.zeros(2, 3),
            "g_dominance": torch.ones(2, 3),
        }

        with torch.no_grad():
            out = model(x)

        self.assertIn("total", out)
        self.assertTrue(torch.all(out["scale"] > 0))
        self.assertTrue(torch.allclose(out["scale"][0], out["scale"][1], atol=1e-6))
        rank_diff = out["rank"][0] - out["rank"][1]
        total_diff = out["total"][0] - out["total"][1]
        self.assertGreaterEqual(float((rank_diff * total_diff).item()), -1e-7)


if __name__ == "__main__":
    unittest.main()
