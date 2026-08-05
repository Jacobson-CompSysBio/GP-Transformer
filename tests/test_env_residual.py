import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HAS_TORCH_RUNTIME = importlib.util.find_spec("torch") is not None
HAS_LOSS_RUNTIME = HAS_TORCH_RUNTIME and importlib.util.find_spec("torchsort") is not None


@unittest.skipUnless(HAS_TORCH_RUNTIME, "torch is required")
class EnvironmentResidualHeadTests(unittest.TestCase):
    def test_head_uses_environment_data_for_the_mean(self):
        import torch

        from models.config import Config
        from models.model import FullTransformer

        config = Config(
            block_size=4,
            n_env_fts=3,
            n_gxe_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            prediction_head="env_residual",
        )
        model = FullTransformer(config, mlp_type="dense")
        model.eval()
        with torch.no_grad():
            model.env_mean_head[-1].weight.normal_(mean=0.0, std=0.1)
            model.env_mean_head[-1].bias.fill_(0.2)

        x = {
            "g_data": torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
            "e_data": torch.tensor([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]]),
        }
        predictions = model(x)

        self.assertEqual(
            set(predictions), {"total", "rank", "env_mean", "residual"}
        )
        self.assertTrue(
            torch.allclose(
                predictions["total"],
                predictions["env_mean"] + predictions["residual"],
            )
        )
        self.assertTrue(
            torch.allclose(predictions["env_mean"][0], predictions["env_mean"][1])
        )

        predictions["total"].square().mean().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNotNone(model.env_mean_head[-1].weight.grad)


@unittest.skipUnless(HAS_LOSS_RUNTIME, "torch and torchsort are required")
class EnvironmentResidualLossTests(unittest.TestCase):
    def test_auxiliary_loss_uses_the_fixed_weights(self):
        import torch

        from utils.loss import env_residual_auxiliary_loss

        env_id = torch.tensor([0, 0, 1, 1])
        env_mean = torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True)
        residual = torch.tensor([0.5, -0.5, 0.5, -0.5], requires_grad=True)
        total = env_mean + residual
        auxiliary, parts = env_residual_auxiliary_loss(
            {"total": total, "env_mean": env_mean},
            total_target=torch.zeros_like(total),
            env_mean_target=torch.zeros_like(env_mean),
            env_id=env_id,
            env_mean_weight=1.0,
            total_weight=0.05,
        )

        self.assertAlmostEqual(float(auxiliary.detach()), 0.5 + 0.05 * 0.5625, places=6)
        self.assertEqual(parts["env_mean_huber_weight_eff"], 1.0)
        self.assertEqual(parts["total_huber_weight_eff"], 0.05)
        auxiliary.backward()
        self.assertTrue(torch.isfinite(env_mean.grad).all())
        self.assertTrue(torch.isfinite(residual.grad).all())


@unittest.skipUnless(HAS_TORCH_RUNTIME, "torch is required")
class SharedTargetScaleTests(unittest.TestCase):
    def test_shared_total_keeps_the_additive_relation(self):
        from utils.dataset import GxE_Dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envs = ["LOC1_2022", "LOC1_2022", "LOC2_2022", "LOC2_2022"]
            x_data = {
                "id": [
                    "LOC1_2022-A/T",
                    "LOC1_2022-B/T",
                    "LOC2_2022-C/T",
                    "LOC2_2022-D/T",
                ],
                "Env": envs,
                "marker_1": [0.0, 0.5, 1.0, 0.0],
            }
            x_data.update({f"env_{index}": [float(index)] * 4 for index in range(705)})
            pd.DataFrame(x_data).to_csv(root / "X_train.csv", index=False)
            pd.DataFrame(
                {"Yield_Mg_ha": [8.0, 10.0, 12.0, 16.0]}
            ).to_csv(root / "y_train.csv", index=False)

            dataset = GxE_Dataset(
                split="train",
                data_path=f"{root}/",
                residual=True,
                scale_targets=True,
                decomposition_scale_mode="shared_total",
            )

            np.testing.assert_allclose(
                dataset.total_series.to_numpy(),
                dataset.env_mean.to_numpy() + dataset.residual.to_numpy(),
                rtol=1e-7,
                atol=1e-7,
            )
            self.assertEqual(dataset.label_scalers["resid"].mean, 0.0)
            self.assertEqual(
                dataset.label_scalers["resid"].std,
                dataset.label_scalers["total"].std,
            )


if __name__ == "__main__":
    unittest.main()
