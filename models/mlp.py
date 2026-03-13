import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

_STAGE_WINDOW_ORDER = (
    "GerEme",
    "EmeEnJ",
    "EnJFlo",
    "FloFla",
    "FlaFlw",
    "FlwStG",
    "StGEnG",
    "EnGMat",
    "MatHar",
)
_WEATHER_STATS = ("min", "max", "mean", "acum")
_CAT_PREFIXES = ("Irrigated=", "Treatment=", "Previous_Crop=")
_CAT_RAW_COLS = ("Irrigated", "Treatment", "Previous_Crop")


def _safe_num_heads(hidden_dim: int, wanted_heads: int) -> int:
    heads = max(1, int(wanted_heads))
    heads = min(heads, hidden_dim)
    while heads > 1 and (hidden_dim % heads != 0):
        heads -= 1
    return max(1, heads)


def _build_env_split_spec(feature_names: Sequence[str]) -> dict:
    idx_location = []
    idx_season = []
    idx_soil = []
    idx_cat = []
    idx_other = []

    weather_by_var: dict[str, dict[str, int]] = {}
    stage_by_family: dict[str, dict[str, int]] = {}
    layered_by_family: dict[str, dict[str, dict[int, int]]] = {}

    for i, raw_name in enumerate(feature_names):
        name = str(raw_name)

        if name in _CAT_RAW_COLS or any(name.startswith(p) for p in _CAT_PREFIXES):
            idx_cat.append(i)
            continue

        if name in {"latitude", "longitude"}:
            idx_location.append(i)
            continue

        if name in {"start_month", "num_days"}:
            idx_season.append(i)
            continue

        if re.match(r"^LL__\d+$", name):
            idx_soil.append(i)
            continue

        m_weather = re.match(r"(.+?)_(min|max|mean|acum)$", name)
        if m_weather:
            var_name, stat = m_weather.group(1), m_weather.group(2)
            weather_by_var.setdefault(var_name, {})[stat] = i
            continue

        m_layered = re.match(r"(.+?)_p([A-Za-z0-9]+)_(\d+)$", name)
        if m_layered:
            fam, win, depth = m_layered.group(1), m_layered.group(2), int(m_layered.group(3))
            layered_by_family.setdefault(fam, {}).setdefault(win, {})[depth] = i
            continue

        m_stage = re.match(r"(.+?)_p([A-Za-z0-9]+)$", name)
        if m_stage:
            fam, win = m_stage.group(1), m_stage.group(2)
            stage_by_family.setdefault(fam, {})[win] = i
            continue

        idx_other.append(i)

    weather_vars = sorted(weather_by_var.keys())
    weather_index = []
    for var_name in weather_vars:
        weather_index.append([weather_by_var[var_name].get(stat, -1) for stat in _WEATHER_STATS])

    stage_families = sorted(stage_by_family.keys())
    stage_index = []
    for win in _STAGE_WINDOW_ORDER:
        stage_index.append([stage_by_family[fam].get(win, -1) for fam in stage_families])

    layered_families = sorted(layered_by_family.keys())
    max_depth = 0
    for fam in layered_families:
        for win_map in layered_by_family[fam].values():
            if win_map:
                max_depth = max(max_depth, max(win_map.keys()))

    layered_index = []
    if max_depth > 0:
        for win in _STAGE_WINDOW_ORDER:
            fam_rows = []
            for fam in layered_families:
                depth_rows = []
                depth_map = layered_by_family[fam].get(win, {})
                for depth in range(1, max_depth + 1):
                    depth_rows.append(depth_map.get(depth, -1))
                fam_rows.append(depth_rows)
            layered_index.append(fam_rows)

    return {
        "idx_static": idx_location + idx_season + idx_soil,
        "idx_cat": idx_cat,
        "idx_other": idx_other,
        "weather_index": weather_index,
        "stage_index": stage_index,
        "layered_index": layered_index,
    }


class Block(nn.Module):
    """
    Flat block with a single linear layer, layernorm, dropout and activation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.15,
        layernorm: bool = True,
    ):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if layernorm:
            self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc(x)))
        if hasattr(self, "layernorm"):
            x = self.layernorm(x)
        return x


def _make_tower(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    activation: nn.Module,
) -> Optional[nn.Module]:
    if input_dim <= 0:
        return None
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation,
        nn.Dropout(dropout),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        activation,
        nn.Dropout(dropout),
        nn.LayerNorm(hidden_dim),
    )


class E_Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.25,
        layernorm: bool = True,
        env_encoder_type: str = "flat",
        env_feature_names: Optional[Sequence[str]] = None,
        stage_n_heads: int = 4,
        stage_n_layers: int = 1,
    ):
        super().__init__()

        self.env_encoder_type = str(env_encoder_type).strip().lower()
        self.use_split = self.env_encoder_type == "split" and env_feature_names is not None
        self.activation = activation

        if not self.use_split:
            self._build_flat(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                activation=activation,
                dropout=dropout,
                layernorm=layernorm,
            )
            return

        spec = _build_env_split_spec([str(c) for c in env_feature_names])

        self.register_buffer(
            "idx_static",
            torch.tensor(spec["idx_static"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "idx_cat",
            torch.tensor(spec["idx_cat"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "idx_other",
            torch.tensor(spec["idx_other"], dtype=torch.long),
            persistent=False,
        )

        weather_index = spec["weather_index"]
        if weather_index:
            weather_tensor = torch.tensor(weather_index, dtype=torch.long)
        else:
            weather_tensor = torch.empty((0, len(_WEATHER_STATS)), dtype=torch.long)
        self.register_buffer("weather_index", weather_tensor, persistent=False)

        stage_index = spec["stage_index"]
        if stage_index and stage_index[0]:
            stage_tensor = torch.tensor(stage_index, dtype=torch.long)
        else:
            stage_tensor = torch.empty((0, 0), dtype=torch.long)
        self.register_buffer("stage_index", stage_tensor, persistent=False)

        layered_index = spec["layered_index"]
        if layered_index and layered_index[0] and layered_index[0][0]:
            layered_tensor = torch.tensor(layered_index, dtype=torch.long)
        else:
            layered_tensor = torch.empty((0, 0, 0), dtype=torch.long)
        self.register_buffer("layered_index", layered_tensor, persistent=False)

        branch_dim = hidden_dim
        self.static_tower = _make_tower(int(self.idx_static.numel()), branch_dim, dropout, activation)
        self.cat_tower = _make_tower(int(self.idx_cat.numel()), branch_dim, dropout, activation)
        self.other_tower = _make_tower(int(self.idx_other.numel()), branch_dim, dropout, activation)

        if self.weather_index.size(0) > 0:
            self.weather_proj = nn.Linear(self.weather_index.size(1), branch_dim)
            self.weather_ln = nn.LayerNorm(branch_dim)
        else:
            self.weather_proj = None

        if self.stage_index.size(0) > 0 and self.stage_index.size(1) > 0:
            self.stage_proj = nn.Linear(self.stage_index.size(1), branch_dim)
            self.stage_pos = nn.Parameter(torch.zeros(1, self.stage_index.size(0), branch_dim))
            n_heads = _safe_num_heads(branch_dim, stage_n_heads)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=branch_dim,
                nhead=n_heads,
                dim_feedforward=branch_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.stage_tf = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(stage_n_layers)))
            nn.init.normal_(self.stage_pos, std=0.02)
        else:
            self.stage_proj = None
            self.stage_pos = None
            self.stage_tf = None

        if self.layered_index.size(0) > 0 and self.layered_index.size(1) > 0 and self.layered_index.size(2) > 0:
            flat_dim = int(self.layered_index.size(1) * self.layered_index.size(2))
            self.layered_proj = nn.Linear(flat_dim, branch_dim)
            self.layered_ln = nn.LayerNorm(branch_dim)
            self.layered_conv = nn.Conv1d(branch_dim, branch_dim, kernel_size=3, padding=1)
        else:
            self.layered_proj = None
            self.layered_ln = None
            self.layered_conv = None

        n_branches = 0
        n_branches += 1 if self.static_tower is not None else 0
        n_branches += 1 if self.cat_tower is not None else 0
        n_branches += 1 if self.weather_proj is not None else 0
        n_branches += 1 if self.stage_proj is not None else 0
        n_branches += 1 if self.layered_proj is not None else 0
        n_branches += 1 if self.other_tower is not None else 0
        self.n_branches = n_branches

        if self.n_branches > 1:
            self.fuse_gate = nn.Linear(branch_dim * self.n_branches, self.n_branches)
        else:
            self.fuse_gate = None

        self.final_layer = nn.Linear(branch_dim, output_dim)
        self.final_activation = activation
        self.output_ln = nn.LayerNorm(output_dim)

    def _build_flat(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        activation: nn.Module,
        dropout: float,
        layernorm: bool,
    ):
        if n_hidden < 1:
            self.hidden_layers = nn.ModuleList()
            self.final_layer = nn.Linear(input_dim, output_dim)
            self.final_activation = activation
            self.output_ln = nn.LayerNorm(output_dim)
            return

        layers = []
        for i in range(n_hidden):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(Block(in_dim, hidden_dim, activation, dropout, layernorm))
        self.hidden_layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.final_activation = activation
        self.output_ln = nn.LayerNorm(output_dim)

    def _forward_flat(self, x):
        if hasattr(self, "hidden_layers"):
            for i, layer in enumerate(self.hidden_layers):
                if i == 0:
                    x = layer(x)
                else:
                    x = x + layer(x)
        x = self.final_activation(self.final_layer(x))
        x = self.output_ln(x)
        return x

    def _forward_split(self, x):
        branch_reprs = []

        if self.static_tower is not None and self.idx_static.numel() > 0:
            branch_reprs.append(self.static_tower(x.index_select(1, self.idx_static)))

        if self.cat_tower is not None and self.idx_cat.numel() > 0:
            branch_reprs.append(self.cat_tower(x.index_select(1, self.idx_cat)))

        if self.weather_proj is not None:
            bsz = x.size(0)
            n_vars = int(self.weather_index.size(0))
            n_stats = int(self.weather_index.size(1))
            weather_tokens = x.new_zeros((bsz, n_vars, n_stats))
            for vi in range(n_vars):
                for si in range(n_stats):
                    idx = int(self.weather_index[vi, si].item())
                    if idx >= 0:
                        weather_tokens[:, vi, si] = x[:, idx]
            weather_repr = self.weather_proj(weather_tokens)
            weather_repr = self.weather_ln(F.gelu(weather_repr))
            branch_reprs.append(weather_repr.mean(dim=1))

        if self.stage_proj is not None:
            bsz = x.size(0)
            n_windows = int(self.stage_index.size(0))
            n_families = int(self.stage_index.size(1))
            stage_tokens = x.new_zeros((bsz, n_windows, n_families))
            for wi in range(n_windows):
                for fi in range(n_families):
                    idx = int(self.stage_index[wi, fi].item())
                    if idx >= 0:
                        stage_tokens[:, wi, fi] = x[:, idx]
            stage_tokens = self.stage_proj(stage_tokens) + self.stage_pos[:, :n_windows, :]
            stage_tokens = self.stage_tf(stage_tokens)
            branch_reprs.append(stage_tokens.mean(dim=1))

        if self.layered_proj is not None:
            bsz = x.size(0)
            n_windows = int(self.layered_index.size(0))
            n_families = int(self.layered_index.size(1))
            n_depth = int(self.layered_index.size(2))
            layered_tokens = x.new_zeros((bsz, n_windows, n_families * n_depth))
            for wi in range(n_windows):
                for fi in range(n_families):
                    for di in range(n_depth):
                        idx = int(self.layered_index[wi, fi, di].item())
                        if idx >= 0:
                            layered_tokens[:, wi, fi * n_depth + di] = x[:, idx]

            layered_hidden = self.layered_proj(layered_tokens)
            layered_hidden = self.layered_ln(F.gelu(layered_hidden))
            layered_hidden = self.layered_conv(layered_hidden.transpose(1, 2)).transpose(1, 2)
            layered_hidden = F.gelu(layered_hidden)
            branch_reprs.append(layered_hidden.mean(dim=1))

        if self.other_tower is not None and self.idx_other.numel() > 0:
            branch_reprs.append(self.other_tower(x.index_select(1, self.idx_other)))

        if not branch_reprs:
            return self._forward_flat(x)

        if len(branch_reprs) == 1:
            fused = branch_reprs[0]
        else:
            stacked = torch.stack(branch_reprs, dim=1)
            gate_logits = self.fuse_gate(torch.cat(branch_reprs, dim=1))
            gates = F.softmax(gate_logits, dim=1)
            fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)

        out = self.final_activation(self.final_layer(fused))
        out = self.output_ln(out)
        return out

    def forward(self, x):
        if self.use_split:
            return self._forward_split(x)
        return self._forward_flat(x)
