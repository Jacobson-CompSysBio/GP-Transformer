import argparse
import os
import torch
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterator, List, Optional
from torch.utils.data import Sampler
import torch.distributed as dist

@dataclass
class LabelScaler:
    mean: float
    std: float
    def transform(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return (np.asarray(x) - self.mean) / (self.std + 1e-8)
    def inverse_transform(self, z):
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        return np.asarray(z) * (self.std + 1e-8) + self.mean

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--g_enc", type=str2bool, default=True)
    p.add_argument("--e_enc", type=str2bool, default=True)
    p.add_argument("--ld_enc", type=str2bool, default=True)
    p.add_argument("--gxe_enc", type=str, default=True)
    p.add_argument("--wg", type=str2bool, default=True, help="Weighted gate for 3-prong architecture")
    p.add_argument("--g_encoder_type", type=str, default=None)
    p.add_argument("--moe_num_experts", type=int, default=None)
    p.add_argument("--moe_top_k", type=int, default=None)
    p.add_argument("--moe_expert_hidden_dim", type=int, default=None)
    p.add_argument("--moe_shared_expert", type=str2bool, default=None)
    p.add_argument("--moe_shared_expert_hidden_dim", type=int, default=None)
    p.add_argument("--moe_loss_weight", type=float, default=None)
    p.add_argument("--full_transformer", type=str2bool, default=False)
    p.add_argument("--full_tf_mlp_type", type=str, default=None)
    p.add_argument("--residual", type=str2bool, default=False)

    p.add_argument("--detach_ymean", type=str2bool, default=True)
    p.add_argument("--lambda_ymean", type=float, default=0.5)
    p.add_argument("--lambda_resid", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gbs", type=int, default=2048)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--early_stop", type=int, default=50)

    p.add_argument("--g_layers", type=int, default=1)
    p.add_argument("--ld_layers", type=int, default=1)
    p.add_argument("--mlp_layers", type=int, default=1)
    p.add_argument("--gxe_layers", type=int, default=4)

    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--scale_targets", type=str2bool, default=False)

    p.add_argument("--loss", type=str, default="pcc",
                   help="composite loss string, e.g. 'mse+envpcc'")
    p.add_argument("--loss_weights", type=str, default="1.0",
                   help="comma separated list of weights for each loss, e.g. '1.0,0.5'")
    p.add_argument("--contrastive_loss", type=str2bool, default=False,
                   help="Add genomic contrastive loss to encourage G embeddings to reflect genetic similarity")
    p.add_argument("--contrastive_weight", type=float, default=0.1,
                   help="Weight for genomic contrastive loss (default 0.1)")
    p.add_argument("--contrastive_temperature", type=float, default=0.1,
                   help="Temperature for contrastive loss softmax (default 0.1)")
    p.add_argument("--env_stratified", type=str2bool, default=False,
                   help="Use environment-stratified sampling for envwise losses (recommended for envpcc)")
    p.add_argument("--min_samples_per_env", type=int, default=32,
                   help="Minimum samples per environment in each batch for env-stratified sampling")
    p.add_argument("--leo_val", type=str2bool, default=False,
                   help="Use Leave-Environment-Out validation (hold out entire environments, not years)")
    p.add_argument("--leo_val_fraction", type=float, default=0.15,
                   help="Fraction of environments to hold out for LEO validation (default 0.15)")
    p.add_argument('--checkpoint_dir', type=str, required=False,
                   help='Directory from train.py for this run')
    return p.parse_args()

def make_run_name(args) -> str:
    # helper to shorten float
    def short(x):
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    def _get_arg_env(attr, env_key, default=None, cast=None):
        val = getattr(args, attr, None)
        if val is None:
            env_val = os.getenv(env_key)
            if env_val is None or env_val == "":
                return default
            return cast(env_val) if cast is not None else env_val
        return val
    
    full_transformer = bool(getattr(args, "full_transformer", False))
    g = "g+" if args.g_enc and not full_transformer else ""
    e = "e+" if args.e_enc and not full_transformer else ""
    ld = "ld+" if args.ld_enc and not full_transformer else ""
    full = "fulltf+" if full_transformer else ""
    wg = "wg+" if args.wg and not full_transformer else ""
    res = "res+" if args.residual else ""
    strat = "strat+" if getattr(args, "env_stratified", False) else ""
    leo = "leo+" if getattr(args, "leo_val", False) else ""
    contr = "contr+" if getattr(args, "contrastive_loss", False) else ""
    
    if (not full_transformer) and (args.gxe_enc in ["tf", "mlp", "cnn"]):
        gxe = f"{args.gxe_enc}+"
    else:
        gxe = ""

    model_type = (full + g + e + ld + gxe + wg + res + strat + leo + contr).rstrip("+")

    # optional MoE encoder tag
    g_encoder_type = _get_arg_env("g_encoder_type", "G_ENCODER_TYPE", "dense", str)
    if isinstance(g_encoder_type, str):
        g_encoder_type = g_encoder_type.lower()
    else:
        g_encoder_type = "moe" if g_encoder_type else "dense"

    full_tf_mlp_type = _get_arg_env("full_tf_mlp_type", "FULL_TF_MLP_TYPE", None, str)
    if full_transformer:
        if full_tf_mlp_type is None:
            full_tf_mlp_type = g_encoder_type
        if isinstance(full_tf_mlp_type, str):
            full_tf_mlp_type = full_tf_mlp_type.lower()
        else:
            full_tf_mlp_type = "moe" if full_tf_mlp_type else "dense"

    mlp_type_for_tag = full_tf_mlp_type if full_transformer else g_encoder_type
    moe_tag = ""
    if mlp_type_for_tag == "moe":
        moe_num_experts = _get_arg_env("moe_num_experts", "MOE_NUM_EXPERTS", 4, int)
        moe_top_k = _get_arg_env("moe_top_k", "MOE_TOP_K", 2, int)
        moe_expert_hidden_dim = _get_arg_env("moe_expert_hidden_dim", "MOE_EXPERT_HIDDEN_DIM", None, int)
        moe_shared_expert = _get_arg_env("moe_shared_expert", "MOE_SHARED_EXPERT", False, str2bool)
        moe_shared_expert_hidden_dim = _get_arg_env(
            "moe_shared_expert_hidden_dim", "MOE_SHARED_EXPERT_HIDDEN_DIM", None, int
        )
        moe_loss_weight = _get_arg_env("moe_loss_weight", "MOE_LOSS_WEIGHT", 0.01, float)

        moe_tag = f"moeenc{moe_num_experts}e{moe_top_k}k"
        if moe_expert_hidden_dim is not None:
            moe_tag += f"{moe_expert_hidden_dim}h"
        if moe_shared_expert:
            shared_dim = moe_shared_expert_hidden_dim if moe_shared_expert_hidden_dim is not None else "auto"
            moe_tag += f"_shared{shared_dim}h"
        moe_tag += f"_lb{short(moe_loss_weight)}"

    full_tag = ""

    # loss tag
    terms = [t.strip().lower() for t in args.loss.split("+")]
    if args.loss_weights is not None:
        weights = [short(w) for w in args.loss_weights.split(",")]
    else:
        weights = ["1"] * len(terms)
    
    # prettier tags: omit weights if single-term with weight 1
    if len(terms) == 1 and weights[0] == "1":
        loss_tag = terms[0]
    else:
        weight_tag = "-".join(weights)
        loss_tag = f"{weight_tag}w_" + "-".join(terms)
    
    scale_targets = "_scaled" if args.scale_targets else ""
    layer_tag = (
        f"{args.gxe_layers}gxe_"
        if full_transformer
        else f"{args.g_layers}g_{args.ld_layers}ld_{args.mlp_layers}mlp_{args.gxe_layers}gxe_"
    )
    return (
        f"{model_type}"
        f"{'_' + moe_tag if moe_tag else ''}"
        f"{'_' + full_tag if full_tag else ''}"
        f"_{loss_tag}_{args.gbs}gbs_{args.lr}lr_{args.weight_decay}wd_"
        f"{args.num_epochs}epochs_{args.early_stop}es_"
        f"{layer_tag}"
        f"{args.heads}heads_{args.emb_size}emb_{args.dropout}do{scale_targets}"
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # optional, but slower
    # torch.use_deterministic_algorithms(True)

# need a seed function for dataloader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # each worker gets a distinct initial seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EnvStratifiedSampler(Sampler[List[int]]):
    """
    Environment-stratified sampler for DDP training.
    
    Creates batches that pack multiple environments together, ensuring each
    environment has at least `min_samples_per_env` samples for reliable 
    within-environment correlation.
    
    OPTIMIZED: Pre-computes batch structure once at init, only shuffles per epoch.
    Uses numpy for fast array operations.
    
    Algorithm:
        1. For each environment, split samples into chunks of `min_samples_per_env`
        2. Pack chunks from different environments into batches of `batch_size`
        3. Each batch now has ~(batch_size / min_samples_per_env) environments,
           each with at least `min_samples_per_env` samples
    
    Usage:
        sampler = EnvStratifiedSampler(
            env_ids=train_ds.env_id_tensor.tolist(),
            batch_size=256,
            shuffle=True,
            rank=rank,
            world_size=world_size,
            min_samples_per_env=32,
        )
        loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4)
    """
    
    def __init__(
        self,
        env_ids: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        drop_last: bool = False,
        min_samples_per_env: int = 32,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.min_samples_per_env = min_samples_per_env
        self.epoch = 0
        
        # DDP settings
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = rank
            self.world_size = world_size if world_size is not None else 1
        
        # Convert to numpy for speed
        env_ids_np = np.array(env_ids)
        
        # Pre-compute chunk structure (done ONCE at init, not every epoch)
        # Group indices by environment using numpy
        unique_envs = np.unique(env_ids_np)
        self.env_indices = {env: np.where(env_ids_np == env)[0] for env in unique_envs}
        
        # Pre-compute which environments have enough samples
        self.valid_envs = [env for env, idx in self.env_indices.items() 
                          if len(idx) >= min_samples_per_env]
        
        # Pre-compute chunk boundaries for each env (start_idx, end_idx pairs)
        self.env_chunks = {}
        for env in self.valid_envs:
            n = len(self.env_indices[env])
            n_full_chunks = n // min_samples_per_env
            self.env_chunks[env] = n_full_chunks
        
        # Total number of chunks
        self.total_chunks = sum(self.env_chunks.values())
        
        # Pre-compute batch structure (how many chunks per batch)
        self.chunks_per_batch = batch_size // min_samples_per_env
        self.n_batches_total = (self.total_chunks + self.chunks_per_batch - 1) // self.chunks_per_batch
        
        # For DDP: each rank gets a subset
        self.n_batches = (self.n_batches_total + self.world_size - 1) // self.world_size
        
        # Build initial batches
        self._current_batches = None
    
    def _build_batches(self):
        """Build batches for current epoch - optimized with numpy."""
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Step 1: Create all chunks with shuffled indices within each env
        all_chunks = []
        for env in self.valid_envs:
            indices = self.env_indices[env].copy()
            if self.shuffle:
                rng.shuffle(indices)
            
            # Split into chunks
            n_chunks = self.env_chunks[env]
            for i in range(n_chunks):
                start = i * self.min_samples_per_env
                end = start + self.min_samples_per_env
                all_chunks.append(indices[start:end])
        
        # Step 2: Shuffle chunk order
        if self.shuffle:
            chunk_order = rng.permutation(len(all_chunks))
            all_chunks = [all_chunks[i] for i in chunk_order]
        
        # Step 3: Pack chunks into batches using numpy concatenation
        all_batches = []
        for i in range(0, len(all_chunks), self.chunks_per_batch):
            batch_chunks = all_chunks[i:i + self.chunks_per_batch]
            if batch_chunks:
                batch = np.concatenate(batch_chunks)
                if not self.drop_last or len(batch) >= self.batch_size // 2:
                    all_batches.append(batch.tolist())
        
        # Step 4: Shuffle batch order
        if self.shuffle:
            batch_order = rng.permutation(len(all_batches))
            all_batches = [all_batches[i] for i in batch_order]
        
        # Step 5: Pad to ensure all ranks have same number of batches (DDP requirement)
        # Without this, ranks with fewer batches finish early and deadlock on gradient sync
        n_total = len(all_batches)
        batches_per_rank = (n_total + self.world_size - 1) // self.world_size  # ceil division
        n_padded = batches_per_rank * self.world_size
        
        # Pad by repeating batches from the beginning
        if n_padded > n_total:
            all_batches = all_batches + all_batches[:n_padded - n_total]
        
        # Distribute to this rank - now guaranteed equal count
        self._current_batches = all_batches[self.rank::self.world_size]
    
    def __iter__(self) -> Iterator[List[int]]:
        self._build_batches()
        return iter(self._current_batches)
    
    def __len__(self) -> int:
        # Approximate length (exact after first iteration)
        if self._current_batches is not None:
            return len(self._current_batches)
        return self.n_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling (call before each epoch like DistributedSampler)."""
        self.epoch = epoch


class HybridEnvSampler(Sampler[int]):
    """
    Hybrid sampler that mixes environment-stratified and random sampling.
    
    This gives a balance between:
    - Environment-stratified: Good for envwise_pcc, but may reduce diversity
    - Random: Good for MSE/global metrics, but bad for envwise correlation
    
    Usage:
        sampler = HybridEnvSampler(
            env_ids=train_ds.env_id_tensor.tolist(),
            batch_size=256,
            env_batch_ratio=0.5,  # 50% env-stratified, 50% random
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    """
    
    def __init__(
        self,
        env_ids: List[int],
        total_samples: int,
        env_batch_ratio: float = 0.5,
        shuffle: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.env_ids = env_ids
        self.total_samples = total_samples
        self.env_batch_ratio = env_batch_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # DDP settings
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = rank
            self.world_size = world_size if world_size is not None else 1
        
        # Group indices by environment
        self.env_to_indices = defaultdict(list)
        for idx, env in enumerate(env_ids):
            self.env_to_indices[env].append(idx)
        
        self.num_samples = (total_samples + self.world_size - 1) // self.world_size
    
    def _generate_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = []
        envs = list(self.env_to_indices.keys())
        
        # Env-stratified portion
        n_env_samples = int(self.total_samples * self.env_batch_ratio)
        if self.shuffle:
            perm = torch.randperm(len(envs), generator=g).tolist()
            envs = [envs[i] for i in perm]
        
        env_idx = 0
        while len(indices) < n_env_samples:
            env = envs[env_idx % len(envs)]
            env_indices = self.env_to_indices[env]
            # Add all samples from this env (or remaining needed)
            needed = min(len(env_indices), n_env_samples - len(indices))
            if self.shuffle:
                perm = torch.randperm(len(env_indices), generator=g)[:needed].tolist()
                indices.extend([env_indices[i] for i in perm])
            else:
                indices.extend(env_indices[:needed])
            env_idx += 1
        
        # Random portion
        n_random = self.total_samples - len(indices)
        if n_random > 0:
            all_indices = list(range(self.total_samples))
            perm = torch.randperm(len(all_indices), generator=g)[:n_random].tolist()
            indices.extend([all_indices[i] for i in perm])
        
        # Shuffle all together
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        
        # Distribute to ranks
        indices = indices[self.rank::self.world_size]
        return indices
    
    def __iter__(self) -> Iterator[int]:
        return iter(self._generate_indices())
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch

