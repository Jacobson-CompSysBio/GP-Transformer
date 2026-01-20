import argparse
import torch
import random
import numpy as np
from dataclasses import dataclass

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
    p.add_argument("--moe", type=str2bool, default=True)
    p.add_argument("--full_transformer", type=str2bool, default=False)
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
    
    g = "g+" if args.g_enc else ""
    e = "e+" if args.e_enc else ""
    ld = "ld+" if args.ld_enc else ""
    full = "fulltf+" if getattr(args, "full_transformer", False) else ""
    moe = "moe+" if args.moe else ""
    res = "res+" if args.residual else ""
    
    if args.gxe_enc in ["tf", "mlp", "cnn"]:
        gxe = f"{args.gxe_enc}+"
    else:
        gxe = ""

    model_type = (full + g + e + ld + gxe + moe + res).rstrip("+")

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
    return (
        f"{model_type}_{loss_tag}_{args.gbs}gbs_{args.lr}lr_{args.weight_decay}wd_"
        f"{args.num_epochs}epochs_{args.early_stop}es_"
        f"{args.g_layers}g_{args.ld_layers}ld_{args.mlp_layers}mlp_{args.gxe_layers}gxe_"
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
