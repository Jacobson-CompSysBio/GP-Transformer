import argparse
import numpy as np
from dataclasses import dataclass

@dataclass
class LabelScaler:
    mean: float
    std: float
    def transform(self, x):
        return (np.asarray(x) - self.mean) / (self.std + 1e-8)
    def inverse(self, z):
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
    p.add_argument("--residual", type=str2bool, default=False)

    p.add_argument("--detach_ymean", type=str2bool, default=True)
    p.add_argument("--lambda_ymean", type=float, default=0.5)
    p.add_argument("--lambda_resid", type=float, default=5.0)

    p.add_argument("--batch_size", type=int, default=32)
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
    p.add_argument("--scale_targets", type=str2bool, default=True)

    p.add_argument("--loss", type=str, default="mse",
                   choices=["mse", "pcc", "both"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight for MSE when --loss both (loss = alpha*MSE + (1-alpha)*(1-PCC))")
    p.add_argument('--checkpoint_dir', type=str, required=False,
                   help='Directory from train.py for this run')
    return p.parse_args()

def make_run_name(args) -> str:
    g = "g+" if args.g_enc else ""
    e = "e+" if args.e_enc else ""
    ld = "ld+" if args.ld_enc else ""
    moe = "moe+" if args.moe else ""
    res = "res+" if args.residual else ""
    if args.gxe_enc in ["tf", "mlp", "cnn"]:
        gxe = f"{args.gxe_enc}+"
    else:
        gxe = ""
    model_type = g + e + ld + gxe + moe + res
    model_type = model_type[:-1]
    loss_tag = args.loss if args.loss != "both" else f"both{args.alpha}"
    scale_targets = "scaled" if args.scale_targets else ""
    return (
        f"{model_type}_{loss_tag}_{args.batch_size}bs_{args.lr}lr_{args.weight_decay}wd_"
        f"{args.num_epochs}epochs_{args.early_stop}es_"
        f"{args.g_layers}g_{args.ld_layers}ld_{args.mlp_layers}mlp_{args.gxe_layers}gxe_"
        f"{args.heads}heads_{args.emb_size}emb_{args.dropout}do_{scale_targets}"    
    )