import argparse

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
    p.add_argument("--final_tf", type=str2bool, default=True)
    p.add_argument("--moe", type=str2bool, default=True)

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
    tf = "tf+" if args.final_tf else ""
    model_type = g + e + ld + tf
    model_type = model_type[:-1]
    loss_tag = args.loss if args.loss != "both" else f"both{args.alpha}"
    return (
        f"{model_type}_{loss_tag}_{args.batch_size}bs_{args.lr}lr_{args.weight_decay}wd_"
        f"{args.num_epochs}epochs_{args.early_stop}es_"
        f"{args.g_layers}g_{args.ld_layers}ld_{args.mlp_layers}mlp_{args.gxe_layers}gxe_"
        f"{args.heads}heads_{args.emb_size}emb_{args.dropout}do"    
    )