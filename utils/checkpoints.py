import json
from pathlib import Path


def resolve_checkpoint_path(ckpt_root: Path | str, checkpoint_tag: str | None = None) -> Path:
    """Resolve a checkpoint alias or the newest numeric checkpoint in a run directory."""
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_root} does not exist.")

    best_path = None
    if checkpoint_tag:
        manifest_path = ckpt_root / "checkpoint_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            alias_info = manifest.get("aliases", {}).get(checkpoint_tag)
            if alias_info and alias_info.get("path"):
                best_path = Path(alias_info["path"])

        if best_path is None:
            alias_path = ckpt_root / f"{checkpoint_tag}.pt"
            if alias_path.exists():
                best_path = alias_path

        if best_path is None:
            raise FileNotFoundError(f"Could not resolve checkpoint_tag={checkpoint_tag!r} in {ckpt_root}.")
        return best_path

    ckpts = sorted([p for p in ckpt_root.glob("checkpoint_*.pt") if p.is_file()])
    if ckpts:
        return ckpts[-1]

    latest_alias = ckpt_root / "latest.pt"
    if latest_alias.exists():
        return latest_alias

    raise FileNotFoundError(f"No checkpoint files found in {ckpt_root}.")
