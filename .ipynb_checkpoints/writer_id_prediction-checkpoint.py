    
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from neural_ode_model import ODERNNWriter


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def pick_file(fold_dir: Path, must_contain: list[str]) -> Path:
    cands = sorted(fold_dir.glob("*.pkl"))
    for p in cands:
        if all(s in p.name for s in must_contain):
            return p
    raise FileNotFoundError(
        f"No .pkl in {fold_dir} containing {must_contain}. Found: {[c.name for c in cands]}"
    )


def load_split_files(data_root: str, fold: int, split: str):
        
    fold_dir = Path(data_root) / str(fold)

    
    try:
        x_path = pick_file(fold_dir, ["all_x_dat", split, "imu", "_e"])
        y_path = pick_file(fold_dir, [f"{split}_ids", "_e"])
        return x_path, y_path
    except FileNotFoundError:
        pass

    
    x_path = pick_file(fold_dir, ["all_x_dat", split, "imu"])
    y_path = pick_file(fold_dir, [f"{split}_ids"])
    return x_path, y_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="model_saved", help="Folder containing saved .pt checkpoints")
    ap.add_argument("--ckpt", default="writerid_odernn_fold0_best.pt", help="Checkpoint filename or path")
    ap.add_argument("--data_root", default="OnHW-equations_dep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        ckpt_path = Path(args.models_dir) / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt} (also tried {ckpt_path})")

    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    id_to_idx = ckpt["id_to_idx"]
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    feat_dim = int(ckpt["feat_dim"])
    num_writers = int(ckpt["num_writers"])
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    
    ode_method = ckpt.get("ode_method", "rk4")
    ode_step_size = float(ckpt.get("ode_step_size", 1.0))
    downsample = int(ckpt.get("downsample", 1))

    
    x_path, y_path = load_split_files(args.data_root, args.fold, args.split)
    X = load_pkl(x_path)
    y = load_pkl(y_path)

    n = len(X)
    idx = args.index % max(n, 1)

    
    seq = np.asarray(X[idx], dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Sample {idx} has shape {seq.shape}, expected (T,F)")
    if seq.shape[0] <= 0:
        seq = np.zeros((1, seq.shape[1]), dtype=np.float32)

    if downsample > 1:
        seq = seq[::downsample]
        if seq.shape[0] <= 0:
            seq = np.zeros((1, seq.shape[1]), dtype=np.float32)

    seq = (seq - mean) / std

    
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    
    model = ODERNNWriter(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_writers=num_writers,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    
    x = torch.from_numpy(seq).float()
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)
    x = pad_sequence([x], batch_first=True).to(device)
    lengths = lengths.to(device)

    
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(int(args.topk), int(probs.numel()))
        topk = torch.topk(probs, k=k)

    true_id = str(list(y)[idx])
    pred_id = idx_to_id[int(topk.indices[0].item())]
    conf = float(topk.values[0].item())

    print(f"[ODE-RNN] ckpt={ckpt_path.name}")
    print(f"Split={args.split} fold={args.fold} index={idx} (requested {args.index})")
    print(f"Files: X={x_path.name} | y={y_path.name}")
    print(f"Config: downsample={downsample} | ode_method={ode_method} | ode_step_size={ode_step_size}")
    print(f"\nTrue writer ID: {true_id}")
    print(f"Pred writer ID: {pred_id} (confidence={conf:.3f})")
    print("\nTop predictions:")
    for p, j in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {idx_to_id[int(j)]}\t{p:.3f}")


if __name__ == "__main__":
    main()
