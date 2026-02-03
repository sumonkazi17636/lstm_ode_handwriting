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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="writerid_odernn_fold0.pt")
    ap.add_argument("--data_root", default="OnHW-equations_dep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    id_to_idx = ckpt["id_to_idx"]
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mean, std = ckpt["mean"], ckpt["std"]
    feat_dim, num_writers = ckpt["feat_dim"], ckpt["num_writers"]
    hidden_dim = ckpt.get("hidden_dim", 128)

    fold_dir = Path(args.data_root) / str(args.fold)
    if args.split == "val":
        X = load_pkl(fold_dir / "all_x_dat_val_imu_e.pkl")
        y = load_pkl(fold_dir / "val_ids_e.pkl")
    else:
        X = load_pkl(fold_dir / "all_x_dat_train_imu_e.pkl")
        y = load_pkl(fold_dir / "train_ids_e.pkl")

    n = len(X)
    idx = args.index % n

    seq = np.asarray(X[idx], dtype=np.float32)
    seq = (seq - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ODERNNWriter(feat_dim=feat_dim, hidden_dim=hidden_dim, num_writers=num_writers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(seq).float()
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)
    x = pad_sequence([x], batch_first=True).to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(args.topk, probs.numel())
        topk = torch.topk(probs, k=k)

    true_id = str(list(y)[idx])
    pred_id = idx_to_id[int(topk.indices[0].item())]
    conf = float(topk.values[0].item())

    print(f"[ODE-RNN] Split={args.split} fold={args.fold} index={idx} (requested {args.index})")
    print(f"True writer ID: {true_id}")
    print(f"Pred writer ID: {pred_id} (confidence={conf:.3f})")
    print("\nTop predictions:")
    for p, j in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {idx_to_id[int(j)]}\t{p:.3f}")


if __name__ == "__main__":
    main()
