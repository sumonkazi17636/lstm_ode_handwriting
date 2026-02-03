import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class BiLSTMWriter(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=2, num_writers=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, num_writers),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(h_last)


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="writerid_bilstm_fold0.pt")
    ap.add_argument("--data_root", default="OnHW-equations_dep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    id_to_idx = ckpt["id_to_idx"]
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mean = ckpt["mean"]
    std = ckpt["std"]
    feat_dim = ckpt["feat_dim"]
    num_writers = ckpt["num_writers"]

    fold_dir = Path(args.data_root) / str(args.fold)

    if args.split == "val":
        X = load_pkl(fold_dir / "all_x_dat_val_imu_e.pkl")
        y = load_pkl(fold_dir / "val_ids_e.pkl")
    else:
        X = load_pkl(fold_dir / "all_x_dat_train_imu_e.pkl")
        y = load_pkl(fold_dir / "train_ids_e.pkl")

    n = len(X)
    idx = args.index % n  # wrap safely

    seq = np.asarray(X[idx], dtype=np.float32)
    seq = (seq - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMWriter(feat_dim=feat_dim, hidden=128, num_layers=2, num_writers=num_writers).to(device)
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

    print(f"Split={args.split} fold={args.fold} index={idx} (requested {args.index})")
    print(f"True writer ID: {true_id}")
    print(f"Pred writer ID: {pred_id} (confidence={conf:.3f})")
    print("\nTop predictions:")
    for p, j in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {idx_to_id[int(j)]}\t{p:.3f}")


if __name__ == "__main__":
    main()
