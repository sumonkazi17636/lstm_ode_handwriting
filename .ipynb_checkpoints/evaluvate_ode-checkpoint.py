import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from neural_ode_model import ODERNNWriter


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


class DS(Dataset):
    def __init__(self, X, y, mean, std, id_to_idx, downsample: int):
        self.X = X
        self.y = [id_to_idx[str(i)] for i in y]
        self.mean = mean
        self.std = std
        self.downsample = int(downsample)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)

        # ✅ MUST match training
        if self.downsample > 1:
            x = x[:: self.downsample]

        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs = pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs, lengths, ys


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ODE-RNN checkpoint (.pt)")
    ap.add_argument("--data_root", default="OnHW-equations_dep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    id_to_idx = ckpt["id_to_idx"]
    mean, std = ckpt["mean"], ckpt["std"]
    feat_dim, num_writers = ckpt["feat_dim"], ckpt["num_writers"]

    # ✅ MUST match training / checkpoint
    hidden_dim = int(ckpt.get("hidden_dim", 128))
    downsample = int(ckpt.get("downsample", 1))
    ode_method = ckpt.get("ode_method", "rk4")
    ode_step_size = float(ckpt.get("ode_step_size", 1.0))

    fold_dir = Path(args.data_root) / str(args.fold)
    Xv = load_pkl(fold_dir / "all_x_dat_val_imu_e.pkl")
    yv = load_pkl(fold_dir / "val_ids_e.pkl")

    ds = DS(Xv, yv, mean, std, id_to_idx, downsample=downsample)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ODERNNWriter(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_writers=num_writers,
        ode_method=ode_method,
        ode_step_size=ode_step_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct = 0
    total = 0
    for x, lengths, y in dl:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        pred = model(x, lengths).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()

    print(f"ODE-RNN Fold{args.fold} VAL accuracy: {correct/total*100:.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    main()
