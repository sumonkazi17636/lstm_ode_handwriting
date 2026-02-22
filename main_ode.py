import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from neural_ode_model import ODERNNWriter


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_mean_std(list_of_arrays):
    
    count = 0
    sum_ = None
    sumsq = None

    for a in list_of_arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"Expected (T,F), got shape {a.shape}")

        if sum_ is None:
            sum_ = np.zeros(a.shape[1], dtype=np.float64)
            sumsq = np.zeros(a.shape[1], dtype=np.float64)

        sum_ += a.sum(axis=0)
        sumsq += (a * a).sum(axis=0)
        count += a.shape[0]

    mean = sum_ / max(count, 1)
    var = sumsq / max(count, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class WriterIMUDataset(Dataset):
    def __init__(self, X, ids, id_to_idx=None, mean=None, std=None, downsample=10):
        self.X = X
        self.ids = [str(i) for i in ids]
        assert len(self.X) == len(self.ids), "X and ids length mismatch"

        uniq = sorted(set(self.ids))
        self.id_to_idx = {pid: k for k, pid in enumerate(uniq)} if id_to_idx is None else id_to_idx

        self.mean = mean
        self.std = std
        self.downsample = int(downsample)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected (T,F), got shape {x.shape}")

        
        if self.downsample > 1:
            x = x[:: self.downsample]

        
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        y = torch.tensor(self.id_to_idx[self.ids[i]], dtype=torch.long)
        return torch.from_numpy(x), y


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad = pad_sequence(xs, batch_first=True)  # (B,Tmax,F)
    ys = torch.stack(ys)
    return xs_pad, lengths, ys


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = ce(logits, y)

        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="OnHW-equations_dep")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--downsample", type=int, default=10)
    ap.add_argument("--ode_method", type=str, default="rk4", choices=["rk4", "euler", "dopri5"])
    ap.add_argument("--ode_step_size", type=float, default=1.0)
    args = ap.parse_args()

    fold_dir = Path(args.data_root) / str(args.fold)
    xtr = load_pkl(fold_dir / "all_x_dat_train_imu_e.pkl")
    ytr = load_pkl(fold_dir / "train_ids_e.pkl")
    xva = load_pkl(fold_dir / "all_x_dat_val_imu_e.pkl")
    yva = load_pkl(fold_dir / "val_ids_e.pkl")

    
    mean, std = compute_mean_std(xtr)

    
    train_ds = WriterIMUDataset(xtr, ytr, mean=mean, std=std, downsample=args.downsample)
    val_ds = WriterIMUDataset(
        xva, yva, id_to_idx=train_ds.id_to_idx, mean=mean, std=std, downsample=args.downsample
    )

    num_writers = len(train_ds.id_to_idx)
    feat_dim = int(np.asarray(xtr[0]).shape[1])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ODERNNWriter(
        feat_dim=feat_dim,
        hidden_dim=args.hidden,
        num_writers=num_writers,
        ode_method=args.ode_method,
        ode_step_size=args.ode_step_size,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    best_acc = -1.0
    best_epoch = -1
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += float(loss.item()) * y.size(0)
            seen += y.size(0)

        tr_loss = running / max(seen, 1)
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc*100:.2f}%")

        sched.step(va_acc)

        current_lr = opt.param_groups[0]["lr"]
        print(f"  Current LR: {current_lr:.6f}")

        
        ckpt_path = Path(f"writerid_odernn_new_fold{args.fold}_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "val_acc": va_acc,
                "model_state": model.state_dict(),
                "id_to_idx": train_ds.id_to_idx,
                "mean": mean,
                "std": std,
                "feat_dim": feat_dim,
                "num_writers": num_writers,
                "hidden_dim": args.hidden,
                "downsample": args.downsample,
                "ode_method": args.ode_method,
                "ode_step_size": args.ode_step_size,
                "model_type": "ODERNNWriter",
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint -> {ckpt_path}")

        
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            best_ckpt = ckpt_path

    print("\n================ TRAINING SUMMARY ================")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    print(f"Achieved at epoch: {best_epoch}")
    print(f"Best checkpoint: {best_ckpt}")
    print("==================================================")


if __name__ == "__main__":
    main()
