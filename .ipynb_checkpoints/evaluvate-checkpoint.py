import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


class DS(Dataset):
    def __init__(self, X, y, mean, std, id_to_idx):
        self.X = X
        self.y = [id_to_idx[str(i)] for i in y]
        self.mean = mean
        self.std = std

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)
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
    ckpt = torch.load("writerid_bilstm_fold0.pt", map_location="cpu", weights_only=False)
    id_to_idx = ckpt["id_to_idx"]
    mean, std = ckpt["mean"], ckpt["std"]
    feat_dim, num_writers = ckpt["feat_dim"], ckpt["num_writers"]

    fold_dir = Path("OnHW-equations_dep") / "0"
    Xv = load_pkl(fold_dir / "all_x_dat_val_imu_e.pkl")
    yv = load_pkl(fold_dir / "val_ids_e.pkl")

    ds = DS(Xv, yv, mean, std, id_to_idx)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMWriter(feat_dim=feat_dim, hidden=128, num_layers=2, num_writers=num_writers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct = 0
    total = 0
    for x, lengths, y in dl:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        pred = model(x, lengths).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()

    print(f"Fold0 VAL accuracy: {correct/total*100:.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    main()
