import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


from neural_ode_model import ODERNNWriter



def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def find_fold_files(data_root: Path, fold: int):
    
    fold_dir = data_root / str(fold)

    
    x_val_e = fold_dir / "all_x_dat_val_imu_e.pkl"
    y_val_e = fold_dir / "val_ids_e.pkl"

    x_val = fold_dir / "all_x_dat_val_imu.pkl"
    y_val = fold_dir / "val_ids.pkl"

    if x_val_e.exists() and y_val_e.exists():
        return x_val_e, y_val_e
    elif x_val.exists() and y_val.exists():
        return x_val, y_val
    else:
        raise FileNotFoundError(
            f"Could not find validation files in {fold_dir}. "
            f"Expected one of: (all_x_dat_val_imu_e.pkl, val_ids_e.pkl) or (all_x_dat_val_imu.pkl, val_ids.pkl)."
        )


def infer_fold_from_name(name: str) -> int | None:
    
    for f in [0, 1, 2, 3, 4, 5]:
        if f"fold{f}" in name:
            return f
    return None


def infer_dataset_from_name(name: str) -> str | None:
   
    if "words500" in name.lower():
        return "OnHW-words500_dep"
    if "equations" in name.lower():
        return "OnHW-equations_dep"
    
    if "odernn_new" in name.lower():
        return "OnHW-equations_dep"
    
    if "writerid_bilstm_fold" in name.lower():
        return "OnHW-equations_dep"
    return None



class SeqDS(Dataset):
    def __init__(self, X, y_ids, mean, std, id_to_idx, downsample: int):
        self.X = X
        self.y = [id_to_idx[str(i)] for i in y_ids]
        self.mean = mean
        self.std = std
        self.downsample = int(downsample)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32)

        
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array (T,F), got {x.shape}")
        if x.shape[0] <= 0:
            
            x = np.zeros((1, x.shape[1]), dtype=np.float32)

        
        if self.downsample > 1:
            x = x[:: self.downsample]
            if x.shape[0] <= 0:
                x = np.zeros((1, x.shape[1]), dtype=np.float32)

        
        x = (x - self.mean) / self.std

        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    xs_pad = pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs_pad, lengths, ys


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



@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, data_root_base: Path, batch: int = 64, device: str = "cuda"):
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    
    model_type = ckpt.get("model_type", "").lower()
    name = ckpt_path.name.lower()

    is_ode = ("odernn" in model_type) or ("odernn" in name) or ("ode" in model_type)
    is_lstm = ("bilstm" in model_type) or ("bilstm" in name) or ("lstm" in model_type)

    if not (is_ode or is_lstm):
        is_ode = "ode_method" in ckpt or "ode_step_size" in ckpt

    
    fold = ckpt.get("fold", None)
    if fold is None:
        fold = infer_fold_from_name(ckpt_path.name)
    if fold is None:
        raise ValueError(f"Could not infer fold from checkpoint {ckpt_path.name}")

    ds_name = ckpt.get("data_root", None)
    if ds_name is None or not isinstance(ds_name, str):
        ds_name = infer_dataset_from_name(ckpt_path.name)
    if ds_name is None:
        ds_name = "OnHW-equations_dep"  # safest default

    data_root = data_root_base / ds_name

    
    x_val_path, y_val_path = find_fold_files(data_root, fold)
    Xv = load_pkl(x_val_path)
    yv = load_pkl(y_val_path)

    
    id_to_idx = ckpt["id_to_idx"]
    mean = ckpt["mean"]
    std = ckpt["std"]
    feat_dim = int(ckpt["feat_dim"])
    num_writers = int(ckpt["num_writers"])
    downsample = int(ckpt.get("downsample", 1))

    ds = SeqDS(Xv, yv, mean, std, id_to_idx, downsample=downsample)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate)

    
    if is_ode:
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        ode_method = ckpt.get("ode_method", "rk4")
        ode_step_size = float(ckpt.get("ode_step_size", 1.0))

        model = ODERNNWriter(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_writers=num_writers,
            ode_method=ode_method,
            ode_step_size=ode_step_size,
        )
    else:
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        num_layers = int(ckpt.get("num_layers", 2))
        model = BiLSTMWriter(
            feat_dim=feat_dim,
            hidden=hidden_dim,
            num_layers=num_layers,
            num_writers=num_writers,
        )

    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct = 0
    total = 0
    for x, lengths, y in dl:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    acc = 100.0 * correct / max(total, 1)

    model_label = "ODE-RNN" if is_ode else "BiLSTM"
    return {
        "checkpoint": ckpt_path.name,
        "model": model_label,
        "dataset": ds_name,
        "fold": fold,
        "val_acc": acc,
        "val_correct": correct,
        "val_total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default="model_saved", help="Folder containing .pt checkpoints")
    ap.add_argument("--data_root_base", type=str, default=".", help="Base path containing dataset folders")
    ap.add_argument("--ckpt", type=str, default=None, help="Evaluate only one checkpoint (optional)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    data_root_base = Path(args.data_root_base)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            ckpt_path = models_dir / args.ckpt
        res = eval_checkpoint(ckpt_path, data_root_base, batch=args.batch, device=args.device)
        print(f"\n {res['model']} | {res['dataset']} | fold {res['fold']} | acc={res['val_acc']:.2f}% "
              f"({res['val_correct']}/{res['val_total']}) | {res['checkpoint']}\n")
        return

    ckpts = sorted(models_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {models_dir}")

    results = []
    for ckpt_path in ckpts:
        try:
            results.append(eval_checkpoint(ckpt_path, data_root_base, batch=args.batch, device=args.device))
        except Exception as e:
            print(f"⚠️ Skipped {ckpt_path.name}: {e}")

    
    print("\n==================== EVALUATION SUMMARY ====================")
    print(f"{'MODEL':<8}  {'DATASET':<18}  {'FOLD':<4}  {'ACC(%)':<7}  {'CKPT'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["dataset"], x["model"], x["fold"])):
        print(f"{r['model']:<8}  {r['dataset']:<18}  {r['fold']:<4}  {r['val_acc']:<7.2f}  {r['checkpoint']}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
