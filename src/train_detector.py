"""Treino do detector SNN nos auto-labels. Uso: python -m src.train_detector"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .snn_detector import EventDetectorSNN


class DetDataset(Dataset):
    def __init__(self, root="data"):
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise SystemExit("sem dados — rode src.auto_label")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        return (torch.from_numpy(d["x"]), torch.from_numpy(d["hm"]),
                torch.from_numpy(d["size"]), torch.tensor(float("size" in d)))


def focal_loss(pred, target, alpha=2, beta=4):
    """Focal loss de heatmap (CornerNet-style)."""
    pos = target == 1
    neg = ~pos
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    loss = -((1 - pred) ** alpha * torch.log(pred) * pos
             + (1 - target) ** beta * pred ** alpha * torch.log(1 - pred) * neg)
    return loss.sum() / max(pos.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    ds = DetDataset()
    n_te = max(1, int(0.15 * len(ds)))
    tr, te = random_split(ds, [len(ds) - n_te, n_te],
                          generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    dl_te = DataLoader(te, batch_size=32)
    print(f"device={device} train={len(tr)} test={len(te)}")

    model = EventDetectorSNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, hm, sz, _ in dl:
            x, hm, sz = x.to(device), hm.to(device), sz.to(device)
            phm, psz = model(x)
            loss = focal_loss(phm.squeeze(1), hm)
            pos = hm > 0.5                       # (B,15,20) celulas com centro
            if pos.any():
                loss = loss + 2.0 * nn.functional.l1_loss(
                    psz.permute(0, 2, 3, 1)[pos], sz.permute(0, 2, 3, 1)[pos])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        tl = 0.0
        with torch.no_grad():
            for x, hm, sz, _ in dl_te:
                x, hm, sz = x.to(device), hm.to(device), sz.to(device)
                phm, psz = model(x)
                tl += focal_loss(phm.squeeze(1), hm).item()
        tl /= len(dl_te)
        if tl < best:
            best = tl
            torch.save(model.state_dict(), "checkpoints/snn_detector.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: test_loss={tl:.4f} (best={best:.4f})")
    print(f"salvo | best={best:.4f}")


if __name__ == "__main__":
    main()
