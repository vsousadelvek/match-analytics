"""Treino da SNN classificadora de blobs. Uso: python -m src.train_player"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .player_snn import PlayerSNN


class DS(Dataset):
    def __init__(self):
        d = np.load("data_players/players.npz")
        self.x = torch.from_numpy(d["x"])
        self.y = torch.from_numpy(d["y"]).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    ds = DS()
    n_te = max(1, int(0.15 * len(ds)))
    tr, te = random_split(ds, [len(ds) - n_te, n_te],
                          generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=64, shuffle=True)
    dl_te = DataLoader(te, batch_size=128)
    print(f"train={len(tr)} test={len(te)} device={device}")

    model = PlayerSNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor([5.0, 1.0], device=device)   # ruido eh minoria ~1:5
    crit = nn.CrossEntropyLoss(weight=w)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        acc = 0.0
        with torch.no_grad():
            for x, y in dl_te:
                acc += (model(x.to(device)).argmax(1) == y.to(device)).float().mean().item()
        acc /= len(dl_te)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "checkpoints/player_snn.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: acc={acc:.3f} best={best:.3f}")
    print(f"salvo | best_acc={best:.3f}")


if __name__ == "__main__":
    main()
