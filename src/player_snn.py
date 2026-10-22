"""SNN classificadora de blobs de eventos: JOGADOR vs RUIDO.

Entrada: crop de eventos 2x48x48 (ON/OFF) — mesma base LIF da serie.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["ruido", "jogador"]


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_mem, v_th=1.0):
        ctx.save_for_backward(v_mem)
        ctx.v_th = v_th
        return (v_mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v_mem,) = ctx.saved_tensors
        x = torch.pi * (v_mem - ctx.v_th)
        return grad_out / (1.0 + x * x), None


class LIF(nn.Module):
    def __init__(self, beta=0.85, v_th=1.0):
        super().__init__()
        self.beta, self.v_th = beta, v_th

    def forward(self, x, mem):
        mem = self.beta * mem + x
        s = SpikeFn.apply(mem, self.v_th)
        return s, mem * (1.0 - s)


class PlayerSNN(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)   # 24
        self.lif1 = LIF()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 12
        self.lif2 = LIF(v_th=0.8)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.lif3 = LIF(v_th=0.6)
        self.fc2 = nn.Linear(128, 2)
        self.lif4 = LIF(v_th=1e9)
        nn.init.normal_(self.fc1.weight, std=0.08)

    def forward(self, x):
        B = x.size(0)
        m1 = torch.zeros(B, 16, 24, 24, device=x.device)
        m2 = torch.zeros(B, 32, 12, 12, device=x.device)
        m3 = torch.zeros(B, 128, device=x.device)
        m4 = torch.zeros(B, 2, device=x.device)
        acc = 0.0
        for _ in range(self.num_steps):
            s, m1 = self.lif1(self.conv1(x), m1)
            s, m2 = self.lif2(self.conv2(s), m2)
            s, m3 = self.lif3(self.fc1(s.flatten(1)), m3)
            _, m4 = self.lif4(self.fc2(s), m4)
            acc = acc + m4
        return acc / self.num_steps


def crop_events(events, box, size=48, pad=4):
    H, W = events.shape
    x, y, w, h = box
    x0 = int(np.clip(x - pad, 0, W - 1))
    y0 = int(np.clip(y - pad, 0, H - 1))
    x1 = int(np.clip(x + w + pad, x0 + 1, W))
    y1 = int(np.clip(y + h + pad, y0 + 1, H))
    c = events[y0:y1, x0:x1]
    on = torch.from_numpy((c > 0).astype("float32"))
    off = torch.from_numpy((c < 0).astype("float32"))
    on = F.interpolate(on[None, None], (size, size), mode="bilinear")[0, 0]
    off = F.interpolate(off[None, None], (size, size), mode="bilinear")[0, 0]
    return torch.stack([on, off])
