"""Detector SNN de jogadores em stream de eventos.

Encoder conv-LIF (do zero) -> heatmap de centros + regressao de
tamanho. Sem anchors, sem NMS classico: pico local na heatmap = centro.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EventDetectorSNN(nn.Module):
    """(B,2,240,320) -> heatmap (B,1,15,20) + size (B,2,15,20).

    stride 16 total (4 convs stride 2)."""
    def __init__(self, num_steps=8):
        super().__init__()
        self.num_steps = num_steps
        self.c1 = nn.Conv2d(2, 32, 3, stride=2, padding=1)    # 120x160
        self.l1 = LIF()
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)   # 60x80
        self.l2 = LIF(v_th=0.8)
        self.c3 = nn.Conv2d(64, 96, 3, stride=2, padding=1)   # 30x40
        self.l3 = LIF(v_th=0.6)
        self.c4 = nn.Conv2d(96, 128, 3, stride=2, padding=1)  # 15x20
        self.l4 = LIF(v_th=0.5)
        self.hm_head = nn.Conv2d(128, 1, 1)
        self.sz_head = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        B = x.size(0)
        m1 = torch.zeros(B, 32, 120, 160, device=x.device)
        m2 = torch.zeros(B, 64, 60, 80, device=x.device)
        m3 = torch.zeros(B, 96, 30, 40, device=x.device)
        m4 = torch.zeros(B, 128, 15, 20, device=x.device)
        acc = 0.0
        for _ in range(self.num_steps):
            s, m1 = self.l1(self.c1(x), m1)
            s, m2 = self.l2(self.c2(s), m2)
            s, m3 = self.l3(self.c3(s), m3)
            s, m4 = self.l4(self.c4(s), m4)
            acc = acc + m4
        feat = acc / self.num_steps           # membrana media = ativacao
        return torch.sigmoid(self.hm_head(feat)), torch.sigmoid(self.sz_head(feat))


def decode(hm, size, k=8, thr=0.35):
    """Pico local na heatmap -> boxes em px (320x240 space)."""
    h2 = hm.reshape(1, 1, *hm.shape[-2:])
    hm_p = F.max_pool2d(h2, 3, stride=1, padding=1)[0, 0]
    flat_hm = hm.reshape(*hm.shape[-2:])
    flat_sz = size.reshape(2, *size.shape[-2:])
    peaks = (flat_hm == hm_p) & (flat_hm > thr)
    idx = peaks.nonzero()
    scores = flat_hm[peaks]
    top = scores.argsort(descending=True)[:k]
    boxes = []
    for t in top:
        gy, gx = idx[t].tolist()
        w = max(8, float(flat_sz[0, gy, gx]) * 320)
        h = max(12, float(flat_sz[1, gy, gx]) * 240)
        cx, cy = (gx + 0.5) * 16, (gy + 0.5) * 16
        boxes.append((cx - w / 2, cy - h / 2, w, h))
    return boxes
