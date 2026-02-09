from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


@dataclass
class CriticParams:
    img_size: int = 224
    device: str = "cuda"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    neg_boxes_per_image: int = 3
    tau_keep: float = 0.70


def _read_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def _clip_xyxy(x1,y1,x2,y2,w,h):
    x1 = max(0, min(int(x1), w-1))
    x2 = max(0, min(int(x2), w-1))
    y1 = max(0, min(int(y1), h-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def _crop_with_context(img: Image.Image, box: Tuple[int,int,int,int], ctx: float = 1.2) -> Image.Image:
    w,h = img.size
    x1,y1,x2,y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*ctx, (y2-y1)*ctx
    nx1 = int(round(cx - bw/2))
    nx2 = int(round(cx + bw/2))
    ny1 = int(round(cy - bh/2))
    ny2 = int(round(cy + bh/2))
    nx1,ny1,nx2,ny2 = _clip_xyxy(nx1,ny1,nx2,ny2,w,h)
    return img.crop((nx1,ny1,nx2,ny2))

def _random_box(w: int, h: int, min_ar=0.02, max_ar=0.25) -> Tuple[int,int,int,int]:
    area = random.uniform(min_ar, max_ar) * w * h
    r = random.uniform(0.5, 2.0)
    bw = int(round((area * r) ** 0.5))
    bh = int(round((area / r) ** 0.5))
    bw = max(2, min(bw, w-2))
    bh = max(2, min(bh, h-2))
    x1 = random.randint(0, w-bw)
    y1 = random.randint(0, h-bh)
    return (x1, y1, x1+bw, y1+bh)


class _CriticDataset(Dataset):
    def __init__(self,
                 pos_items: List[Tuple[Path, Tuple[int,int,int,int]]],
                 neg_imgs: List[Path],
                 img_size: int,
                 neg_per_img: int,
                 seed: int = 42):
        random.seed(seed)
        self.pos = pos_items
        self.neg_imgs = neg_imgs
        self.neg_per_img = neg_per_img
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        self.index = []
        for i in range(len(self.pos)):
            self.index.append((1, i, 0))
        for j in range(len(self.neg_imgs)):
            for k in range(neg_per_img):
                self.index.append((0, j, k))

    def __len__(self): 
        return len(self.index)

    def __getitem__(self, i):
        is_pos, idx, k = self.index[i]
        if is_pos == 1:
            img_path, box = self.pos[idx]
            img = _read_rgb(img_path)
            crop = _crop_with_context(img, box, ctx=1.2)
            x = self.tf(crop)
            y = torch.tensor([1.0], dtype=torch.float32)
            return x, y
        img_path = self.neg_imgs[idx]
        img = _read_rgb(img_path)
        w,h = img.size
        box = _random_box(w,h)
        crop = _crop_with_context(img, box, ctx=1.0)
        x = self.tf(crop)
        y = torch.tensor([0.0], dtype=torch.float32)
        return x, y


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[critic] warning: failed to load pretrained resnet18 weights: {e}")
            base = models.resnet18(weights=None)
        for p in base.parameters():
            p.requires_grad = False
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B,512,1,1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)


def train_critic(pos_items: List[Tuple[Path, Tuple[int,int,int,int]]],
                 neg_imgs: List[Path],
                 out_ckpt: Path,
                 p: CriticParams,
                 seed: int = 42) -> Path:
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(p.device if torch.cuda.is_available() or p.device == "cpu" else "cpu")

    ds = _CriticDataset(pos_items, neg_imgs, p.img_size, p.neg_boxes_per_image, seed=seed)
    dl = DataLoader(ds, batch_size=p.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = CriticNet().to(device)
    opt = optim.AdamW(model.head.parameters(), lr=p.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(1, p.epochs+1):
        losses = []
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logit = model(x)
            loss = loss_fn(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[critic] epoch {ep}/{p.epochs} loss={sum(losses)/len(losses):.4f}")

    torch.save({"state": model.state_dict(), "img_size": p.img_size}, out_ckpt)
    print(f"[critic] saved: {out_ckpt}")
    return out_ckpt


@torch.no_grad()
def batch_score(ckpt: Path,
                items: List[Tuple[Path, Tuple[int,int,int,int]]],
                device: str = "cuda",
                batch_size: int = 64) -> List[float]:
    ck = torch.load(ckpt, map_location="cpu")
    img_size = int(ck.get("img_size", 224))
    model = CriticNet()
    model.load_state_dict(ck["state"])
    model.eval()

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    probs: List[float] = []
    xs = []
    for img_path, box in items:
        img = _read_rgb(img_path)
        crop = _crop_with_context(img, box, ctx=1.2)
        xs.append(tf(crop))
        if len(xs) >= batch_size:
            x = torch.stack(xs, dim=0).to(dev)
            p = torch.sigmoid(model(x)).squeeze(1).detach().cpu().numpy().tolist()
            probs.extend([float(v) for v in p])
            xs = []
    if xs:
        x = torch.stack(xs, dim=0).to(dev)
        p = torch.sigmoid(model(x)).squeeze(1).detach().cpu().numpy().tolist()
        probs.extend([float(v) for v in p])
    return probs
