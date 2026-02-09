from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass
class PatchCoreParams:
    img_size: int = 224
    backbone: str = "wide_resnet50_2"   # or "resnet50"
    device: str = "cuda"

    # Memory construction
    patches_per_image: int = 512
    max_patches_total: int = 200_000
    memory_dtype: str = "fp16"         # "fp16" (cuda only) or "fp32"

    # Nearest-neighbor search (chunked to cap peak memory)
    patch_chunk: int = 4096            # query patches per chunk (N is usually small, but keep for safety)
    mem_chunk: int = 8192              # memory patches per chunk (controls peak sim matrix size)
    topk: int = 200                    # pooling over anomaly map pixels



class _FeatureBackbone(torch.nn.Module):
    """
    Frozen ResNet backbone that returns two feature maps (layer2 and layer3),
    which are concatenated to form patch embeddings (simple PatchCore-style).
    """
    def __init__(self, name: str):
        super().__init__()
        if name == "wide_resnet50_2":
            try:
                m = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            except Exception as e:
                print(f"[patchcore] warning: failed to load pretrained weights for wide_resnet50_2: {e}")
                m = models.wide_resnet50_2(weights=None)
        elif name == "resnet50":
            try:
                m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except Exception as e:
                print(f"[patchcore] warning: failed to load pretrained weights for resnet50: {e}")
                m = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def _concat_feats(f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
    if f2.shape[-2:] != f3.shape[-2:]:
        f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([f2, f3], dim=1)
    return feat


class PatchCore:
    def __init__(self, params: PatchCoreParams):
        self.p = params
        self.device = torch.device(self.p.device if torch.cuda.is_available() or self.p.device == "cpu" else "cpu")
        self.bb = _FeatureBackbone(self.p.backbone).to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize((self.p.img_size, self.p.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.memory: Optional[np.ndarray] = None
        self._mem_torch: Optional[torch.Tensor] = None

        if self.device.type == 'cuda':
            # allow TF32 for faster matmul on Ampere+ (ok for this retrieval-style scoring)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _read_rgb(self, img_path: Path) -> Image.Image:
        return Image.open(img_path).convert("RGB")

    @torch.no_grad()
    def _extract_patches(self, img_t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
        f2, f3 = self.bb(img_t)
        feat = _concat_feats(f2, f3)          # [1,C,h,w]
        B,C,H,W = feat.shape
        patches = feat.permute(0,2,3,1).reshape(B*H*W, C)  # [N,C]
        patches = F.normalize(patches, p=2, dim=1)
        return patches, (H, W)

    def _get_mem_tensor(self) -> torch.Tensor:
        assert self.memory is not None and self.memory.size > 0, 'memory not built/loaded'
        if self._mem_torch is not None:
            return self._mem_torch
        mem = torch.from_numpy(self.memory)
        dtype = torch.float32
        if self.device.type == 'cuda' and self.p.memory_dtype.lower() == 'fp16':
            dtype = torch.float16
        mem = mem.to(self.device, dtype=dtype, non_blocking=True)
        self._mem_torch = mem
        return mem


    def build_memory(self, good_imgs: List[Path], seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mem_list = []
        for pth in good_imgs:
            img = self._read_rgb(pth)
            x = self.tf(img).unsqueeze(0).to(self.device)
            patches, _ = self._extract_patches(x)
            arr = patches.detach().cpu().numpy()
            if arr.shape[0] > self.p.patches_per_image:
                idx = rng.choice(arr.shape[0], size=self.p.patches_per_image, replace=False)
                arr = arr[idx]
            mem_list.append(arr)

        mem = np.concatenate(mem_list, axis=0) if mem_list else np.zeros((0, 1), dtype=np.float32)
        if mem.shape[0] > self.p.max_patches_total:
            idx = rng.choice(mem.shape[0], size=self.p.max_patches_total, replace=False)
            mem = mem[idx]
        self.memory = mem.astype(np.float32)
        self._mem_torch = None
        return self.memory

    def save_memory(self, npz_path: Path) -> None:
        assert self.memory is not None, "memory is None"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, memory=self.memory)

    def load_memory(self, npz_path: Path) -> np.ndarray:
        mem = np.load(npz_path)["memory"]
        self.memory = mem.astype(np.float32)
        self._mem_torch = None
        return self.memory

    @torch.no_grad()
    def anomaly_map_and_scores(self, img_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Returns:
          amap: float32 (img_size,img_size)
          scores: dict with s_max, s_topk, s_mean
        """
        assert self.memory is not None and self.memory.size > 0, "memory not built/loaded"

        img = self._read_rgb(img_path)
        x = self.tf(img).unsqueeze(0).to(self.device)
        patches, (h, w) = self._extract_patches(x)  # [N,C]

        mem = self._get_mem_tensor()                 # [M,C] on device (cached)
        patches = patches.to(dtype=mem.dtype)
        N = patches.shape[0]
        patch_chunk = int(self.p.patch_chunk)
        mem_chunk = int(self.p.mem_chunk)

        # cosine distance because normalized: dist = 1 - max_cos_sim
        dists = []
        for s in range(0, N, patch_chunk):
            pch = patches[s:s+patch_chunk]          # [c,C]
            best_sim = None
            for ms in range(0, mem.shape[0], mem_chunk):
                memc = mem[ms:ms+mem_chunk]         # [mc,C]
                sim = torch.matmul(pch, memc.t())   # [c,mc]
                nn_sim = sim.max(dim=1).values      # [c]
                best_sim = nn_sim if best_sim is None else torch.maximum(best_sim, nn_sim)
            nn_dist = (1.0 - best_sim).clamp(min=0)
            dists.append(nn_dist)
        dist = torch.cat(dists, dim=0)              # [N]

        dist_map = dist.reshape(h, w).unsqueeze(0).unsqueeze(0)     # [1,1,h,w]
        dist_up = F.interpolate(dist_map, size=(self.p.img_size, self.p.img_size),
                                mode="bilinear", align_corners=False)
        amap = dist_up[0,0].detach().cpu().numpy().astype(np.float32)

        flat = amap.reshape(-1)
        s_max = float(flat.max())
        k = min(self.p.topk, flat.shape[0])
        s_topk = float(np.partition(flat, -k)[-k:].mean())
        s_mean = float(flat.mean())
        return amap, {"s_max": s_max, "s_topk": s_topk, "s_mean": s_mean}