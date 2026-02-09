from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image
import os
import shutil


IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images_flat(dir_path: Path) -> List[Path]:
    """Recursively list images under dir_path (kept name for backward-compat)."""
    files: List[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink, then symlink, then copy."""
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def clip_xyxy(x1,y1,x2,y2,w,h):
    x1 = max(0, min(int(x1), w-1))
    x2 = max(0, min(int(x2), w-1))
    y1 = max(0, min(int(y1), h-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    return float(inter) / float(area_a + area_b - inter + 1e-9)

def yolo_label_path(label_dir: Path, img_path: Path, img_root: Optional[Path] = None) -> Path:
    """
    Resolve the YOLO label path for an image.

    Preferred (nested folders):
      label_dir = .../Label/Easy
      img_root  = .../Flaw/Easy
      img_path  = .../Flaw/Easy/<subdirs>/name.jpg
      => label_dir/<subdirs>/name.txt

    Fallback (flat):
      label_dir/name.txt
    """
    # mirrored relative path (if possible)
    if img_root is not None:
        try:
            rel = img_path.relative_to(img_root).with_suffix(".txt")
            cand = label_dir / rel
            if cand.exists():
                return cand
        except Exception:
            pass

    # flat stem match
    cand2 = label_dir / (img_path.stem + ".txt")
    return cand2


def yolo_to_xyxy_px(xc,yc,w,h, img_w, img_h):
    x1 = int(round((xc - w/2) * img_w))
    x2 = int(round((xc + w/2) * img_w))
    y1 = int(round((yc - h/2) * img_h))
    y2 = int(round((yc + h/2) * img_h))
    return clip_xyxy(x1,y1,x2,y2,img_w,img_h)

def xyxy_px_to_yolo(x1,y1,x2,y2, img_w, img_h):
    xc = ((x1+x2)/2)/img_w
    yc = ((y1+y2)/2)/img_h
    w  = (x2-x1)/img_w
    h  = (y2-y1)/img_h
    return xc,yc,w,h

def write_yolo_txt(label_path: Path, boxes_xyxy: List[Tuple[int,int,int,int]], img_w: int, img_h: int, cls: int = 0):
    ensure_dir(label_path.parent)
    if not boxes_xyxy:
        label_path.write_text("")
        return
    out = []
    for (x1,y1,x2,y2) in boxes_xyxy:
        xc,yc,bw,bh = xyxy_px_to_yolo(x1,y1,x2,y2,img_w,img_h)
        # clamp
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        out.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(out) + "\n")

def triage_decision(score: float, tau_norm: float, tau_def: float) -> str:
    if score <= tau_norm:
        return "HC_normal"
    if score >= tau_def:
        return "HC_defect"
    return "unsure"

def map_stats(amap: np.ndarray, thr: float) -> Dict[str, float]:
    binm = (amap >= thr).astype(np.uint8)
    area_ratio = float(binm.mean())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    n_cc = max(0, n-1)
    largest = 0
    if n_cc > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max())
    return {"area_thr": area_ratio, "n_cc": float(n_cc), "largest_cc_area_px": float(largest)}

def bbox_from_amap(
    amap: np.ndarray,
    q: float,
    min_area_ratio: float,
    max_area_ratio: float,
    allow_multi: bool = False
) -> List[Tuple[int,int,int,int,float]]:
    H, W = amap.shape
    thr = float(np.quantile(amap.reshape(-1), q))
    binm = (amap >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    boxes = []
    for comp in range(1, n):
        x, y, w, h, area = stats[comp]
        area_ratio = area / float(H*W)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        x1,y1 = int(x), int(y)
        x2,y2 = int(x+w), int(y+h)
        x1,y1,x2,y2 = clip_xyxy(x1,y1,x2,y2,W,H)
        sc = float(amap[y1:y2, x1:x2].mean()) if (y2>y1 and x2>x1) else 0.0
        boxes.append((x1,y1,x2,y2,sc))
    if not boxes:
        return []
    boxes.sort(key=lambda t: t[4], reverse=True)
    return boxes if allow_multi else [boxes[0]]

def eval_pseudo_on_audit(audit_pairs: List[Tuple[Path, Path, Optional[Tuple[int,int,int,int]]]]) -> Dict[str, float]:
    """
    audit_pairs: list of (img_path, gt_label_path, pseudo_xyxy or None)
    """
    n_gt = 0
    n_pred = 0
    hits = 0
    ious = []
    for img_path, lab_path, pseudo in audit_pairs:
        rows = load_yolo_txt(lab_path)
        if len(rows) == 0:
            continue
        w,h = Image.open(img_path).size
        # pick largest gt box if multiple
        best = None
        best_area = -1
        for c,xc,yc,bw,bh in rows:
            x1,y1,x2,y2 = yolo_to_xyxy_px(xc,yc,bw,bh,w,h)
            area = max(0,x2-x1)*max(0,y2-y1)
            if area > best_area:
                best_area = area
                best = (x1,y1,x2,y2)
        if best is None:
            continue
        n_gt += 1
        if pseudo is None:
            continue
        n_pred += 1
        i = iou_xyxy(best, pseudo)
        ious.append(i)
        if i >= 0.5:
            hits += 1
    precision = hits / n_pred if n_pred > 0 else 0.0
    recall = hits / n_gt if n_gt > 0 else 0.0
    miou = float(np.mean(ious)) if ious else 0.0
    return {"audit_n": float(n_gt), "precision@0.5": float(precision), "recall@0.5": float(recall), "mean_iou": float(miou)}

def export_yolo_dataset(out_root: Path,
                        train_labeled: List[Path],
                        audit_post: List[Path],
                        label_dir: Path,
                        good_imgs: List[Path],
                        pseudo_boxes: List[Tuple[Path, Tuple[int,int,int,int]]],
                        img_root: Optional[Path] = None,
                        max_val_good: int = 200,
                        seed: int = 42) -> None:
    """
    Export a single-class YOLO dataset (nc=1, name='defect').

    Layout:
      out_root/
        images/{train,val}/...
        labels/{train,val}/...

    Train:
      - GT-labeled positives (from train_labeled)
      - pseudo positives (one box each)
      - Good negatives (empty label)

    Val:
      - audit_post positives (GT labels)
      - held-out Good negatives (empty label)

    Important:
      - If your dataset has nested folders, pass img_root as the image root (e.g., .../Flaw/Easy).
        Images & labels are exported *with the same relative path* to avoid name collisions.
      - label_dir should be the matching label root (e.g., .../Label/Easy), mirroring folders.
    """
    out_root = Path(out_root)
    img_train = out_root / "images" / "train"
    img_val   = out_root / "images" / "val"
    lab_train = out_root / "labels" / "train"
    lab_val   = out_root / "labels" / "val"

    # split Good into train/val so we don't duplicate identical images across splits
    rng = np.random.default_rng(seed)
    good_list = list(good_imgs)
    rng.shuffle(good_list)
    val_good = good_list[:max_val_good] if max_val_good > 0 else []
    train_good = good_list[len(val_good):]

    # compute a common Good root (best-effort) to preserve subfolders and avoid filename collisions
    good_root: Optional[Path] = None
    if good_list:
        try:
            import os
            common = os.path.commonpath([str(p) for p in good_list])
            good_root = Path(common).parent if Path(common).suffix else Path(common)
        except Exception:
            good_root = None

    def _dst_img(base: Path, p: Path) -> Path:
        root = None
        if img_root is not None:
            try:
                p.relative_to(img_root)
                root = img_root
            except Exception:
                root = None
        if root is None and good_root is not None:
            try:
                p.relative_to(good_root)
                root = good_root
            except Exception:
                root = None
        if root is not None:
            rel = p.relative_to(root)
            return base / rel
        return base / p.name

    def _dst_lab(base: Path, p: Path) -> Path:
        root = None
        if img_root is not None:
            try:
                p.relative_to(img_root)
                root = img_root
            except Exception:
                root = None
        if root is None and good_root is not None:
            try:
                p.relative_to(good_root)
                root = good_root
            except Exception:
                root = None
        if root is not None:
            rel = p.relative_to(root).with_suffix(".txt")
            return base / rel
        return base / (p.stem + ".txt")

    # helper to ensure parent dirs exist before linking/writing
    def _ensure(p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)

    # Train: GT positives
    for p in train_labeled:
        dst = _dst_img(img_train, p)
        _ensure(dst)
        link_or_copy(p, dst)

        lp = yolo_label_path(label_dir, p, img_root=img_root)
        dst_lp = _dst_lab(lab_train, p)
        _ensure(dst_lp)
        link_or_copy(lp, dst_lp)

    # Train: pseudo positives
    for p, box in pseudo_boxes:
        dst = _dst_img(img_train, p)
        _ensure(dst)
        link_or_copy(p, dst)

        w, h = Image.open(p).size
        dst_lp = _dst_lab(lab_train, p)
        _ensure(dst_lp)
        write_yolo_txt(dst_lp, w, h, [box])

    # Train: Good negatives
    for p in train_good:
        dst = _dst_img(img_train, p)
        _ensure(dst)
        link_or_copy(p, dst)

        dst_lp = _dst_lab(lab_train, p)
        _ensure(dst_lp)
        dst_lp.write_text("")

    # Val: audit_post GT positives
    for p in audit_post:
        dst = _dst_img(img_val, p)
        _ensure(dst)
        link_or_copy(p, dst)

        lp = yolo_label_path(label_dir, p, img_root=img_root)
        dst_lp = _dst_lab(lab_val, p)
        _ensure(dst_lp)
        link_or_copy(lp, dst_lp)

    # Val: held-out Good negatives
    for p in val_good:
        dst = _dst_img(img_val, p)
        _ensure(dst)
        link_or_copy(p, dst)

        dst_lp = _dst_lab(lab_val, p)
        _ensure(dst_lp)
        dst_lp.write_text("")

