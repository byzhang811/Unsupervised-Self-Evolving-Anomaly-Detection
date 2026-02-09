from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from se_patchcore import PatchCore, PatchCoreParams
from se_triage_pseudo import (
    list_images_flat, yolo_label_path, triage_decision,
    bbox_from_amap, map_stats, eval_pseudo_on_audit, export_yolo_dataset
)
from se_critic import CriticParams, train_critic, batch_score
from se_yolo import write_data_yaml, train_ultralytics


def split_audit(labeled_imgs: List[Path], seed: int, frac_total: float, frac_pre: float, frac_post: float):
    assert abs(frac_pre + frac_post - frac_total) < 1e-6, "pre+post must equal total"
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labeled_imgs))
    rng.shuffle(idx)
    n_total = int(round(len(idx) * frac_total))
    n_pre = int(round(len(idx) * frac_pre))
    n_post = n_total - n_pre
    audit_pre = [labeled_imgs[i] for i in idx[:n_pre]]
    audit_post = [labeled_imgs[i] for i in idx[n_pre:n_pre+n_post]]
    train_labeled = [p for p in labeled_imgs if (p not in set(audit_pre) and p not in set(audit_post))]
    return train_labeled, audit_pre, audit_post


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--good_dir", type=str, required=True)
    ap.add_argument("--flaw_dir", type=str, required=True)
    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)

    # Subsample switches for smoke tests
    ap.add_argument("--max_good", type=int, default=0, help="0 = use all")
    ap.add_argument("--max_flaw", type=int, default=0, help="0 = use all")

    # PatchCore
    ap.add_argument("--pc_img_size", type=int, default=224)
    ap.add_argument("--pc_backbone", type=str, default="wide_resnet50_2")
    ap.add_argument("--pc_device", type=str, default="cuda")
    ap.add_argument("--pc_patches_per_image", type=int, default=512)
    ap.add_argument("--pc_max_patches_total", type=int, default=200000)
    ap.add_argument("--pc_topk", type=int, default=200)

    # Triage / pseudo
    ap.add_argument("--tau_norm_q", type=float, default=0.995)     # from good score distribution
    ap.add_argument("--tau_def_q", type=float, default=0.90)       # from flaw score distribution
    ap.add_argument("--map_thr_q", type=float, default=0.98)       # per-image map threshold for cc/box
    ap.add_argument("--min_box_area_ratio", type=float, default=0.0002)
    ap.add_argument("--max_box_area_ratio", type=float, default=0.50)
    ap.add_argument("--allow_multi_box", action="store_true")

    # Audit
    ap.add_argument("--audit_total", type=float, default=0.10)
    ap.add_argument("--audit_pre", type=float, default=0.05)
    ap.add_argument("--audit_post", type=float, default=0.05)
    ap.add_argument("--enable_eval", action="store_true")

    # Critic
    ap.add_argument("--enable_critic", action="store_true")
    ap.add_argument("--critic_epochs", type=int, default=5)
    ap.add_argument("--critic_batch", type=int, default=64)
    ap.add_argument("--critic_lr", type=float, default=1e-3)
    ap.add_argument("--critic_tau_keep", type=float, default=0.70)
    ap.add_argument("--critic_pos_topq", type=float, default=0.97)  # positives from top quantile of pseudo candidates
    ap.add_argument("--critic_device", type=str, default="cuda")

    # YOLO
    ap.add_argument("--skip_yolo_train", action="store_true")
    ap.add_argument("--yolo_model", type=str, default="yolov10n.pt", help="path or name; recommend absolute path on SCC")
    ap.add_argument("--yolo_imgsz", type=int, default=640)
    ap.add_argument("--yolo_epochs", type=int, default=50)
    ap.add_argument("--yolo_batch", type=int, default=16)
    ap.add_argument("--yolo_device", type=str, default="0")

    args = ap.parse_args()

    good_dir = Path(args.good_dir)
    flaw_dir = Path(args.flaw_dir)
    label_dir = Path(args.label_dir)

    # quick sanity: label_dir should point to the matching label root for flaw_dir (e.g., Label/Easy)
    if not any(label_dir.rglob('*.txt')):
        print(f"[warn] label_dir has no .txt labels under it: {label_dir} (check your --label_dir)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    good_imgs = list_images_flat(good_dir)
    flaw_imgs = list_images_flat(flaw_dir)
    if args.max_good and args.max_good > 0:
        good_imgs = good_imgs[:args.max_good]
    if args.max_flaw and args.max_flaw > 0:
        flaw_imgs = flaw_imgs[:args.max_flaw]

    # Labeled subset inside flaw (audit source)
    labeled_flaw = [p for p in flaw_imgs if yolo_label_path(label_dir, p, img_root=flaw_dir).exists()]
    print(f"[data] good={len(good_imgs)} flaw={len(flaw_imgs)} labeled_in_flaw={len(labeled_flaw)}")

    # Audit splits
    train_labeled, audit_pre, audit_post = split_audit(
        labeled_flaw, seed=args.seed,
        frac_total=args.audit_total, frac_pre=args.audit_pre, frac_post=args.audit_post
    )
    print(f"[audit] train_labeled={len(train_labeled)} audit_pre={len(audit_pre)} audit_post={len(audit_post)}")

    # PatchCore init
    pc_params = PatchCoreParams(
        img_size=args.pc_img_size,
        backbone=args.pc_backbone,
        device=args.pc_device,
        patches_per_image=args.pc_patches_per_image,
        max_patches_total=args.pc_max_patches_total,
        topk=args.pc_topk
    )
    pc = PatchCore(pc_params)

    mem_path = out_dir / "patchcore_memory.npz"
    if mem_path.exists():
        pc.load_memory(mem_path)
        print(f"[patchcore] loaded memory: {mem_path}")
    else:
        print("[patchcore] building memory bank from Good...")
        pc.build_memory(good_imgs, seed=args.seed)
        pc.save_memory(mem_path)

    # Score distributions for thresholds
    good_scores = []
    for p in tqdm(good_imgs, desc="score good"):
        _, sd = pc.anomaly_map_and_scores(p)
        good_scores.append(sd["s_topk"])
    flaw_scores = []
    for p in tqdm(flaw_imgs, desc="score flaw"):
        _, sd = pc.anomaly_map_and_scores(p)
        flaw_scores.append(sd["s_topk"])

    tau_norm = float(np.quantile(np.array(good_scores), args.tau_norm_q))
    tau_def  = float(np.quantile(np.array(flaw_scores), args.tau_def_q))
    print(f"[triage] tau_norm={tau_norm:.6f} (good q={args.tau_norm_q}) | tau_def={tau_def:.6f} (flaw q={args.tau_def_q})")

    # Localize + triage + candidate boxes
    triage_rows = []
    pseudo_candidates: List[Tuple[Path, Tuple[int,int,int,int], float]] = []  # (img, box, s_topk)

    audit_pre_set = set(audit_pre)
    audit_post_set = set(audit_post)

    for p in tqdm(flaw_imgs, desc="localize+triage"):
        amap, sd = pc.anomaly_map_and_scores(p)
        thr_map = float(np.quantile(amap.reshape(-1), args.map_thr_q))
        st = map_stats(amap, thr_map)

        boxes = bbox_from_amap(
            amap,
            q=args.map_thr_q,
            min_area_ratio=args.min_box_area_ratio,
            max_area_ratio=args.max_box_area_ratio,
            allow_multi=args.allow_multi_box
        )

        # boxes in resized space -> original space
        w0,h0 = Image.open(p).size
        sx = w0 / float(args.pc_img_size)
        sy = h0 / float(args.pc_img_size)

        best_box = None
        best_sc = 0.0
        if boxes:
            x1,y1,x2,y2,sc = boxes[0]  # already sorted by sc
            ox1 = int(round(x1*sx)); ox2 = int(round(x2*sx))
            oy1 = int(round(y1*sy)); oy2 = int(round(y2*sy))
            best_box = (ox1,oy1,ox2,oy2)
            best_sc = float(sc)

        decision = triage_decision(sd["s_topk"], tau_norm=tau_norm, tau_def=tau_def)

        triage_rows.append({
            "img_path": str(p),
            "s_topk": sd["s_topk"],
            "s_max": sd["s_max"],
            "s_mean": sd["s_mean"],
            "thr_map": thr_map,
            "area_thr": st["area_thr"],
            "n_cc": st["n_cc"],
            "decision": decision,
            "has_box": int(best_box is not None),
            "box_xyxy": json.dumps(best_box) if best_box else "",
            "box_score": best_sc,
            "has_gt": int(yolo_label_path(label_dir, p, img_root=flaw_dir).exists()),
            "is_audit_pre": int(p in audit_pre_set),
            "is_audit_post": int(p in audit_post_set),
        })

        # candidates for pseudo: high-confidence defect + has box + NOT in audit_post (final holdout)
        if decision == "HC_defect" and best_box is not None and (p not in audit_post_set):
            pseudo_candidates.append((p, best_box, sd["s_topk"]))

    triage_csv = out_dir / "triage.csv"
    pd.DataFrame(triage_rows).to_csv(triage_csv, index=False)
    print(f"[out] triage csv: {triage_csv}")

    # audit_pre evaluation: pseudo vs gt (optional)
    if args.enable_eval:
        audit_pairs = []
        # find pseudo for audit_pre items
        row_by_path = {r["img_path"]: r for r in triage_rows}
        for p in audit_pre:
            gt = yolo_label_path(label_dir, p, img_root=flaw_dir)
            r = row_by_path.get(str(p), None)
            pb = None
            if r and r["box_xyxy"]:
                pb = tuple(json.loads(r["box_xyxy"]))
            audit_pairs.append((p, gt, pb))
        met = eval_pseudo_on_audit(audit_pairs)
        print("[audit_pre] pseudo vs gt:", met)
        (out_dir/"audit_pre_metrics.json").write_text(json.dumps(met, indent=2))

    # Critic filtering (optional)
    kept_pseudo: List[Tuple[Path, Tuple[int,int,int,int]]] = []
    critic_ckpt = out_dir / "critic.pt"

    if args.enable_critic:
        if len(pseudo_candidates) == 0:
            raise RuntimeError("No pseudo candidates. Try lower --tau_def_q or relax box filters.")
        # positives: top quantile by s_topk (cleaner)
        cand_scores = np.array([s for (_,_,s) in pseudo_candidates], dtype=np.float32)
        thr_pos = float(np.quantile(cand_scores, args.critic_pos_topq))
        pos_items = [(p, box) for (p, box, s) in pseudo_candidates if s >= thr_pos]
        print(f"[critic] pos_items={len(pos_items)} thr_pos={thr_pos:.6f}")

        cparams = CriticParams(
            img_size=args.pc_img_size,  # use same size for simplicity
            device=args.critic_device,
            epochs=args.critic_epochs,
            batch_size=args.critic_batch,
            lr=args.critic_lr,
            tau_keep=args.critic_tau_keep
        )
        train_critic(pos_items, good_imgs, critic_ckpt, cparams, seed=args.seed)

        # score all pseudo candidates in batches
        score_items = [(p, box) for (p, box, _) in pseudo_candidates]
        probs = batch_score(critic_ckpt, score_items, device=args.critic_device, batch_size=args.critic_batch)
        assert len(probs) == len(score_items)

        for (p, box, s_topk), prob in zip(pseudo_candidates, probs):
            if prob >= args.critic_tau_keep:
                kept_pseudo.append((p, box))
        print(f"[pseudo] kept after critic={len(kept_pseudo)} / candidates={len(pseudo_candidates)}")
    else:
        kept_pseudo = [(p, box) for (p, box, _) in pseudo_candidates]
        print(f"[pseudo] kept without critic={len(kept_pseudo)}")

    # Export YOLO dataset
    yolo_root = out_dir / "yolo_dataset"
    export_yolo_dataset(
        out_root=yolo_root,
        train_labeled=train_labeled,
        audit_post=audit_post,
        label_dir=label_dir,
        good_imgs=good_imgs,
        pseudo_boxes=kept_pseudo,
        img_root=flaw_dir,
        max_val_good=200,
        seed=args.seed
    )
    print(f"[yolo] dataset exported: {yolo_root}")

    data_yaml = write_data_yaml(out_dir/"data.yaml",
                                train_img_dir=yolo_root/"images"/"train",
                                val_img_dir=yolo_root/"images"/"val",
                                nc=1,
                                names=["defect"])
    print(f"[yolo] data.yaml: {data_yaml}")

    # Train YOLO (optional)
    if not args.skip_yolo_train:
        proj = out_dir / "yolo_runs"
        train_ultralytics(
            model_path_or_name=args.yolo_model,
            data_yaml=data_yaml,
            imgsz=args.yolo_imgsz,
            epochs=args.yolo_epochs,
            batch=args.yolo_batch,
            device=args.yolo_device,
            project_dir=proj,
            run_name="round0"
        )

    summary = {
        "good": len(good_imgs),
        "flaw": len(flaw_imgs),
        "labeled_in_flaw": len(labeled_flaw),
        "train_labeled": len(train_labeled),
        "audit_pre": len(audit_pre),
        "audit_post": len(audit_post),
        "tau_norm": tau_norm,
        "tau_def": tau_def,
        "pseudo_candidates": len(pseudo_candidates),
        "pseudo_kept": len(kept_pseudo),
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] summary:", summary)


if __name__ == "__main__":
    main()
