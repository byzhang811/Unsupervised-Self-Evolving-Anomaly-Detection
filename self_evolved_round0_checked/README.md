# Self-Evolved Round0 (checked)

This bundle contains **5 Python files**:

- `run_round0.py` — orchestrates Round-0 (PatchCore localize -> triage -> pseudo labels -> YOLO train -> optional critic)
- `se_patchcore.py` — PatchCore-style localizer (Good-only training)
- `se_triage_pseudo.py` — triage logic, pseudo-box extraction, YOLO dataset export (supports nested folders)
- `se_critic.py` — self-supervised critic (pos = pseudo boxes; neg = random boxes on Good)
- `se_yolo.py` — ultralytics YOLO train wrapper + data.yaml writer

## Expected dataset layout (your case)

You said you have something like:

- Images: `.../Detection_data/Part_01/Flaw/Easy/**.jpg` (often with subfolders like `object/`, `window_reflection/`, ...)
- Labels: `.../Detection_data/Part_01/Label/Easy/**.txt` (YOLO format, same subfolder structure as images)
- Good images: `.../Detection_data/Part_01/Good/**.jpg` (no labels needed)

### IMPORTANT: `--label_dir` must match `--flaw_dir`

If you set:
- `--flaw_dir = .../Flaw/Easy`
then set:
- `--label_dir = .../Label/Easy`

The code mirrors the relative subfolders to find labels.

## Run on SCC (example)

```bash
python run_round0.py   --good_dir  /projectnb/vipcnns/Boyang_Clutter/Unsupervised/Detection_data/Part_01/Good   --flaw_dir  /projectnb/vipcnns/Boyang_Clutter/Unsupervised/Detection_data/Part_01/Flaw/Easy   --label_dir /projectnb/vipcnns/Boyang_Clutter/Unsupervised/Detection_data/Part_01/Label/Easy   --out_dir   /projectnb/vipcnns/Boyang_Clutter/Unsupervised/runs/round0_easy   --enable_eval   --enable_critic
```

If you ONLY want to export the YOLO dataset + triage outputs (no YOLO training):

```bash
python run_round0.py ... --skip_yolo_train
```

## Notes / dependencies

- PatchCore & Critic use torchvision pretrained weights **if available**; if downloading fails on compute nodes, the code falls back to `weights=None` (it will run but quality may drop).
- You likely need: `torch`, `torchvision`, `opencv-python`, `ultralytics`, `tqdm`, `numpy`, `Pillow`.
