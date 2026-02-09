# Self-Evolved Round0

This bundle contains **5 Python files**:

- `run_round0.py` — orchestrates Round-0 (PatchCore localize -> triage -> pseudo labels -> YOLO train -> optional critic)
- `se_patchcore.py` — PatchCore-style localizer (Good-only training)
- `se_triage_pseudo.py` — triage logic, pseudo-box extraction, YOLO dataset export (supports nested folders)
- `se_critic.py` — self-supervised critic (pos = pseudo boxes; neg = random boxes on Good)
- `se_yolo.py` — ultralytics YOLO train wrapper + data.yaml writer
