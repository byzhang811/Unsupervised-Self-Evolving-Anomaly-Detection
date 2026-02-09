from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import yaml


def write_data_yaml(yaml_path: Path, train_img_dir: Path, val_img_dir: Path, nc: int = 1, names: Optional[List[str]] = None) -> Path:
    if names is None:
        names = ["defect"]
    d = {
        "path": "",
        "train": str(train_img_dir),
        "val": str(val_img_dir),
        "nc": int(nc),
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(d, sort_keys=False))
    return yaml_path


def train_ultralytics(model_path_or_name: str,
                      data_yaml: Path,
                      imgsz: int,
                      epochs: int,
                      batch: int,
                      device: str,
                      project_dir: Path,
                      run_name: str):
    from ultralytics import YOLO

    try:
        model = YOLO(model_path_or_name)
    except Exception as e:
        print(f"[yolo] failed to load {model_path_or_name}: {e}")
        print("[yolo] fallback to yolov8n.pt")
        model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True
    )
    return results
