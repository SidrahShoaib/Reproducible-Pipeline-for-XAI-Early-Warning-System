#python 10_create_temporal_splits.py --data-root "..\Data" --output-root "outputs"
import argparse
from pathlib import Path
import pandas as pd
from src.modelling_common import *

p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, default=Path("data")); p.add_argument("--output-root", type=Path, default=Path("outputs")); a = p.parse_args(); ensure_dirs(a.output_root)
folds = [
    {"fold": 1, "train": ["2024 S1"], "validate": "2024 S2"},
    {"fold": 2, "train": ["2024 S1", "2024 S2"], "validate": "2025 S1"},
]
for cp in CHECKPOINTS:
    train, val, test = load_all(a.data_root, cp)
    # After loading, drop rows with missing target
    train = train.dropna(subset=["at_risk"])
    val = val.dropna(subset=["at_risk"])
    test = test.dropna(subset=["at_risk"])

    base = a.output_root / "temporal_splits" / cp
    base.mkdir(parents=True, exist_ok=True)
    for f in folds:
        train[train.semester.isin(f["train"])].to_csv(base / f"fold_{f['fold']}_train.csv", index=False)
        train[train.semester == f["validate"]].to_csv(base / f"fold_{f['fold']}_validation.csv", index=False)
    train.to_csv(base / "development_all.csv", index=False)
    val.to_csv(base / "validation_2025_S2.csv", index=False)
    test.to_csv(base / "testing_2026_S1.csv", index=False)
    pd.DataFrame(folds).to_csv(base / "fold_definitions.csv", index=False)
print("Created semester-forward folds and final validation/test files.")
