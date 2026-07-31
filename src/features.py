"""Batch MFCC feature extraction."""
from __future__ import annotations

import pandas as pd

from src.audio import mfcc_summary, wav_paths_for_patient
from src.paths import PROCESSED_DIR


def extract_mfcc_table(cohort: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Extract one row per recording for patients in split_name."""
    rows = []
    subset = cohort[cohort["split"] == split_name]
    for _, row in subset.iterrows():
        pid = int(row["Patient ID"])
        for wav_path in wav_paths_for_patient(pid, row["Recording locations:"]):
            feats = mfcc_summary(wav_path)
            rows.append({
                "Patient ID": pid,
                "wav_file": wav_path.name,
                "split": split_name,
                "y": row["y"],
                "Outcome": row["Outcome"],
                **{f"mfcc_{i}": v for i, v in enumerate(feats)},
            })
    return pd.DataFrame(rows)


def ensure_mfcc_features(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build or load train/val/test MFCC CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        path = PROCESSED_DIR / f"mfcc_features_{split_name}.csv"
        if path.exists():
            out[split_name] = pd.read_csv(path)
        else:
            out[split_name] = extract_mfcc_table(cohort, split_name)
            out[split_name].to_csv(path, index=False)
    return out
