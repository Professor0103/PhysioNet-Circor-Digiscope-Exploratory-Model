"""Load spreadsheets, splits, and saved feature tables."""
from __future__ import annotations

import pandas as pd

from src.paths import CSV_PATH, PROCESSED_DIR


def load_training_table() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def load_patient_splits() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "patient_splits.csv")


def load_modelling_cohort(exclude_unknown_murmur: bool = True) -> pd.DataFrame:
    """Patient-level table with split assignment and binary murmur label."""
    df = load_training_table()
    splits = load_patient_splits()
    if exclude_unknown_murmur:
        df = df[df["Murmur"] != "Unknown"]
    cohort = df.merge(splits, on="Patient ID").copy()
    cohort["y"] = (cohort["Murmur"] == "Present").astype(int)
    return cohort


def load_mfcc_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return train features, validation features, and MFCC column names."""
    feat_train = pd.read_csv(PROCESSED_DIR / "mfcc_features_train.csv")
    feat_val = pd.read_csv(PROCESSED_DIR / "mfcc_features_val.csv")
    mfcc_cols = [c for c in feat_train.columns if c.startswith("mfcc_")]
    return feat_train, feat_val, mfcc_cols
