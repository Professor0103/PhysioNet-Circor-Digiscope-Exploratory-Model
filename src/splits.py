"""Patient-level grouping and train/val/test assignment."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import PROCESSED_DIR, RANDOM_SEED


def build_patient_groups(df: pd.DataFrame) -> tuple[pd.Series, dict[int, set[int]]]:
    """Return (patient_id -> group_id, link graph)."""
    graph: dict[int, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        pid = int(row["Patient ID"])
        if pd.notna(row["Additional ID"]):
            aid = int(row["Additional ID"])
            graph[pid].add(aid)
            graph[aid].add(pid)

    patient_to_group: dict[int, int] = {}
    group_id = 0
    for pid in df["Patient ID"]:
        if pid in patient_to_group:
            continue
        stack = [pid]
        while stack:
            node = stack.pop()
            if node in patient_to_group:
                continue
            patient_to_group[node] = group_id
            for neighbour in graph.get(node, set()):
                if neighbour not in patient_to_group:
                    stack.append(neighbour)
        group_id += 1

    return pd.Series(patient_to_group, name="group_id"), dict(graph)


def group_outcome_label(outcomes: pd.Series) -> str:
    return "Abnormal" if (outcomes == "Abnormal").any() else "Normal"


def make_patient_splits(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> pd.DataFrame:
    """Build stratified patient-level splits; save to processed dir."""
    groups, graph = build_patient_groups(df)
    df_groups = df.assign(group_id=df["Patient ID"].map(groups))

    strat_df = (
        df_groups.groupby("group_id")
        .agg(n_patients=("Patient ID", "nunique"), outcome=("Outcome", group_outcome_label))
        .reset_index()
    )

    group_ids = strat_df["group_id"].values
    labels = strat_df["outcome"].values
    test_frac = 1.0 - train_frac - val_frac

    train_groups, temp_groups, _, temp_y = train_test_split(
        group_ids, labels, test_size=(val_frac + test_frac), stratify=labels, random_state=RANDOM_SEED,
    )
    val_groups, test_groups = train_test_split(
        temp_groups, test_size=0.5, stratify=temp_y, random_state=RANDOM_SEED,
    )

    split_map = pd.concat([
        pd.Series("train", index=train_groups),
        pd.Series("val", index=val_groups),
        pd.Series("test", index=test_groups),
    ])

    patient_splits = (
        df[["Patient ID"]]
        .assign(group_id=lambda d: d["Patient ID"].map(groups))
        .assign(split=lambda d: d["group_id"].map(split_map))
    )

    assert patient_splits["split"].isna().sum() == 0
    assert patient_splits.groupby("group_id")["split"].nunique().max() == 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    patient_splits.to_csv(PROCESSED_DIR / "patient_splits.csv", index=False)
    return patient_splits
