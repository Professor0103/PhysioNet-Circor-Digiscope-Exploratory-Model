"""
CirCor heart-sound screening pipeline — portfolio entry point.

Runs patient splits, MFCC extraction, murmur classification, and deferral evaluation.
Thresholds (0.35 / 0.65) were selected on validation; test evaluation is confirmatory.

Usage:
    python pipeline.py              # full run 
    python pipeline.py --eval-test  # test split only 

Data: PhysioNet CirCor v1.0.3 under data/raw/the-circor-digiscope-phonocardiogram-dataset-1.0.3/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project root on path when run from repo root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deferral import deferral_metrics, forced_fn_count
from src.features import ensure_mfcc_features
from src.load_data import load_modelling_cohort, load_training_table
from src.paths import CSV_PATH, PROCESSED_DIR, RANDOM_SEED, RECORDINGS_DIR
from src.splits import make_patient_splits

# Frozen on validation 
T_LOW, T_HIGH = 0.35, 0.65


def _check_data() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Download CirCor v1.0.3 into data/raw/.")
    if not RECORDINGS_DIR.exists():
        raise FileNotFoundError(f"Missing recordings dir {RECORDINGS_DIR}")


def _mfcc_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("mfcc_")]


def _train_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    model.fit(X_train, y_train)
    return model


def _eval_split(
    name: str,
    model: Pipeline,
    features: pd.DataFrame,
    mfcc_cols: list[str],
    label_col: str,
    present_name: str = "Present",
) -> dict:
    X = features[mfcc_cols]
    y = features[label_col].values
    prob = model.predict_proba(X)[:, 1]
    forced = (prob >= 0.5).astype(int)

    print(f"\n{'=' * 60}")
    print(f"{name} — forced binary (threshold 0.5)")
    print(classification_report(y, forced, target_names=[f"Not {present_name}", present_name]))
    print(f"AUROC: {round(roc_auc_score(y, prob), 3)}")

    defer = deferral_metrics(prob, y, T_LOW, T_HIGH)
    print(f"\n{name} — deferral ({T_LOW} / {T_HIGH})")
    print(f"  coverage: {defer['coverage']:.3f}  deferred: {defer['n_defer']}")
    print(f"  fn_auto: {defer['fn_auto']}  fp_auto: {defer['fp_auto']}")
    print(f"  sensitivity_auto: {defer['sensitivity_auto']:.3f}  specificity_auto: {defer['specificity_auto']:.3f}")
    print(f"  forced_fn (all recordings): {forced_fn_count(prob, y)}")

    return {
        "split": name,
        "auroc": round(float(roc_auc_score(y, prob)), 3),
        "forced_fn": forced_fn_count(prob, y),
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in defer.items()},
    }


def run_pipeline(eval_test: bool = False) -> None:
    _check_data()
    results_dir = PROJECT_ROOT / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Splits (patient-level, stratified by Outcome)
    splits_path = PROCESSED_DIR / "patient_splits.csv"
    if not splits_path.exists():
        print("Building patient splits...")
        make_patient_splits(load_training_table())
    else:
        print("Using existing patient splits:", splits_path)

    # Cohort: exclude Murmur Unknown (documented exclusion, n=68)
    cohort = load_modelling_cohort(exclude_unknown_murmur=True)
    print(f"Modelling cohort: {len(cohort)} patients (Unknown murmur excluded)")

    if eval_test:
        features = ensure_mfcc_features(cohort)
        mfcc_cols = _mfcc_columns(features["train"])
        model = _train_classifier(features["train"][mfcc_cols], features["train"]["y"])
        summary = _eval_split("TEST murmur", model, features["test"], mfcc_cols, "y", "Present")
        _save_results(results_dir, {"murmur_test": summary})
        return

    #MFCC features (one row per recording)
    print("Ensuring MFCC feature tables (train/val/test)...")
    features = ensure_mfcc_features(cohort)
    mfcc_cols = _mfcc_columns(features["train"])
    print(f"  train {features['train'].shape}  val {features['val'].shape}  test {features['test'].shape}")

    #Murmur model: train on train, tune thresholds already frozen
    print("\nTraining murmur classifier (logistic regression + MFCCs)...")
    murmur_model = _train_classifier(features["train"][mfcc_cols], features["train"]["y"])

    all_results: dict = {"thresholds": {"t_low": T_LOW, "t_high": T_HIGH}, "seed": RANDOM_SEED}
    all_results["murmur_val"] = _eval_split(
        "VALIDATION murmur", murmur_model, features["val"], mfcc_cols, "y", "Present",
    )
    all_results["murmur_test"] = _eval_split(
        "TEST murmur", murmur_model, features["test"], mfcc_cols, "y", "Present",
    )

    #Outcome extension (same features, different label)
    outcome_lookup = cohort[["Patient ID", "Outcome"]].drop_duplicates()
    for split in features:
        if "Outcome" not in features[split].columns:
            features[split] = features[split].merge(outcome_lookup, on="Patient ID", how="left")
        features[split]["y_outcome"] = (features[split]["Outcome"] == "Abnormal").astype(int)

    outcome_train = features["train"].copy()
    outcome_model = _train_classifier(outcome_train[mfcc_cols], outcome_train["y_outcome"])
    all_results["outcome_val"] = _eval_split(
        "VALIDATION outcome", outcome_model, features["val"], mfcc_cols, "y_outcome", "Abnormal",
    )
    all_results["outcome_test"] = _eval_split(
        "TEST outcome", outcome_model, features["test"], mfcc_cols, "y_outcome", "Abnormal",
    )

    _save_results(results_dir, all_results)
    print(f"\nResults saved to {results_dir / 'pipeline_results.json'}")


def _save_results(results_dir: Path, payload: dict) -> None:
    path = results_dir / "pipeline_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CirCor uncertainty-aware screening pipeline")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate test split only (murmur); assumes features and splits exist",
    )
    args = parser.parse_args()
    run_pipeline(eval_test=args.eval_test)


if __name__ == "__main__":
    main()
