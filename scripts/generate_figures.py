#!/usr/bin/env python3
"""Generate essay figures into results/figures/. Run from project root:

    .venv\\Scripts\\python.exe scripts/generate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.deferral import deferral_labels, deferral_metrics, forced_fn_count
from src.load_data import load_mfcc_features, load_modelling_cohort, load_training_table
from src.paths import PROCESSED_DIR, RANDOM_SEED, RECORDINGS_DIR
from src.audio import mfcc_summary, wav_paths_for_patient

FIG_DIR = PROJECT_ROOT / "results" / "figures"
T_LOW, T_HIGH = 0.35, 0.65

sns.set_theme(style="whitegrid", font_scale=1.05)


def save(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved", path.name)
    return path


def fig01_pipeline_flowchart() -> None:
    """Study pipeline — matches pipeline.py stages."""
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        "CirCor data",
        "Patient splits",
        "MFCC features",
        "Train LR",
        "Val: pick\n0.35/0.65",
        "Test:\nconfirm",
    ]
    xs = np.linspace(0.3, 9.7, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.25), 1.4, 0.5, boxstyle="round,pad=0.02",
            facecolor="#e8f4fc", edgecolor="#333", linewidth=1.2,
        ))
        ax.text(x + 0.7, 0.5, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.55, 0.5), xytext=(x + 1.45, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Fig 1 — Study pipeline", fontsize=12, fontweight="bold", pad=8)
    save("fig01_pipeline_flowchart.png")


def fig02_deferral_axis() -> None:
    """Probability axis with three decision zones."""
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    zones = [(0, T_LOW, "#cfe2f3", "Auto-Absent\n(prob < 0.35)"),
             (T_LOW, T_HIGH, "#fff2cc", "Defer\n(human review)"),
             (T_HIGH, 1, "#d9ead3", "Auto-Present\n(prob > 0.65)")]
    for x0, x1, color, label in zones:
        ax.add_patch(mpatches.Rectangle((x0, 0.35), x1 - x0, 0.35, facecolor=color, edgecolor="#333"))
        ax.text((x0 + x1) / 2, 0.52, label, ha="center", va="center", fontsize=10)
    ax.plot([0, 1], [0.22, 0.22], color="k", lw=1)
    for tick in [0, T_LOW, T_HIGH, 1]:
        ax.plot([tick, tick], [0.18, 0.26], color="k", lw=1)
        ax.text(tick, 0.08, f"{tick:.2f}" if tick not in (0, 1) else str(int(tick)), ha="center", fontsize=9)
    ax.text(0.5, 0.02, "Predicted probability (Murmur Present)", ha="center", fontsize=10)
    ax.set_title("Fig 2 — Three-way deferral rule (thresholds frozen from validation)", fontsize=11, fontweight="bold")
    save("fig02_deferral_axis.png")


def _murmur_model_and_probs():
    feat_train, feat_val, mfcc_cols = load_mfcc_features()
    feat_test = pd.read_csv(PROCESSED_DIR / "mfcc_features_test.csv")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    model.fit(feat_train[mfcc_cols], feat_train["y"])
    return {
        "val": (feat_val, model.predict_proba(feat_val[mfcc_cols])[:, 1], feat_val["y"].values),
        "test": (feat_test, model.predict_proba(feat_test[mfcc_cols])[:, 1], feat_test["y"].values),
    }


def fig03_risk_coverage() -> None:
    """Risk–coverage curve on validation."""
    feat_val = load_mfcc_features()[1]
    _, val_prob, y_val = _murmur_model_and_probs()["val"]
    rows = []
    for hw in np.linspace(0, 0.45, 25):
        m = deferral_metrics(val_prob, y_val, 0.5 - hw, 0.5 + hw)
        if m.get("coverage", 0) == 0:
            continue
        err = (m["fn_auto"] + m["fp_auto"]) / int(m["coverage"] * len(y_val))
        rows.append({"coverage": m["coverage"], "error_rate": err})
    rc = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc["coverage"], rc["error_rate"], "o-", color="#2563eb")
    ax.axvline(0.559, color="#dc2626", ls="--", lw=1, label="Chosen (0.35/0.65)")
    ax.set(xlabel="Coverage (fraction auto-decided)", ylabel="Error rate among auto-decided",
           title="Fig 3 — Risk–coverage curve (validation, murmur)")
    ax.invert_xaxis()
    ax.legend()
    save("fig03_risk_coverage_validation.png")


def fig04_reliability() -> None:
    """Calibration / reliability plot on validation."""
    _, val_prob, y_val = _murmur_model_and_probs()["val"]
    frac, mean_pred = calibration_curve(y_val, val_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac, "o-", label="Model")
    ax.set(xlabel="Mean predicted P(Present)", ylabel="Fraction actually Present",
           title="Fig 4 — Reliability plot (validation, murmur)")
    ax.legend()
    save("fig04_reliability_validation.png")


def fig05_confusion_test() -> None:
    """Forced vs deferral-style confusion on test (auto decisions only for defer panel)."""
    _, test_prob, y_test = _murmur_model_and_probs()["test"]
    forced_pred = (test_prob >= 0.5).astype(int)
    labels = deferral_labels(test_prob, T_LOW, T_HIGH)
    auto = labels != "defer"
    defer_pred = np.full(len(y_test), -1)
    defer_pred[auto] = (labels[auto] == "present").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, forced_pred), display_labels=["Absent", "Present"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Forced binary (threshold 0.5)\nall recordings")

    mask = auto
    ConfusionMatrixDisplay(
        confusion_matrix(y_test[mask], defer_pred[mask]), display_labels=["Absent", "Present"]
    ).plot(ax=axes[1], cmap="Greens", colorbar=False)
    axes[1].set_title(f"Deferral auto-decisions only\n(prob < {T_LOW} or > {T_HIGH})")

    fig.suptitle("Fig 5 — Test set confusion matrices (murmur)", fontweight="bold", y=1.02)
    save("fig05_confusion_test_forced_vs_defer.png")


def fig06_fn_reconciliation() -> None:
    """Decompose 51 forced FN on test."""
    _, test_prob, y_test = _murmur_model_and_probs()["test"]
    present = y_test == 1
    n_auto_fn = int((present & (test_prob < T_LOW)).sum())
    n_deferred_fn = int((present & (test_prob >= T_LOW) & (test_prob < 0.5)).sum())
    n_forced_tp_band = int((present & (test_prob >= 0.5) & (test_prob <= T_HIGH)).sum())
    n_auto_tp = int((present & (test_prob > T_HIGH)).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["Auto-Absent\n(FN)", "Deferred\n(was FN forced)", "Deferred\n(was TP forced)", "Auto-Present\n(TP)"]
    vals = [n_auto_fn, n_deferred_fn, n_forced_tp_band, n_auto_tp]
    colors = ["#dc2626", "#f59e0b", "#fbbf24", "#16a34a"]
    ax.bar(cats, vals, color=colors, edgecolor="#333")
    ax.set_ylabel("True Present recordings (n=112)")
    ax.set_title("Fig 6 — Where test murmurs went (FN reconciliation)", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center", fontsize=11)
    save("fig06_fn_reconciliation_test.png")


def fig07_murmur_outcome_crosstab() -> None:
    """Patient-level crosstab from training table."""
    df = load_training_table()
    ct = pd.crosstab(df["Murmur"], df["Outcome"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Patients"})
    ax.set_title("Fig 7 — Murmur vs Outcome (patients)", fontweight="bold")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Murmur")
    save("fig07_murmur_outcome_crosstab.png")


def fig08_waveform_spectrogram() -> None:
    """Example recording — patient 2530 AV."""
    import librosa.display

    wav = wav_paths_for_patient(2530, "AV+PV+TV+MV")[0]
    y, sr = librosa.load(wav, sr=4000)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set(xlabel="Time (s)", title=f"Waveform — {wav.name}")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2000)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], fmax=2000)
    axes[1].set_title("Log-mel spectrogram")
    fig.suptitle("Fig 8 — Example heart-sound recording (4000 Hz)", fontweight="bold", y=1.01)
    save("fig08_waveform_spectrogram_example.png")


def main() -> None:
    print("Generating figures ->", FIG_DIR)
    fig01_pipeline_flowchart()
    fig02_deferral_axis()
    fig03_risk_coverage()
    fig04_reliability()
    fig05_confusion_test()
    fig06_fn_reconciliation()
    fig07_murmur_outcome_crosstab()
    fig08_waveform_spectrogram()
    print("Done. See results/figures/CONTENTS.md for the index.")


if __name__ == "__main__":
    main()
