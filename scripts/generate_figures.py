
"""To generate figures into results/figures/. Run from project root:

    .venv\\Scripts\\python.exe scripts/generate_figures.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
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


C_MISS = "#e34948"   # automated false negative
C_DEFER = "#eda100"  # deferred to human review
C_CATCH = "#008300"  # automated true positive
C_INK = "#333333"
C_LINE = "#2a78d6"
SEQ_CMAP = "Blues" 

sns.set_theme(style="whitegrid", font_scale=1.05)


def save(name: str, pad: float = 0.08) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad)
    plt.close()
    print("  saved", path.name)
    return path


FLOW_FILL = "#e8f4fc"
FLOW_EXCL = "#f2f0ec"   
FLOW_STAGE = "#dcedf9"


def _flow_box(ax, x0, y0, x1, y1, heading, body, fill=FLOW_FILL) -> None:
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=fill, edgecolor=C_INK, linewidth=1.2, zorder=2,
    ))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if not body:
        ax.text(cx, cy, heading, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_INK, zorder=3)
        return

    h_head, gap, line_h = 3.3, 1.1, 2.9
    h_body = (body.count("\n") + 1) * line_h
    top = cy + (h_head + gap + h_body) / 2
    ax.text(cx, top - h_head / 2, heading, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_INK, zorder=3)
    ax.text(cx, top - h_head - gap - h_body / 2, body, ha="center", va="center",
            fontsize=8.5, color=C_INK, linespacing=1.45, zorder=3)


def _flow_arrow(ax, x0, y0, x1, y1, style="-|>") -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=1.5, color=C_INK,
                                shrinkA=0, shrinkB=0), zorder=1)


def fig01_pipeline_flowchart() -> None:
    """Study flow diagram."""
    raw = load_training_table()
    cohort = load_modelling_cohort(exclude_unknown_murmur=True)
    n_all, n_cohort = len(raw), cohort["Patient ID"].nunique()
    n_excl = n_all - n_cohort

    feat_train, feat_val, mfcc_cols = load_mfcc_features()
    feat_test = pd.read_csv(PROCESSED_DIR / "mfcc_features_test.csv")
    n_rec_all = 3163 
    splits = {
        "Training": (feat_train, "70%"),
        "Validation": (feat_val, "15%"),
        "Test": (feat_test, "15%"),
    }

    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    ax.set_xlim(-2, 102)      
    ax.set_ylim(-2, 102)
    ax.axis("off")
    ax.set_facecolor("white")

    mx0, mx1 = 22, 78
    cols = [(1, 32), (34, 66), (68, 99)]

    _flow_box(ax, mx0, 90, mx1, 99.5, "CirCor DigiScope Phonocardiogram v1.0.3",
              f"{n_all} patients · {n_rec_all} recordings")

    _flow_arrow(ax, 50, 90, 50, 82)
    _flow_arrow(ax, 50, 86.5, 79, 86.5)
    _flow_box(ax, 80, 82.5, 100, 90.5, "Excluded",
              f"Murmur “Unknown”\n({n_excl} patients)", fill=FLOW_EXCL)

    _flow_box(ax, mx0, 73.5, mx1, 82, "Modelling cohort",
              f"{n_cohort} patients")
    _flow_arrow(ax, 50, 73.5, 50, 66)

    _flow_box(ax, mx0 - 8, 54.5, mx1 + 8, 66, "Patient-level split (70 / 15 / 15)",
              "Repeat visits linked by “Additional ID” and kept together\n"
              "Stratified by clinical outcome · random seed 42")

    centres = [(x0 + x1) / 2 for x0, x1 in cols]
    ax.plot([50, 50], [54.5, 51.5], color=C_INK, lw=1.5, zorder=1)
    ax.plot([centres[0], centres[-1]], [51.5, 51.5], color=C_INK, lw=1.5, zorder=1)
    for (x0, x1), (name, (frame, pct)) in zip(cols, splits.items()):
        cx = (x0 + x1) / 2
        _flow_arrow(ax, cx, 51.5, cx, 48)
        _flow_box(ax, x0, 37.5, x1, 48, f"{name} ({pct})",
                  f"{frame['Patient ID'].nunique()} patients\n{len(frame)} recordings",
                  fill=FLOW_STAGE)
        _flow_arrow(ax, cx, 37.5, cx, 33)

    _flow_box(ax, cols[0][0], 23.5, cols[-1][1], 33, "Feature extraction",
              f"Each recording loaded at 4000 Hz · {len(mfcc_cols) // 2} MFCCs "
              f"summarised by temporal mean and SD\n"
              f"= {len(mfcc_cols)} features per recording")

    uses = [
        ("Fit model", "Logistic regression,\nbalanced class weights;\n"
                      "standardisation fitted\non training data only"),
        ("Select thresholds", "Deferral band\n0.35 / 0.65 chosen under\n"
                              "a ≥ 50% coverage floor;\ncalibration assessed"),
        ("Confirm, once", "Locked thresholds applied\nto held-out data;\n"
                          "forced 0.5 rule vs deferral"),
    ]
    for (x0, x1), (head, body) in zip(cols, uses):
        _flow_arrow(ax, (x0 + x1) / 2, 23.5, (x0 + x1) / 2, 17.5)
        _flow_box(ax, x0, 1, x1, 17.5, head, body, fill=FLOW_STAGE)

    for (xa, xb), label in (((cols[0][1], cols[1][0]), "model"),
                            ((cols[1][1], cols[2][0]), "thresholds")):
        _flow_arrow(ax, xa, 9.25, xb, 9.25)
        ax.text((xa + xb) / 2, 10.6, label, ha="center", va="bottom", rotation=90,
                fontsize=7.5, style="italic", color=C_INK, zorder=4)

    ax.set_title("Study pipeline", fontsize=12.5, fontweight="bold", pad=12)
    save("fig01_pipeline_flowchart.png")


def fig02_deferral_axis() -> None:
    """Probability axis with three decision zones."""
    margin = 0.05
    bar_y, bar_h = 0.48, 0.36
    axis_y = 0.34
    tick_bottom, tick_top = 0.30, 0.38
    lw = 1.6

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-0.08, 1)
    ax.axis("off")
    zones = [(0, T_LOW, "#cfe2f3", "Auto-Absent\n(prob < 0.35)"),
             (T_LOW, T_HIGH, "#fff2cc", "Defer\n(human review)"),
             (T_HIGH, 1, "#d9ead3", "Auto-Present\n(prob > 0.65)")]
    for x0, x1, color, label in zones:
        ax.add_patch(mpatches.Rectangle(
            (x0, bar_y), x1 - x0, bar_h, facecolor=color, edgecolor="none",
        ))
        ax.text((x0 + x1) / 2, bar_y + bar_h / 2, label, ha="center", va="center", fontsize=9)

    ax.add_patch(mpatches.Rectangle(
        (0, bar_y), 1, bar_h, fill=False, edgecolor="#333", linewidth=lw, clip_on=False,
    ))
    for x_div in (T_LOW, T_HIGH):
        ax.plot([x_div, x_div], [bar_y, bar_y + bar_h], color="#333", lw=lw, solid_capstyle="butt")

    ax.plot([0, 1], [axis_y, axis_y], color="#333", lw=lw, solid_capstyle="butt", clip_on=False)
    for tick in [0, T_LOW, T_HIGH, 1]:
        ax.plot([tick, tick], [tick_bottom, tick_top], color="#333", lw=lw, solid_capstyle="butt", clip_on=False)
        tick_lbl = f"{tick:.2f}" if tick not in (0, 1) else str(int(tick))
        ax.text(tick, 0.22, tick_lbl, ha="center", va="top", fontsize=9)
    ax.text(0.5, 0.04, "Predicted probability (Murmur Present)", ha="center", va="top", fontsize=10)
    ax.set_title(
        "Three-way deferral rule (thresholds frozen from validation)",
        fontsize=11, fontweight="bold", pad=14,
    )
    save("fig02_deferral_axis.png", pad=0.12)


@lru_cache(maxsize=1)
def _murmur_model_and_probs() -> dict:
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
    _, val_prob, y_val = _murmur_model_and_probs()["val"]
    rows = []
    for hw in np.linspace(0, 0.45, 25):
        m = deferral_metrics(val_prob, y_val, 0.5 - hw, 0.5 + hw)
        n_auto = len(y_val) - m["n_defer"]          # exact count, not coverage*n
        if n_auto == 0:
            continue
        rows.append({
            "coverage": n_auto / len(y_val),
            "error_rate": (m["fn_auto"] + m["fp_auto"]) / n_auto,
        })
    rc = pd.DataFrame(rows)

    chosen = deferral_metrics(val_prob, y_val, T_LOW, T_HIGH)
    n_auto_chosen = len(y_val) - chosen["n_defer"]
    chosen_cov = n_auto_chosen / len(y_val)
    chosen_err = (chosen["fn_auto"] + chosen["fp_auto"]) / n_auto_chosen

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(rc["coverage"], rc["error_rate"], "-", color=C_LINE, lw=2, zorder=2)
    ax.plot(rc["coverage"], rc["error_rate"], "o", color=C_LINE, ms=5,
            mec="white", mew=1.2, zorder=3)
    ax.plot([chosen_cov], [chosen_err], "o", ms=11, color=C_DEFER,
            mec=C_INK, mew=1.6, zorder=4)
    ax.annotate(
        f"Chosen band 0.35 / 0.65\ncoverage {chosen_cov:.3f}, error {chosen_err:.3f}",
        xy=(chosen_cov, chosen_err), xytext=(-22, -62), textcoords="offset points",
        fontsize=9, color=C_INK, ha="right",
        arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.2,
                        connectionstyle="arc3,rad=0.15"),
    )
    ax.axvline(0.5, color=C_INK, ls=":", lw=1.2, zorder=1)
    ax.text(0.5, ax.get_ylim()[0], " coverage floor (50%)", fontsize=8.5,
            color=C_INK, va="bottom", ha="left")
    ax.set(xlabel="Coverage (fraction decided automatically)",
           ylabel="Error rate among auto-decided",
           title="Risk–coverage trade-off (validation, murmur)")
    ax.invert_xaxis()
    save("fig03_risk_coverage_validation.png")


def fig04_reliability() -> None:
    """Reliability plot on validation."""
    _, val_prob, y_val = _murmur_model_and_probs()["val"]
    frac, mean_pred = calibration_curve(y_val, val_prob, n_bins=10, strategy="quantile")

    brier = brier_score_loss(y_val, val_prob)
    base_rate = float(y_val.mean())
    brier_ref = base_rate * (1 - base_rate)  # Brier of a constant base-rate predictor

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    for a in (ax, axh):
        a.axvspan(T_LOW, T_HIGH, color=C_DEFER, alpha=0.13, lw=0, zorder=0)

    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color=C_INK, label="Perfect calibration", zorder=2)
    ax.axhline(base_rate, ls=":", lw=1.3, color=C_INK, zorder=2)
    ax.text(0.015, base_rate + 0.015, f"Observed base rate {base_rate:.3f}",
            fontsize=8.5, color=C_INK, va="bottom")
    ax.plot(mean_pred, frac, "-", lw=2, color=C_LINE, zorder=3)
    ax.plot(mean_pred, frac, "o", ms=8, color=C_LINE, mec="white", mew=1.4,
            label="Model (validation)", zorder=4)

    ax.set(ylabel="Observed fraction Present", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(
        "Reliability of predicted murmur probability\n"
        f"Brier = {brier:.3f} vs {brier_ref:.3f} for a constant base-rate predictor",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.text(
        (T_LOW + T_HIGH) / 2, 0.035, "deferral band", ha="center", fontsize=8.5, color=C_INK,
    )

    axh.hist(val_prob[y_val == 0], bins=np.linspace(0, 1, 31), color=C_LINE,
             alpha=0.55, label="Absent")
    axh.hist(val_prob[y_val == 1], bins=np.linspace(0, 1, 31), color=C_MISS,
             alpha=0.7, label="Present")
    axh.set(xlabel="Predicted P(Murmur Present)", ylabel="Recordings", xlim=(0, 1))
    axh.legend(fontsize=8.5, frameon=True)

    save("fig04_reliability_validation.png")


def fig05_confusion_test() -> None:
    """Forced vs deferral confusion on test."""
    _, test_prob, y_test = _murmur_model_and_probs()["test"]
    forced_pred = (test_prob >= 0.5).astype(int)
    labels = deferral_labels(test_prob, T_LOW, T_HIGH)
    auto = labels != "defer"
    defer_pred = np.full(len(y_test), -1)
    defer_pred[auto] = (labels[auto] == "present").astype(int)

    cm_forced = confusion_matrix(y_test, forced_pred)
    cm_auto = confusion_matrix(y_test[auto], defer_pred[auto])

    vmax = max(cm_forced.max(), cm_auto.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cm, title in (
        (axes[0], cm_forced,
         f"Forced binary (threshold 0.5)\nall {len(y_test)} recordings — coverage 1.000"),
        (axes[1], cm_auto,
         f"Deferral, auto-decisions only\n{auto.sum()} of {len(y_test)} recordings — "
         f"coverage {auto.mean():.3f}"),
    ):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Absent", "Present"])
        disp.plot(ax=ax, cmap=SEQ_CMAP, colorbar=False, values_format="d")
        disp.im_.set_clim(0, vmax)
        ax.set_title(title, fontsize=10)
        ax.grid(False)

    for ax in axes:
        ax.add_patch(mpatches.Rectangle(
            (-0.5, 0.5), 1, 1, fill=False, edgecolor=C_MISS, linewidth=2.6, zorder=5,
        ))
    fig.suptitle(
        "Test-set confusion matrices (murmur); false negatives outlined",
        fontweight="bold", y=1.02,
    )
    save("fig05_confusion_test_forced_vs_defer.png")


def fig06_fn_reconciliation() -> None:
    _, test_prob, y_test = _murmur_model_and_probs()["test"]
    present = y_test == 1
    n_auto_fn = int((present & (test_prob < T_LOW)).sum())
    n_deferred_fn = int((present & (test_prob >= T_LOW) & (test_prob < 0.5)).sum())
    n_forced_tp_band = int((present & (test_prob >= 0.5) & (test_prob <= T_HIGH)).sum())
    n_auto_tp = int((present & (test_prob > T_HIGH)).sum())

    n_present = int(present.sum())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cats = [
        "Auto-Absent\n(missed)",
        "Deferred\n(forced rule missed)",
        "Deferred\n(forced rule caught)",
        "Auto-Present\n(caught)",
    ]
    vals = [n_auto_fn, n_deferred_fn, n_forced_tp_band, n_auto_tp]

    colors = [C_MISS, C_DEFER, C_DEFER, C_CATCH]
    hatches = ["", "", "///", ""]

    bars = ax.bar(cats, vals, color=colors, edgecolor=C_INK, linewidth=1.1, width=0.68)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v}  ({v / n_present:.0%})", ha="center", fontsize=10, color=C_INK)

    ax.set_ylabel(f"True Present recordings (n = {n_present})")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title(
        f"Disposition of true murmurs by decision zone (test set, n = {n_present})",
        fontweight="bold", fontsize=11,
    )
    legend = [
        mpatches.Patch(facecolor=C_MISS, edgecolor=C_INK, label="Automated miss"),
        mpatches.Patch(facecolor=C_DEFER, edgecolor=C_INK, label="Deferred to clinician"),
        mpatches.Patch(facecolor=C_CATCH, edgecolor=C_INK, label="Automated catch"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="upper left", frameon=True)
    save("fig06_fn_reconciliation_test.png")


def fig07_murmur_outcome_crosstab() -> None:
    """Patient-level crosstab from training table."""
    df = load_training_table()
    ct = pd.crosstab(df["Murmur"], df["Outcome"])
    fig, ax = plt.subplots(figsize=(5.2, 4))
   
    sns.heatmap(ct, annot=True, fmt="d", cmap=SEQ_CMAP, ax=ax,
                linewidths=2, linecolor="white", cbar_kws={"label": "Patients"})
    ax.set_title(
        f"Murmur vs Outcome (n = {len(df)} patients)", fontweight="bold", fontsize=11,
    )
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
    axes[1].set_title("Log-Mel spectrogram")
    fig.suptitle("Example heart-sound recording (4000 Hz)", fontweight="bold", y=1.01)
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
