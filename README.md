# Should AI-assisted Auscultation Diagnose or Defer in Congenital Heart Disease in Low-Resource Settings?
Work completed for my medical school's Arkell Prize in Congenital Heart Disease 

Uncertainty-aware **decision support** for paediatric heart-sound screening using the [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/) dataset (v1.0.3).

> **Disclaimer:** My work is for educational research only. This system does not diagnose or exclude congenital heart disease. Auto-Absent is not proof of a normal heart.

## Research question

Can a model that **defers uncertain cases** (to expert review) reduce **false reassurance** compared with a forced binary classifier that must always answer?

## How to start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place extracted CirCor data at:

`data/raw/the-circor-digiscope-phonocardiogram-dataset-1.0.3/`

Run the full pipeline:

```bash
.venv\Scripts\python.exe pipeline.py
```

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline.py` | Training and testing splits, MFCCs, train, deferral eval |
| `src/` | Modules: paths, splits, audio features, deferral metrics |
| `requirements.txt` | Pinned dependencies |
| `scripts/generate_figures.py` | To generate figures locally | 

## Method summary

- **Split:** patient-level (70/15/15), linked `Additional ID` pairs grouped
- **Label:** Murmur Present vs Absent (68 Unknown excluded)
- **Features:** 40 MFCC summaries per recording (4000 Hz)
- **Model:** Logistic regression (`class_weight=balanced`)
- **Deferral:** auto-Absent if prob < 0.35 and auto-Present if prob > 0.65 and if not then defer
- **Thresholds:** selected on validation (single run) 

## Key test result (murmur)

| Policy | AUROC | Missed murmurs (FN) | Deferred | Coverage |
|---|---:|---:|---:|---:|
| Forced (≥ 0.5) | 0.676 | 51 (all recordings) | 0 | 100% |
| Deferral (0.35/0.65) | 0.676 | 19 (among auto only) | 219 | 53% |

## Licence & data

Use CirCor under its PhysioNet terms. 
