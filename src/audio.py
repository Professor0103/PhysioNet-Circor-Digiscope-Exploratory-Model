"""Audio path resolution and MFCC feature extraction."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from src.paths import RECORDINGS_DIR


def wav_paths_for_patient(patient_id: int, location_string: str) -> list[Path]:
    """Return ordered .wav paths for one patient row (handles _1/_2 suffixes)."""
    sites = location_string.split("+")
    counts: dict[str, int] = {}
    paths: list[Path] = []
    for site in sites:
        counts[site] = counts.get(site, 0) + 1
        n = counts[site]
        plain = RECORDINGS_DIR / f"{patient_id}_{site}.wav"
        paths.append(plain if plain.exists() else RECORDINGS_DIR / f"{patient_id}_{site}_{n}.wav")
    return paths


def mfcc_summary(wav_path: Path, n_mfcc: int = 20, sr: int = 4000) -> np.ndarray:
    """20 MFCC means + 20 MFCC stds = 40 features per recording."""
    y, file_sr = librosa.load(wav_path, sr=sr)
    if file_sr != sr:
        raise ValueError(f"Expected {sr} Hz, got {file_sr} from {wav_path.name}")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])
