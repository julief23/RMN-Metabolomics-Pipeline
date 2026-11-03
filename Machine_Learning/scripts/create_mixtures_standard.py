#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

OUTPUT_CSV = Path("results/mixture_formatted_standard.csv")

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000
SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"

compounds_to_add = [
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//12.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//15.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//19.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//22.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//25.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//29.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//32.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//35.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//39.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//42.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//45.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//49.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//52.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//55.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//59.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//62.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//65.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures/9.csv")},
]

concentration_cols = ["Ala", "Arg", "Asn", "Asp", "Glc", "Gln", "Glu", "Ile", "Lac", "Leu", "Met", "Phe", "Tyr", "Val"]
ppm_grid = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)
ppm_cols = [f"ppm_{v:.4f}" for v in ppm_grid]

new_rows = []

for entry in compounds_to_add:
    spec_df = pd.read_csv(entry["spectrum_csv"])
    spec_df = spec_df.sort_values("ppm")

    ppm = spec_df["ppm"].to_numpy(dtype=float)
    signal = spec_df[SIGNAL_COLUMN].to_numpy(dtype=float)

    valid = np.isfinite(ppm) & np.isfinite(signal)
    ppm = ppm[valid]
    signal = signal[valid]

    ppm, unique_idx = np.unique(ppm, return_index=True)
    signal = signal[unique_idx]

    interpolator = interp1d(
        ppm, signal, kind="linear",
        bounds_error=False, fill_value=0.0,
    )
    interpolated = interpolator(ppm_grid)

    row = {
        "sample_name": entry["sample_name"],
        "sample_kind": "reference",
        "nmr_type": "pure_shift",
    }

    for col in concentration_cols:
        row[col] = 0.0
    row[entry["compound_col"]] = entry["concentration"]

    for col_name, val in zip(ppm_cols, interpolated):
        row[col_name] = val

    new_rows.append(row)
    print(f"Added: {entry['sample_name']}")

df = pd.DataFrame(new_rows)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFichier créé : {OUTPUT_CSV}")
print(f"Nombre de lignes : {len(df)}")
