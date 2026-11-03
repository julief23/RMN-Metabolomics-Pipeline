#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

OUTPUT_CSV = Path("results/mixture_formatted_pure_shift.csv")

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000
SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"

compounds_to_add = [
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//10.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//13.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//16.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//20.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//23.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//26.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//30.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//33.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//36.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//40.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//43.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//46.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//50.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//53.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//56.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//60.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//63.csv")},
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/mixtures//66.csv")},
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
