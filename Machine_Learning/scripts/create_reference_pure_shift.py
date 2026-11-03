#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

OUTPUT_CSV = Path("results/reference_pure_shift.csv")

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000
SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"

compounds_to_add = [
    {"sample_name": "Xi-Reference-Ala-10",  "compound_col": "Ala", "concentration": 40.63306768,  "spectrum_csv": Path("data/normalized_csv/references/Alanine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Arg-10",  "compound_col": "Arg", "concentration": 16.6967252,   "spectrum_csv": Path("data/normalized_csv/references/Arginine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Asn-10",  "compound_col": "Asn", "concentration": 6.97850439,   "spectrum_csv": Path("data/normalized_csv/references/Asparagine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Asp-10",  "compound_col": "Asp", "concentration": 5.052592035,  "spectrum_csv": Path("data/normalized_csv/references/Aspartate-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Glc-20", "compound_col": "Glc", "concentration": 49.398781,  "spectrum_csv": Path("data/normalized_csv/references/Glucose-Reference/20.csv")},
    {"sample_name": "Xi-Reference-Gln-10",  "compound_col": "Gln", "concentration": 16.95402299,  "spectrum_csv": Path("data/normalized_csv/references/Glutamine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Glu-10",  "compound_col": "Glu", "concentration": 10.95853161,  "spectrum_csv": Path("data/normalized_csv/references/Glutamate-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Ile-10",  "compound_col": "Ile", "concentration": 7.743902439,  "spectrum_csv": Path("data/normalized_csv/references/Isoleucine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Lac-10",  "compound_col": "Lac", "concentration": 97.68576358,  "spectrum_csv": Path("data/normalized_csv/references/Lactate-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Leu-10", "compound_col": "Leu", "concentration": 5.487804878,  "spectrum_csv": Path("data/normalized_csv/references/Leucine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Met-10",  "compound_col": "Met", "concentration": 5.19772118,   "spectrum_csv": Path("data/normalized_csv/references/Methionine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Phe-10",  "compound_col": "Phe", "concentration": 4.80025,      "spectrum_csv": Path("data/normalized_csv/references/Phenylalanine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Tyr-10",  "compound_col": "Tyr", "concentration": 2.174512942,  "spectrum_csv": Path("data/normalized_csv/references/Tyrosine-Reference/10.csv")},
    {"sample_name": "Xi-Reference-Val-10", "compound_col": "Val", "concentration": 5.021349274,  "spectrum_csv": Path("data/normalized_csv/references/Valine-Reference/10.csv")},
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