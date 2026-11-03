#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

REFERENCE1_CSV = Path("results/reference1.csv")
OUTPUT_CSV = Path("results/reference1_updated.csv")

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000
SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"

compounds_to_add = [
    {
        "sample_name": "Xi-Reference-Asn-9",
        "compound_col": "Asn",
        "concentration": 6.97850439,
        "spectrum_csv": Path("data/normalized_csv/references/Asparagine-Reference/11.csv"),
        "nmr_type": "standard",
    },
    {
        "sample_name": "Xi-Reference-Asp-9",
        "compound_col": "Asp",
        "concentration": 5.052592035,
        "spectrum_csv": Path("data/normalized_csv/references/Aspartate-Reference/11.csv"),
        "nmr_type": "standard",
    },
]

df_ref = pd.read_csv(REFERENCE1_CSV)
ppm_grid = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)
ppm_cols = [f"ppm_{i:04d}" for i in range(N_POINTS)]

concentration_cols = [
    c for c in df_ref.columns
    if c not in ["sample_name", "sample_kind", "nmr_type"] + ppm_cols
]

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
        "nmr_type": entry["nmr_type"],
    }

    for col in concentration_cols:
        row[col] = 0.0
    row[entry["compound_col"]] = entry["concentration"]

    for i, val in enumerate(interpolated):
        row[f"ppm_{i:04d}"] = val

    new_rows.append(row)
    print(f"Added: {entry['sample_name']}")

df_new = pd.DataFrame(new_rows)
df_updated = pd.concat([df_ref, df_new], ignore_index=True)
df_updated.to_csv(OUTPUT_CSV, index=False)
print(f"\nFichier mis à jour : {OUTPUT_CSV}")
print(f"Nombre de lignes : {len(df_updated)}")