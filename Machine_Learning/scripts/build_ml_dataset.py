#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

INPUT_FILE = Path("results/assembled_spectra_with_concentrations.csv")
OUTPUT_DIR = Path("results")
OUTPUT_STANDARD = OUTPUT_DIR / "ml_dataset_standard.csv"
OUTPUT_PURE_SHIFT = OUTPUT_DIR / "ml_dataset_pure_shift.csv"

SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"
PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000


def build_dataset_for_group(df, ppm_grid):
    rows_standard = []
    rows_pure_shift = []

    metadata_cols = [
        "spectrum_file", "spectrum_name", "sample_name",
        "sample_kind", "nmr_type",
    ]
    signal_cols = [
        "ppm", "intensity", "tsp_integral",
        "intensity_divided_by_tsp_integral",
    ]
    concentration_cols = [
        col for col in df.columns
        if col not in metadata_cols + signal_cols
    ]

    for sample_name, group in df.groupby("sample_name"):
        group = group.sort_values("ppm")
        nmr_type = str(group["nmr_type"].iloc[0]).strip().lower()
        sample_kind = str(group["sample_kind"].iloc[0]).strip().lower()

        ppm = group["ppm"].to_numpy(dtype=float)
        signal = group[SIGNAL_COLUMN].to_numpy(dtype=float)

        valid = np.isfinite(ppm) & np.isfinite(signal)
        ppm = ppm[valid]
        signal = signal[valid]

        if len(ppm) < 2:
            continue

        order = np.argsort(ppm)
        ppm = ppm[order]
        signal = signal[order]

        ppm, unique_idx = np.unique(ppm, return_index=True)
        signal = signal[unique_idx]

        interpolator = interp1d(
            ppm, signal, kind="linear",
            bounds_error=False, fill_value=0.0,
        )
        interpolated_signal = interpolator(ppm_grid)

        row = {
            "sample_name": sample_name,
            "sample_kind": sample_kind,
            "nmr_type": nmr_type,
        }

        first_row = group.iloc[0]
        for compound in concentration_cols:
            row[compound] = first_row[compound]

        for i, value in enumerate(interpolated_signal):
            row[f"ppm_{i:04d}"] = value

        if nmr_type == "standard":
            rows_standard.append(row)
        elif nmr_type == "pure_shift":
            rows_pure_shift.append(row)

    return pd.DataFrame(rows_standard), pd.DataFrame(rows_pure_shift)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = {
        "sample_name", "sample_kind", "nmr_type",
        "ppm", SIGNAL_COLUMN,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ppm_grid = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)

    df_standard, df_pure_shift = build_dataset_for_group(df, ppm_grid)

    # Exclure les références Asn et Asp du dataset Standard uniquement
    asn_asp_cols = [c for c in ["Asn", "Asp"] if c in df_standard.columns]
    if asn_asp_cols:
        df_standard = df_standard[
            ~(
                (df_standard["sample_kind"] == "reference") &
                (df_standard[asn_asp_cols].gt(0).any(axis=1))
            )
        ]
        print("Asn/Asp references excluded from Standard dataset.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_standard.to_csv(OUTPUT_STANDARD, index=False)
    df_pure_shift.to_csv(OUTPUT_PURE_SHIFT, index=False)

    print(f"Standard dataset written: {OUTPUT_STANDARD}")
    print(f"Pure Shift dataset written: {OUTPUT_PURE_SHIFT}")
    print(f"Standard shape: {df_standard.shape}")
    print(f"Pure Shift shape: {df_pure_shift.shape}")


if __name__ == "__main__":
    main()