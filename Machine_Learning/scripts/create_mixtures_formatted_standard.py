#!/usr/bin/env python3
"""
create_mixtures_formatted_standard.py

Lit les fichiers JCAMP-DX de mélanges, sélectionne les spectres Standard
(noesy), et génère results/mixtures_formatted_standard.csv au même format
que mixtures_formatted.csv (colonnes ppm_0000 à ppm_4999).

Usage:
    python create_mixtures_formatted_standard.py \
        --dx_dir data/normalized_csv/mixtures/spectra \
        --csv_dir data/normalized_csv/mixtures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

PPM_MIN = 0.0
PPM_MAX = 10.0
N_POINTS = 5000
SIGNAL_COLUMN = "intensity_divided_by_tsp_integral"
OUTPUT_CSV = Path("results/mixtures_formatted_standard.csv")


def detect_pulse_sequence(dx_path: Path) -> str:
    try:
        with open(dx_path, "r", errors="replace") as f:
            for line in f:
                if "PULSE SEQUENCE" in line.upper():
                    line_lower = line.lower()
                    if "psyche" in line_lower or "sapphire" in line_lower:
                        return "pure_shift"
                    elif "noesy" in line_lower:
                        return "standard"
    except Exception as e:
        print(f"  Erreur lecture {dx_path.name}: {e}")
    return "unknown"


def csv_to_formatted_row(csv_path: Path, sample_id: str, ppm_grid: np.ndarray) -> dict:
    df = pd.read_csv(csv_path)
    df = df.sort_values("ppm")

    ppm = df["ppm"].to_numpy(dtype=float)
    signal = df[SIGNAL_COLUMN].to_numpy(dtype=float)

    valid = np.isfinite(ppm) & np.isfinite(signal)
    ppm, signal = ppm[valid], signal[valid]
    ppm, unique_idx = np.unique(ppm, return_index=True)
    signal = signal[unique_idx]

    interpolator = interp1d(ppm, signal, kind="linear", bounds_error=False, fill_value=0.0)
    interpolated = interpolator(ppm_grid)

    row = {"sample_name": sample_id, "sample_kind": "mixture", "nmr_type": "standard"}
    for i, val in enumerate(interpolated):
        row[f"ppm_{i:04d}"] = val
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx_dir", required=True, help="Dossier contenant les fichiers .dx/.jdx")
    parser.add_argument("--csv_dir", required=True, help="Dossier contenant les CSV normalisés")
    parser.add_argument("--output_csv", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    dx_dir = Path(args.dx_dir)
    csv_dir = Path(args.csv_dir)
    output_csv_path = Path(args.output_csv)

    dx_files = sorted(dx_dir.glob("*.dx")) + sorted(dx_dir.glob("*.jdx"))
    if not dx_files:
        print(f"Aucun fichier .dx/.jdx trouvé dans {dx_dir}")
        return

    print(f"{len(dx_files)} fichiers DX trouvés.\n")

    ppm_grid = np.linspace(PPM_MIN, PPM_MAX, N_POINTS)
    formatted_rows = []
    std_count = ps_count = unknown_count = 0

    for dx_path in dx_files:
        sample_id = dx_path.stem
        nmr_type = detect_pulse_sequence(dx_path)

        if nmr_type == "standard":
            std_count += 1
        elif nmr_type == "pure_shift":
            ps_count += 1
            print(f"  {sample_id} → pure_shift (ignoré)")
            continue
        else:
            unknown_count += 1
            print(f"  {sample_id} → unknown (ignoré)")
            continue

        csv_path = csv_dir / f"{sample_id}.csv"
        if not csv_path.exists():
            print(f"  {sample_id} → CSV non trouvé : {csv_path} (ignoré)")
            continue

        formatted_row = csv_to_formatted_row(csv_path, sample_id, ppm_grid)
        formatted_rows.append(formatted_row)
        print(f"  {sample_id} → standard ✓")

    if not formatted_rows:
        print("\nAucun spectre Standard trouvé.")
        return

    df_out = pd.DataFrame(formatted_rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    print(f"\n── Résumé ──────────────────────────────")
    print(f"  Standard traités : {std_count}")
    print(f"  Pure Shift ignorés : {ps_count}")
    print(f"  Inconnus ignorés   : {unknown_count}")
    print(f"\nCSV sauvegardé : {output_csv_path} ({len(formatted_rows)} spectres)")


if __name__ == "__main__":
    main()