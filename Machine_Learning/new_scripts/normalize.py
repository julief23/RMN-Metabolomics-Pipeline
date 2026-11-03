#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def detect_ppm_columns(df):
    ppm_cols = [c for c in df.columns if str(c).startswith("ppm_")]

    if not ppm_cols:
        raise ValueError("No ppm_ spectral columns were found.")

    return ppm_cols


def ppm_from_headers(ppm_cols, divide_by_1000=True):
    ppm = []

    for col in ppm_cols:
        value = str(col).replace("ppm_", "").replace(",", ".")
        value = float(value)

        if divide_by_1000:
            value /= 1000.0

        ppm.append(value)

    return np.array(ppm, dtype=float)


def compute_reference_integrals(
    X,
    ppm,
    ref_ppm=0.0,
    window=0.005,
    use_abs=False,
):
    mask = np.abs(ppm - ref_ppm) <= window

    if not np.any(mask):
        raise ValueError(
            f"No ppm points found in the reference window "
            f"{ref_ppm - window} to {ref_ppm + window} ppm."
        )

    ppm_ref = ppm[mask]
    X_ref = X[:, mask]

    order = np.argsort(ppm_ref)
    ppm_ref = ppm_ref[order]
    X_ref = X_ref[:, order]

    if use_abs:
        X_ref = np.abs(X_ref)

    if X_ref.shape[1] == 1:
        integrals = X_ref[:, 0]
    else:
        integrals = np.trapezoid(X_ref, x=ppm_ref, axis=1)

    return integrals, mask


def normalize_spectra(
    X,
    integrals,
    target_integral=1.0,
    min_abs_integral=1e-12,
):
    X_norm = X.copy()

    integrals = np.asarray(integrals, dtype=float)
    bad = (~np.isfinite(integrals)) | (np.abs(integrals) < min_abs_integral)

    scaling_factors = np.ones_like(integrals, dtype=float)

    good = ~bad
    scaling_factors[good] = target_integral / integrals[good]

    X_norm[good, :] = X_norm[good, :] * scaling_factors[good, None]

    return X_norm, scaling_factors, bad


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize real NMR spectra by the integral of the reference peak "
            "around 0 ppm, coherently with the normalized augmentation script."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input real spectra CSV.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="real_spectra_normalized_0ppm.csv",
        help="Output normalized CSV.",
    )

    parser.add_argument(
        "--summary-out",
        default="real_spectra_normalization_summary.csv",
        help="Output CSV containing reference integrals and scaling factors.",
    )

    parser.add_argument(
        "--ref-ppm",
        type=float,
        default=0.0,
        help="Reference peak position in ppm.",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=0.005,
        help="Half-width of integration window around ref-ppm, in ppm.",
    )

    parser.add_argument(
        "--target-integral",
        type=float,
        default=1.0,
        help="Target reference integral after normalization.",
    )

    parser.add_argument(
        "--no-divide-ppm-by-1000",
        action="store_true",
        help=(
            "Use this only if headers are already true ppm values. "
            "By default, ppm_0005 is interpreted as 0.005 ppm."
        ),
    )

    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="Use absolute intensities when computing the reference integral.",
    )

    parser.add_argument(
        "--min-abs-integral",
        type=float,
        default=1e-12,
        help="Minimum absolute integral allowed for normalization.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    ppm_cols = detect_ppm_columns(df)
    metadata_cols = [c for c in df.columns if c not in ppm_cols]

    ppm = ppm_from_headers(
        ppm_cols,
        divide_by_1000=not args.no_divide_ppm_by_1000,
    )

    X = (
        df[ppm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    print("\nInput file:")
    print(f"  {args.input}")

    print("\nDetected columns:")
    print(f"  metadata columns: {len(metadata_cols)}")
    print(f"  ppm columns:      {len(ppm_cols)}")
    print(f"  spectra:          {X.shape[0]}")

    print("\nPPM interpretation:")
    print(f"  ppm range: {ppm.min()} to {ppm.max()}")
    print(f"  ref ppm:   {args.ref_ppm}")
    print(f"  window:    ±{args.window} ppm")

    integrals, ref_mask = compute_reference_integrals(
        X=X,
        ppm=ppm,
        ref_ppm=args.ref_ppm,
        window=args.window,
        use_abs=args.use_abs,
    )

    print("\nReference region:")
    print(f"  points used: {ref_mask.sum()}")
    print(f"  from ppm:    {ppm[ref_mask].min()}")
    print(f"  to ppm:      {ppm[ref_mask].max()}")

    X_norm, scaling_factors, bad = normalize_spectra(
        X=X,
        integrals=integrals,
        target_integral=args.target_integral,
        min_abs_integral=args.min_abs_integral,
    )

    df_out = df.copy()
    df_out[ppm_cols] = X_norm
    df_out.to_csv(args.output, index=False)

    summary_df = pd.DataFrame(
        {
            "spectrum_index": np.arange(X.shape[0]),
            "reference_integral_before_normalization": integrals,
            "scaling_factor_applied": scaling_factors,
            "normalization_ok": ~bad,
        }
    )

    # Preserve useful metadata in the summary
    for col in metadata_cols:
        summary_df[col] = df[col].values

    summary_df.to_csv(args.summary_out, index=False)

    mean_integral = np.mean(integrals)
    std_integral = np.std(integrals)
    cv_integral = std_integral / mean_integral if mean_integral != 0 else np.nan

    print("\nReference integral before normalization:")
    print(f"  min    = {np.min(integrals):.10g}")
    print(f"  max    = {np.max(integrals):.10g}")
    print(f"  mean   = {mean_integral:.10g}")
    print(f"  median = {np.median(integrals):.10g}")
    print(f"  std    = {std_integral:.10g}")
    print(f"  CV     = {cv_integral:.6f}")

    print("\nNormalization:")
    print(f"  target reference integral = {args.target_integral}")
    print(f"  successfully normalized   = {(~bad).sum()} / {len(bad)}")
    print(f"  not normalized            = {bad.sum()} / {len(bad)}")

    print("\nSaved files:")
    print(f"  normalized spectra: {args.output}")
    print(f"  summary:            {args.summary_out}")


if __name__ == "__main__":
    main()
