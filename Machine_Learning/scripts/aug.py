#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def detect_columns(df):
    ppm_cols = [c for c in df.columns if str(c).startswith("ppm_")]

    if not ppm_cols:
        raise ValueError("No spectral columns found. Expected columns starting with 'ppm_'.")

    non_ppm_cols = [c for c in df.columns if c not in ppm_cols]

    conc_cols = []
    metadata_cols = []

    for c in non_ppm_cols:
        converted = pd.to_numeric(df[c], errors="coerce")

        if converted.notna().any() and (converted.fillna(0.0) > 0).any():
            conc_cols.append(c)
            df[c] = converted.fillna(0.0)
        else:
            metadata_cols.append(c)

    if not conc_cols:
        raise ValueError("No valid numeric concentration columns found.")

    print("\nDetected concentration columns:")
    for c in conc_cols:
        print(f"  {c}")

    if metadata_cols:
        print("\nIgnored metadata columns:")
        for c in metadata_cols:
            print(f"  {c}")

    return conc_cols, ppm_cols


def ppm_values_from_headers(ppm_cols):
    ppm = []

    for c in ppm_cols:
        value = str(c).replace("ppm_", "").replace(",", ".")
        ppm.append(float(value))

    return np.array(ppm)


def normalize_pure_spectra(df, conc_cols, ppm_cols):
    spectra = []
    compound_names = []

    for i, row in df.iterrows():
        concentrations = pd.to_numeric(row[conc_cols], errors="coerce").fillna(0.0)

        nonzero = concentrations[concentrations > 0]

        if len(nonzero) != 1:
            raise ValueError(
                f"Row {i} should correspond to one pure compound only, "
                f"but {len(nonzero)} non-zero concentrations were found: "
                f"{list(nonzero.index)}"
            )

        compound = nonzero.index[0]
        concentration = float(nonzero.iloc[0])

        spectrum = pd.to_numeric(row[ppm_cols], errors="coerce").fillna(0.0).to_numpy()
        normalized_spectrum = spectrum / concentration

        spectra.append(normalized_spectrum)
        compound_names.append(compound)

    return compound_names, np.array(spectra)


def generate_concentrations(
    compound_names,
    n_compounds,
    min_conc,
    max_conc,
    rng,
    ref_concentrations=None,
    variation_fraction=1.0,
):
    """
    Generate concentrations for one mixture.

    If ref_concentrations is provided, each compound concentration is drawn
    uniformly in [max(0, ref * (1 - variation_fraction)),
                   ref * (1 + variation_fraction)].
    Otherwise, uniform in [min_conc, max_conc].
    """
    if ref_concentrations is not None:
        concentrations = np.zeros(n_compounds)
        for i, compound in enumerate(compound_names):
            if compound in ref_concentrations:
                ref = ref_concentrations[compound]
                low = max(min_conc, ref * (1.0 - variation_fraction))
                high = ref * (1.0 + variation_fraction)
                concentrations[i] = rng.uniform(low, high)
            else:
                concentrations[i] = rng.uniform(min_conc, max_conc)
        return concentrations
    else:
        return rng.uniform(min_conc, max_conc, size=n_compounds)


def add_realistic_noise(
    spectrum,
    ppm,
    rng,
    white_noise_fraction=0.003,
    correlated_noise_fraction=0.002,
    baseline_fraction=0.004,
    intensity_jitter_fraction=0.015,
):
    noisy = spectrum.copy()

    signal_scale = np.max(np.abs(noisy))
    if signal_scale == 0:
        signal_scale = 1.0

    noisy *= rng.normal(1.0, intensity_jitter_fraction)

    white_noise = rng.normal(
        loc=0.0,
        scale=white_noise_fraction * signal_scale,
        size=noisy.shape,
    )

    raw = rng.normal(0.0, 1.0, size=noisy.shape)
    window = max(5, len(noisy) // 200)
    kernel = np.ones(window) / window
    correlated = np.convolve(raw, kernel, mode="same")

    std_corr = np.std(correlated)
    if std_corr > 0:
        correlated *= correlated_noise_fraction * signal_scale / std_corr
    else:
        correlated *= 0.0

    x = np.linspace(-1.0, 1.0, len(noisy))

    baseline = (
        rng.normal(0.0, baseline_fraction * signal_scale) * x
        + rng.normal(0.0, baseline_fraction * signal_scale) * x**2
        + rng.normal(0.0, baseline_fraction * signal_scale)
        * np.sin(
            2 * np.pi * rng.uniform(0.5, 2.5) * x
            + rng.uniform(0.0, 2 * np.pi)
        )
    )

    noisy += white_noise
    noisy += correlated
    noisy += baseline

    return noisy


def generate_mixtures(
    compound_names,
    normalized_spectra,
    ppm,
    ppm_cols,
    n_samples,
    min_conc,
    max_conc,
    rng,
    ref_concentrations=None,
    variation_fraction=1.0,
):
    n_compounds = len(compound_names)
    rows = []

    for _ in range(n_samples):
        concentrations = generate_concentrations(
            compound_names=compound_names,
            n_compounds=n_compounds,
            min_conc=min_conc,
            max_conc=max_conc,
            rng=rng,
            ref_concentrations=ref_concentrations,
            variation_fraction=variation_fraction,
        )

        mixture = np.sum(
            normalized_spectra * concentrations[:, None],
            axis=0,
        )

        mixture = add_realistic_noise(
            spectrum=mixture,
            ppm=ppm,
            rng=rng,
        )

        row = {}

        for compound, conc in zip(compound_names, concentrations):
            row[compound] = conc

        for col, intensity in zip(ppm_cols, mixture):
            row[col] = intensity

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial mixed NMR spectra from pure-compound spectra."
    )

    parser.add_argument(
        "-i", "--input", required=True,
        help="Input CSV containing pure spectra.",
    )
    parser.add_argument(
        "-o", "--output", default="artificial_mixtures.csv",
        help="Output CSV file.",
    )
    parser.add_argument(
        "-n", "--n_samples", type=int, default=1000,
        help="Number of artificial spectra to generate.",
    )
    parser.add_argument(
        "--min_conc", type=float, default=0.01,
        help="Minimum concentration for each compound. Must be strictly positive.",
    )
    parser.add_argument(
        "--max_conc", type=float, default=1.0,
        help="Maximum concentration for each compound.",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed.",
    )
    parser.add_argument(
        "--ref-concentrations-csv", type=str, default=None,
        help=(
            "Optional CSV with reference concentrations (Table S1 M1 column). "
            "Expected columns: 'compound' and 'concentration'. "
            "If provided, concentrations are drawn around these reference values."
        ),
    )
    parser.add_argument(
        "--variation-fraction", type=float, default=1.0,
        help=(
            "Fraction of variation around reference concentrations. "
            "1.0 means +-100%% of the reference value (default). "
            "Only used when --ref-concentrations-csv is provided."
        ),
    )

    args = parser.parse_args()

    if args.min_conc <= 0:
        raise ValueError("--min_conc must be strictly positive.")

    if args.max_conc <= args.min_conc and args.ref_concentrations_csv is None:
        raise ValueError("--max_conc must be larger than --min_conc.")

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    conc_cols, ppm_cols = detect_columns(df)
    ppm = ppm_values_from_headers(ppm_cols)

    compound_names, normalized_spectra = normalize_pure_spectra(
        df=df,
        conc_cols=conc_cols,
        ppm_cols=ppm_cols,
    )

    # Load reference concentrations if provided
    ref_concentrations = None
    if args.ref_concentrations_csv is not None:
        ref_df = pd.read_csv(args.ref_concentrations_csv)
        if "compound" not in ref_df.columns or "concentration" not in ref_df.columns:
            raise ValueError(
                "ref-concentrations-csv must have 'compound' and 'concentration' columns."
            )
        ref_concentrations = dict(zip(ref_df["compound"], ref_df["concentration"]))
        print(f"\nReference concentrations loaded from: {args.ref_concentrations_csv}")
        for compound, conc in ref_concentrations.items():
            low = max(args.min_conc, conc * (1.0 - args.variation_fraction))
            high = conc * (1.0 + args.variation_fraction)
            print(f"  {compound}: [{low:.4f}, {high:.4f}] mM (ref={conc:.4f})")

    output_df = generate_mixtures(
        compound_names=compound_names,
        normalized_spectra=normalized_spectra,
        ppm=ppm,
        ppm_cols=ppm_cols,
        n_samples=args.n_samples,
        min_conc=args.min_conc,
        max_conc=args.max_conc,
        rng=rng,
        ref_concentrations=ref_concentrations,
        variation_fraction=args.variation_fraction,
    )

    output_df.to_csv(args.output, index=False)

    print(f"\nSaved {args.n_samples} artificial spectra to:")
    print(args.output)


if __name__ == "__main__":
    main()