#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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

    print(f"\nDetected {len(ppm_cols)} ppm columns.")

    return conc_cols, ppm_cols


def ppm_values_from_headers(ppm_cols, divide_by_1000=False):
    ppm = []

    for c in ppm_cols:
        value = str(c).replace("ppm_", "").replace(",", ".")
        value = float(value)

        if divide_by_1000:
            value /= 1000.0

        ppm.append(value)

    return np.array(ppm, dtype=float)




def threshold_pure_spectrum(
    spectrum,
    threshold_fraction=0.001,
    threshold_mode="relative_to_max",
):
    """
    Keep only meaningful peaks in a pure/reference spectrum before mixing.

    This is intentionally applied before concentration scaling/mixing noise effects.
    It removes very small numerical/background features from the pure libraries but
    does not erase weak peaks from final noisy generated spectra.

    threshold_mode:
      - relative_to_max: threshold = fraction * max(abs(spectrum)) for each compound
      - absolute: threshold = fraction, interpreted directly in spectrum intensity units
    """
    s = np.asarray(spectrum, dtype=float).copy()

    if threshold_fraction is None or threshold_fraction <= 0:
        return s

    if threshold_mode == "relative_to_max":
        scale = np.max(np.abs(s))
        if scale == 0 or not np.isfinite(scale):
            return s
        threshold = threshold_fraction * scale
    elif threshold_mode == "absolute":
        threshold = threshold_fraction
    else:
        raise ValueError(
            "threshold_mode must be either 'relative_to_max' or 'absolute'."
        )

    s[np.abs(s) < threshold] = 0.0
    return s

def normalize_pure_spectra(
    df,
    conc_cols,
    ppm_cols,
    pure_peak_threshold_fraction=0.001,
    pure_peak_threshold_mode="relative_to_max",
):
    """
    Each row must correspond to one pure compound.

    The pure spectrum is divided by its known concentration, so that later:

        mixture = sum(concentration_i * normalized_spectrum_i)
    """

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

        spectrum = (
            pd.to_numeric(row[ppm_cols], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        normalized_spectrum = spectrum / concentration

        normalized_spectrum = threshold_pure_spectrum(
            normalized_spectrum,
            threshold_fraction=pure_peak_threshold_fraction,
            threshold_mode=pure_peak_threshold_mode,
        )

        spectra.append(normalized_spectrum)
        compound_names.append(compound)

    print("\nPure spectra normalized by their concentration:")
    print(
        f"Pure-spectrum pre-mixing threshold: "
        f"{pure_peak_threshold_fraction} ({pure_peak_threshold_mode})"
    )
    for c in compound_names:
        print(f"  {c}")

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
    Generate concentrations for one artificial mixture.

    If reference concentrations are provided, the concentration of compound i is:

        uniform[
            max(min_conc, ref_i * (1 - variation_fraction)),
            ref_i * (1 + variation_fraction)
        ]

    If no reference value exists for a compound, or no reference file is provided,
    the concentration is sampled from:

        uniform[min_conc, max_conc]
    """

    concentrations = np.zeros(n_compounds, dtype=float)

    if ref_concentrations is not None:
        for i, compound in enumerate(compound_names):
            if compound in ref_concentrations:
                ref = float(ref_concentrations[compound])

                low = max(min_conc, ref * (1.0 - variation_fraction))
                high = ref * (1.0 + variation_fraction)

                if high < low:
                    high = low

                concentrations[i] = rng.uniform(low, high)
            else:
                concentrations[i] = rng.uniform(min_conc, max_conc)

    else:
        concentrations = rng.uniform(min_conc, max_conc, size=n_compounds)

    return concentrations


def smooth_1d(y, sigma):
    if sigma <= 0:
        return y

    if SCIPY_AVAILABLE:
        return gaussian_filter1d(y, sigma=sigma, mode="nearest")

    window = max(3, int(round(6 * sigma)))
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def shift_spectrum(spectrum, ppm, shift_ppm):
    """
    Apply a small chemical shift displacement safely,
    whether ppm is increasing or decreasing.
    """

    ppm = np.asarray(ppm)
    spectrum = np.asarray(spectrum)

    order = np.argsort(ppm)
    ppm_sorted = ppm[order]
    spectrum_sorted = spectrum[order]

    shifted_sorted = np.interp(
        ppm_sorted,
        ppm_sorted - shift_ppm,
        spectrum_sorted,
        left=0.0,
        right=0.0,
    )

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    return shifted_sorted[inverse_order]


def distort_pure_spectra_before_mixing(
    normalized_spectra,
    ppm,
    rng,
    shift_sigma_ppm=0.0015,
    per_compound_gain_sigma=0.03,
    per_compound_linewidth_sigma_max=0.75,
):
    """
    Apply small compound-level distortions before mixing.
    Set the corresponding parameters to zero for almost ideal linear mixtures.
    """

    distorted = []

    for spectrum in normalized_spectra:
        s = spectrum.copy()

        if shift_sigma_ppm > 0:
            shift = rng.normal(0.0, shift_sigma_ppm)
            s = shift_spectrum(s, ppm, shift)

        if per_compound_linewidth_sigma_max > 0:
            sigma = rng.uniform(0.0, per_compound_linewidth_sigma_max)
            s = smooth_1d(s, sigma=sigma)

        if per_compound_gain_sigma > 0:
            gain = rng.lognormal(mean=0.0, sigma=per_compound_gain_sigma)
            s *= gain

        distorted.append(s)

    return np.array(distorted)


def add_realistic_noise(
    spectrum,
    ppm,
    rng,
    white_noise_fraction=0.0007,
    correlated_noise_fraction=0.0004,
    baseline_fraction=0.0007,
    absolute_white_noise_std=None,
    absolute_correlated_noise_std=None,
    absolute_baseline_std=None,
    intensity_jitter_fraction=0.005,
    global_scale_sigma=0.08,
    local_gain_sigma=0.02,
    local_gain_smooth=50.0,
    linewidth_sigma_max=0.5,
    mixture_shift_sigma_ppm=0.0,
    mixture_extra_broadening_sigma_max=0.0,
    threshold_fraction=0.0,
    clip_negative_intensities=False,
):
    """
    Add optional experimental-like perturbations to one artificial mixture.

    These effects are intentionally optional. To generate very clean spectra,
    set the corresponding values close to zero.
    """

    noisy = spectrum.copy()

    signal_scale = np.max(np.abs(noisy))
    if signal_scale == 0:
        signal_scale = 1.0

    if intensity_jitter_fraction > 0:
        noisy *= rng.normal(1.0, intensity_jitter_fraction)

    if global_scale_sigma > 0:
        noisy *= rng.lognormal(mean=0.0, sigma=global_scale_sigma)

    if linewidth_sigma_max > 0:
        sigma = rng.uniform(0.0, linewidth_sigma_max)
        noisy = smooth_1d(noisy, sigma=sigma)

    # Optional small global chemical-shift displacement of the whole mixture.
    # This mimics imperfect referencing/alignment in experimental spectra.
    if mixture_shift_sigma_ppm > 0:
        shift = rng.normal(0.0, mixture_shift_sigma_ppm)
        noisy = shift_spectrum(noisy, ppm, shift)

    # Optional extra post-mixing broadening applied to the whole spectrum.
    # This mimics differences in shimming, viscosity, acquisition, or processing.
    if mixture_extra_broadening_sigma_max > 0:
        sigma = rng.uniform(0.0, mixture_extra_broadening_sigma_max)
        noisy = smooth_1d(noisy, sigma=sigma)

    if local_gain_sigma > 0:
        local_gain = rng.normal(1.0, local_gain_sigma, size=noisy.shape)
        local_gain = smooth_1d(local_gain, sigma=local_gain_smooth)

        mean_gain = np.mean(local_gain)
        if mean_gain != 0:
            local_gain /= mean_gain

        noisy *= local_gain

    signal_scale = np.max(np.abs(noisy))
    if signal_scale == 0:
        signal_scale = 1.0

    # In quantitative NMR the signal scales with concentration, but the
    # instrumental/acquisition noise floor is approximately independent of
    # analyte concentration. Therefore, prefer absolute_*_std values.
    # If they are not provided, keep the older relative behavior for compatibility.
    white_std = (
        white_noise_fraction * signal_scale
        if absolute_white_noise_std is None
        else absolute_white_noise_std
    )

    white_noise = rng.normal(
        loc=0.0,
        scale=white_std,
        size=noisy.shape,
    )

    raw = rng.normal(0.0, 1.0, size=noisy.shape)
    window = max(5, len(noisy) // 200)
    kernel = np.ones(window) / window
    correlated = np.convolve(raw, kernel, mode="same")

    std_corr = np.std(correlated)
    correlated_std = (
        correlated_noise_fraction * signal_scale
        if absolute_correlated_noise_std is None
        else absolute_correlated_noise_std
    )

    if std_corr > 0:
        correlated *= correlated_std / std_corr
    else:
        correlated *= 0.0

    x = np.linspace(-1.0, 1.0, len(noisy))

    baseline_std = (
        baseline_fraction * signal_scale
        if absolute_baseline_std is None
        else absolute_baseline_std
    )

    baseline = (
        rng.normal(0.0, baseline_std) * x
        + rng.normal(0.0, baseline_std) * x**2
        + rng.normal(0.0, baseline_std)
        * np.sin(
            2 * np.pi * rng.uniform(0.5, 2.5) * x
            + rng.uniform(0.0, 2 * np.pi)
        )
    )

    noisy += white_noise
    noisy += correlated
    noisy += baseline

    if threshold_fraction is not None and threshold_fraction > 0:
        threshold = threshold_fraction * np.max(np.abs(noisy))
        noisy[np.abs(noisy) < threshold] = 0.0

    if clip_negative_intensities:
        noisy[noisy < 0.0] = 0.0

    return noisy


def normalize_by_reference_integral(
    spectrum,
    ppm,
    ref_ppm=0.0,
    window=0.005,
    target_integral=1.0,
    use_abs=False,
    min_abs_integral=1e-12,
):
    """
    Normalize one generated spectrum by the integral around the reference peak.

    The complete spectrum is multiplied by:

        target_integral / reference_integral

    If your ppm headers are true ppm values, e.g. ppm_0.000, ppm_0.001,
    use window=0.005 for ±0.005 ppm.

    If your ppm headers are encoded as ppm_0000, ppm_0001, ..., and you do NOT
    use --divide-ppm-by-1000, then use window=5.0 instead.
    """

    ppm = np.asarray(ppm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float).copy()

    mask = np.abs(ppm - ref_ppm) <= window

    if not np.any(mask):
        idx = np.argmin(np.abs(ppm - ref_ppm))
        integral = spectrum[idx]
    else:
        ppm_ref = ppm[mask]
        y_ref = spectrum[mask]

        order = np.argsort(ppm_ref)
        ppm_ref = ppm_ref[order]
        y_ref = y_ref[order]

        if use_abs:
            y_ref = np.abs(y_ref)

        if len(y_ref) == 1:
            integral = y_ref[0]
        else:
            integral = np.trapezoid(y_ref, x=ppm_ref)

    if not np.isfinite(integral) or abs(integral) < min_abs_integral:
        return spectrum, integral, False

    spectrum *= target_integral / integral

    return spectrum, integral, True


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
    shift_sigma_ppm=0.0015,
    per_compound_gain_sigma=0.03,
    per_compound_linewidth_sigma_max=0.75,
    white_noise_fraction=0.0007,
    correlated_noise_fraction=0.0004,
    baseline_fraction=0.0007,
    absolute_white_noise_std=None,
    absolute_correlated_noise_std=None,
    absolute_baseline_std=None,
    intensity_jitter_fraction=0.005,
    global_scale_sigma=0.08,
    local_gain_sigma=0.02,
    local_gain_smooth=50.0,
    linewidth_sigma_max=0.5,
    mixture_shift_sigma_ppm=0.0,
    mixture_extra_broadening_sigma_max=0.0,
    threshold_fraction=0.0,
    clip_negative_intensities=False,
    normalize_generated_by_reference=False,
    reference_ppm=0.0,
    reference_window=0.005,
    target_reference_integral=1.0,
    reference_use_abs=False,
):
    n_compounds = len(compound_names)
    rows = []

    n_normalized = 0
    n_not_normalized = 0

    #for sample_index in range(n_samples):
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

        distorted_spectra = distort_pure_spectra_before_mixing(
            normalized_spectra=normalized_spectra,
            ppm=ppm,
            rng=rng,
            shift_sigma_ppm=shift_sigma_ppm,
            per_compound_gain_sigma=per_compound_gain_sigma,
            per_compound_linewidth_sigma_max=per_compound_linewidth_sigma_max,
        )

        mixture = np.sum(
            distorted_spectra * concentrations[:, None],
            axis=0,
        )

        mixture = add_realistic_noise(
            spectrum=mixture,
            ppm=ppm,
            rng=rng,
            white_noise_fraction=white_noise_fraction,
            correlated_noise_fraction=correlated_noise_fraction,
            baseline_fraction=baseline_fraction,
            absolute_white_noise_std=absolute_white_noise_std,
            absolute_correlated_noise_std=absolute_correlated_noise_std,
            absolute_baseline_std=absolute_baseline_std,
            intensity_jitter_fraction=intensity_jitter_fraction,
            global_scale_sigma=global_scale_sigma,
            local_gain_sigma=local_gain_sigma,
            local_gain_smooth=local_gain_smooth,
            linewidth_sigma_max=linewidth_sigma_max,
            mixture_shift_sigma_ppm=mixture_shift_sigma_ppm,
            mixture_extra_broadening_sigma_max=mixture_extra_broadening_sigma_max,
            threshold_fraction=threshold_fraction,
            clip_negative_intensities=clip_negative_intensities,
        )

        ref_integral_before = np.nan
        ref_normalized_ok = False

        if normalize_generated_by_reference:
            mixture, ref_integral_before, ref_normalized_ok = normalize_by_reference_integral(
                spectrum=mixture,
                ppm=ppm,
                ref_ppm=reference_ppm,
                window=reference_window,
                target_integral=target_reference_integral,
                use_abs=reference_use_abs,
            )

            if ref_normalized_ok:
                n_normalized += 1
            else:
                n_not_normalized += 1

        row = {}

        for compound, conc in zip(compound_names, concentrations):
            row[compound] = conc

        for col, intensity in zip(ppm_cols, mixture):
            row[col] = intensity

        rows.append(row)

    if normalize_generated_by_reference:
        print("\nGenerated-spectrum normalization summary:")
        print(f"  normalized successfully: {n_normalized}")
        print(f"  not normalized:          {n_not_normalized}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate artificial mixed NMR spectra from pure-compound spectra, "
            "with optional final normalization by the 0 ppm reference integral."
        )
    )

    parser.add_argument("-i", "--input", required=True, help="Input CSV containing pure spectra.")
    parser.add_argument("-o", "--output", default="artificial_mixtures.csv", help="Output CSV file.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="Number of artificial spectra to generate.")

    parser.add_argument("--min_conc", type=float, default=0.01)
    parser.add_argument("--max_conc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--ref-concentrations-csv", type=str, default=None)
    parser.add_argument("--variation-fraction", type=float, default=1.0)

    parser.add_argument(
        "--divide-ppm-by-1000",
        action="store_true",
        help="Use this if ppm_0005 means 0.005 ppm. Recommended for ppm_0000-style headers.",
    )

    parser.add_argument(
        "--pure-peak-threshold-fraction",
        type=float,
        default=0.001,
        help=(
            "Threshold applied to each normalized pure spectrum BEFORE mixing. "
            "Use 0.0 to disable. Default: 0.001 = keep peaks above 0.1% of each pure spectrum max."
        ),
    )
    parser.add_argument(
        "--pure-peak-threshold-mode",
        choices=["relative_to_max", "absolute"],
        default="relative_to_max",
        help="How to interpret --pure-peak-threshold-fraction.",
    )

    parser.add_argument("--shift-sigma-ppm", type=float, default=0.0015)
    parser.add_argument("--per-compound-gain-sigma", type=float, default=0.03)
    parser.add_argument("--per-compound-linewidth-sigma-max", type=float, default=0.75)

    parser.add_argument("--white-noise-fraction", type=float, default=0.0007)
    parser.add_argument("--correlated-noise-fraction", type=float, default=0.0004)
    parser.add_argument("--baseline-fraction", type=float, default=0.0007)
    parser.add_argument(
        "--absolute-white-noise-std",
        type=float,
        default=None,
        help=(
            "Absolute standard deviation of white noise in spectral intensity units. "
            "Recommended for quantitative NMR because S/N then scales with concentration."
        ),
    )
    parser.add_argument(
        "--absolute-correlated-noise-std",
        type=float,
        default=None,
        help="Absolute standard deviation of smooth/correlated noise.",
    )
    parser.add_argument(
        "--absolute-baseline-std",
        type=float,
        default=None,
        help="Absolute amplitude scale for baseline distortions.",
    )
    parser.add_argument("--intensity-jitter-fraction", type=float, default=0.005)
    parser.add_argument("--global-scale-sigma", type=float, default=0.08)
    parser.add_argument("--local-gain-sigma", type=float, default=0.02)
    parser.add_argument("--local-gain-smooth", type=float, default=50.0)
    parser.add_argument("--linewidth-sigma-max", type=float, default=0.5)
    parser.add_argument(
        "--mixture-shift-sigma-ppm",
        type=float,
        default=0.0,
        help="Optional global chemical-shift sigma in ppm applied to each generated mixture after mixing.",
    )
    parser.add_argument(
        "--mixture-extra-broadening-sigma-max",
        type=float,
        default=0.0,
        help="Optional extra post-mixing Gaussian broadening sigma max in data points.",
    )
    parser.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.0,
        help=(
            "Final post-noise threshold. For quantitative NMR augmentation this should normally stay 0.0; "
            "use --pure-peak-threshold-fraction instead."
        ),
    )

    parser.add_argument("--clip-negative-intensities", action="store_true")

    parser.add_argument(
        "--normalize-generated-by-reference",
        action="store_true",
        help="Normalize each generated spectrum by the reference integral.",
    )

    parser.add_argument("--reference-ppm", type=float, default=0.0)
    parser.add_argument(
        "--reference-window",
        type=float,
        default=0.005,
        help="Half-width of reference integration window in ppm. Use 0.005 with --divide-ppm-by-1000.",
    )
    parser.add_argument("--target-reference-integral", type=float, default=1.0)
    parser.add_argument("--reference-use-abs", action="store_true")

    args = parser.parse_args()

    if args.min_conc <= 0:
        raise ValueError("--min_conc must be strictly positive.")

    if args.max_conc <= args.min_conc and args.ref_concentrations_csv is None:
        raise ValueError("--max_conc must be larger than --min_conc.")

    if args.variation_fraction < 0:
        raise ValueError("--variation-fraction must be >= 0.")

    if args.n_samples <= 0:
        raise ValueError("--n_samples must be positive.")

    if not SCIPY_AVAILABLE:
        print("\nWARNING: scipy is not available. Using moving-average smoothing fallback.")

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    conc_cols, ppm_cols = detect_columns(df)

    ppm = ppm_values_from_headers(
        ppm_cols,
        divide_by_1000=args.divide_ppm_by_1000,
    )

    print(f"\nppm range interpreted as: {ppm.min()} to {ppm.max()}")

    compound_names, normalized_spectra = normalize_pure_spectra(
        df=df,
        conc_cols=conc_cols,
        ppm_cols=ppm_cols,
        pure_peak_threshold_fraction=args.pure_peak_threshold_fraction,
        pure_peak_threshold_mode=args.pure_peak_threshold_mode,
    )

    ref_concentrations = None

    if args.ref_concentrations_csv is not None:
        ref_df = pd.read_csv(args.ref_concentrations_csv)

        if "compound" not in ref_df.columns or "concentration" not in ref_df.columns:
            raise ValueError(
                "ref-concentrations-csv must contain columns: compound, concentration"
            )

        ref_concentrations = dict(zip(ref_df["compound"], ref_df["concentration"]))

        print(f"\nReference concentrations loaded from: {args.ref_concentrations_csv}")

        for compound, conc in ref_concentrations.items():
            conc = float(conc)
            low = max(args.min_conc, conc * (1.0 - args.variation_fraction))
            high = conc * (1.0 + args.variation_fraction)

            if high < low:
                high = low

            print(f"  {compound}: [{low:.6g}, {high:.6g}] mM (ref={conc:.6g})")

    print("\nSpectral realism parameters:")
    print(f"  pure_peak_threshold_fraction = {args.pure_peak_threshold_fraction}")
    print(f"  pure_peak_threshold_mode = {args.pure_peak_threshold_mode}")
    print(f"  shift_sigma_ppm = {args.shift_sigma_ppm}")
    print(f"  per_compound_gain_sigma = {args.per_compound_gain_sigma}")
    print(f"  per_compound_linewidth_sigma_max = {args.per_compound_linewidth_sigma_max}")
    print(f"  white_noise_fraction = {args.white_noise_fraction}")
    print(f"  correlated_noise_fraction = {args.correlated_noise_fraction}")
    print(f"  baseline_fraction = {args.baseline_fraction}")
    print(f"  absolute_white_noise_std = {args.absolute_white_noise_std}")
    print(f"  absolute_correlated_noise_std = {args.absolute_correlated_noise_std}")
    print(f"  absolute_baseline_std = {args.absolute_baseline_std}")
    print(f"  intensity_jitter_fraction = {args.intensity_jitter_fraction}")
    print(f"  global_scale_sigma = {args.global_scale_sigma}")
    print(f"  local_gain_sigma = {args.local_gain_sigma}")
    print(f"  linewidth_sigma_max = {args.linewidth_sigma_max}")
    print(f"  mixture_shift_sigma_ppm = {args.mixture_shift_sigma_ppm}")
    print(f"  mixture_extra_broadening_sigma_max = {args.mixture_extra_broadening_sigma_max}")
    print(f"  threshold_fraction = {args.threshold_fraction}")
    print(f"  clip_negative_intensities = {args.clip_negative_intensities}")

    print("\nGenerated-spectrum reference normalization:")
    print(f"  normalize_generated_by_reference = {args.normalize_generated_by_reference}")
    print(f"  reference_ppm = {args.reference_ppm}")
    print(f"  reference_window = {args.reference_window}")
    print(f"  target_reference_integral = {args.target_reference_integral}")
    print(f"  reference_use_abs = {args.reference_use_abs}")

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
        shift_sigma_ppm=args.shift_sigma_ppm,
        per_compound_gain_sigma=args.per_compound_gain_sigma,
        per_compound_linewidth_sigma_max=args.per_compound_linewidth_sigma_max,
        white_noise_fraction=args.white_noise_fraction,
        correlated_noise_fraction=args.correlated_noise_fraction,
        baseline_fraction=args.baseline_fraction,
        absolute_white_noise_std=args.absolute_white_noise_std,
        absolute_correlated_noise_std=args.absolute_correlated_noise_std,
        absolute_baseline_std=args.absolute_baseline_std,
        intensity_jitter_fraction=args.intensity_jitter_fraction,
        global_scale_sigma=args.global_scale_sigma,
        local_gain_sigma=args.local_gain_sigma,
        local_gain_smooth=args.local_gain_smooth,
        linewidth_sigma_max=args.linewidth_sigma_max,
        mixture_shift_sigma_ppm=args.mixture_shift_sigma_ppm,
        mixture_extra_broadening_sigma_max=args.mixture_extra_broadening_sigma_max,
        threshold_fraction=args.threshold_fraction,
        clip_negative_intensities=args.clip_negative_intensities,
        normalize_generated_by_reference=args.normalize_generated_by_reference,
        reference_ppm=args.reference_ppm,
        reference_window=args.reference_window,
        target_reference_integral=args.target_reference_integral,
        reference_use_abs=args.reference_use_abs,
    )

    output_df.to_csv(args.output, index=False)

    print(f"\nSaved {args.n_samples} artificial spectra to:")
    print(args.output)


if __name__ == "__main__":
    main()
