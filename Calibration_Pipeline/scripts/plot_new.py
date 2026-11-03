#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import nmrglue as ng
except ImportError:
    sys.stderr.write("Error: nmrglue is not installed. Install it with: pip install nmrglue\n")
    sys.exit(1)

try:
    from scipy.signal import find_peaks, peak_widths, savgol_filter
except ImportError:
    sys.stderr.write("Error: scipy is not installed. Install it with: pip install scipy\n")
    sys.exit(1)


def read_peak_regions(csv_path: Path) -> dict[str, dict[str, float | str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Peak region file not found: {csv_path}")

    regions = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            compound = row["compound"].strip()
            regions[compound] = {
                "signal_name": row["signal_name"].strip(),
                "ppm_min": float(row["ppm_min"]),
                "ppm_max": float(row["ppm_max"]),
                "expected_protons": float(row["expected_protons"]),
            }
    return regions


def infer_compound_from_path(input_path: Path) -> str:
    parent = input_path.parent.name.strip()
    if parent.endswith("-Reference"):
        return parent.replace("-Reference", "")
    raise ValueError(f"Cannot infer compound from parent folder: {parent}")


def infer_nmr_mode_from_filename(input_path: Path) -> str:
    stem = input_path.stem
    try:
        sample_num = int(stem)
    except ValueError:
        raise ValueError(f"Cannot infer NMR mode from filename: {stem}")

    unit = sample_num % 10
    if unit == 0:
        return "pure_shift"
    elif unit == 1:
        return "standard"
    else:
        raise ValueError(
            f"Sample number {sample_num} does not match expected convention "
            f"(ending with 0 for Pure Shift or 1 for Standard)."
        )


def load_jcamp_spectrum(path: Path):
    dic, data = ng.jcampdx.read(str(path))
    if isinstance(data, list):
        real = np.asarray(data[0], dtype=float).ravel()
        imag = np.asarray(data[1], dtype=float).ravel() if len(data) > 1 else None
    else:
        arr = np.asarray(data).ravel()
        if np.iscomplexobj(arr):
            real = arr.real.astype(float)
            imag = arr.imag.astype(float)
        else:
            real = arr.astype(float)
            imag = None
    return dic, real, imag


def make_ppm_axis(dic: dict, data: np.ndarray) -> np.ndarray:
    udic = ng.jcampdx.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)
    return uc.ppm_scale()


def choose_odd_window_length(n: int, requested: int) -> int:
    if n < 3:
        return 0
    w = min(requested, n if n % 2 == 1 else n - 1)
    if w < 3:
        return 0
    if w % 2 == 0:
        w -= 1
    return w


def smooth_signal(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    wl = choose_odd_window_length(len(y), window_length)
    if wl == 0:
        return y.copy()
    po = min(polyorder, wl - 1)
    if po < 1:
        return y.copy()
    return savgol_filter(y, window_length=wl, polyorder=po, mode="interp")


def robust_reference_peak(
    ppm_axis: np.ndarray,
    intensity: np.ndarray,
    guess_ppm: float,
    window_ppm: float,
    peak_sign: str = "positive",
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    prominence_fraction: float = 0.10,
    rel_height: float = 0.5,
):
    low = guess_ppm - window_ppm
    high = guess_ppm + window_ppm
    mask = (ppm_axis >= low) & (ppm_axis <= high)
    if not np.any(mask):
        mask = (ppm_axis >= high) & (ppm_axis <= low)
    idx_global = np.where(mask)[0]
    if idx_global.size < 3:
        raise ValueError("Too few points found in the reference window.")

    x_local = ppm_axis[idx_global]
    y_local_raw = intensity[idx_global].astype(float)

    if peak_sign == "positive":
        y_work = y_local_raw.copy()
    elif peak_sign == "negative":
        y_work = -y_local_raw
    elif peak_sign == "absolute":
        y_work = np.abs(y_local_raw)
    else:
        raise ValueError("peak_sign must be 'positive', 'negative', or 'absolute'.")

    y_smooth = smooth_signal(y_work, smooth_window, smooth_polyorder)
    y_range = float(np.max(y_smooth) - np.min(y_smooth))
    min_prom = max(prominence_fraction * y_range, 0.0)
    peaks, props = find_peaks(y_smooth, prominence=min_prom)

    if peaks.size == 0:
        peak_local_idx = int(np.argmax(y_smooth))
    else:
        prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
        peak_ppm_candidates = x_local[peaks]
        distance_penalty = np.abs(peak_ppm_candidates - guess_ppm)
        score = prominences - 0.25 * np.max(prominences + 1e-12) * (
            distance_penalty / max(window_ppm, 1e-12)
        )
        peak_local_idx = int(peaks[int(np.argmax(score))])

    widths, _, left_ips, right_ips = peak_widths(y_smooth, [peak_local_idx], rel_height=rel_height)
    left_i = int(max(0, np.floor(left_ips[0])))
    right_i = int(min(len(y_local_raw) - 1, np.ceil(right_ips[0])))
    if right_i <= left_i:
        left_i = max(0, peak_local_idx - 2)
        right_i = min(len(y_local_raw) - 1, peak_local_idx + 2)

    x_peak = x_local[left_i:right_i + 1]
    y_peak_raw = y_local_raw[left_i:right_i + 1]

    if peak_sign == "positive":
        baseline = np.min(y_peak_raw)
        weights = y_peak_raw - baseline
    elif peak_sign == "negative":
        baseline = np.max(y_peak_raw)
        weights = baseline - y_peak_raw
    else:
        weights = np.abs(y_peak_raw)

    weights = np.clip(weights, 0.0, None)
    if np.sum(weights) <= 0:
        center_ppm = float(x_local[peak_local_idx])
    else:
        center_ppm = float(np.sum(x_peak * weights) / np.sum(weights))

    nearest_idx = int(np.argmin(np.abs(ppm_axis - center_ppm)))
    peak_height = float(intensity[nearest_idx])
    return center_ppm, peak_height


def extract_region(ppm: np.ndarray, y: np.ndarray, ppm_min: float, ppm_max: float):
    low = min(ppm_min, ppm_max)
    high = max(ppm_min, ppm_max)
    mask = (ppm >= low) & (ppm <= high)
    idx = np.where(mask)[0]
    if idx.size < 2:
        raise ValueError("Not enough points found in the integration interval.")
    x_region = ppm[idx]
    y_region = y[idx]
    order = np.argsort(x_region)
    return x_region[order], y_region[order]


def linear_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, x1 = x[0], x[-1]
    y0, y1 = y[0], y[-1]
    if x1 == x0:
        return np.full_like(y, y0)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def edge_baseline(x: np.ndarray, y: np.ndarray, edge_fraction: float = 0.15) -> np.ndarray:
    n = len(y)
    k = max(2, int(np.ceil(edge_fraction * n)))
    k = min(k, n // 2) if n >= 4 else 2

    left_y = np.median(y[:k])
    right_y = np.median(y[-k:])

    x0, x1 = x[0], x[-1]
    if x1 == x0:
        return np.full_like(y, left_y)

    return left_y + (right_y - left_y) * (x - x0) / (x1 - x0)


def robust_noise_sigma(y: np.ndarray) -> float:
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    sigma = 1.4826 * mad
    return float(sigma)


def integrate_with_noise_rejection(
    x: np.ndarray,
    y: np.ndarray,
    baseline_mode: str = "edge",
    edge_fraction: float = 0.15,
    noise_sigma_multiplier: float = 2.0,
):
    if baseline_mode == "edge":
        baseline = edge_baseline(x, y, edge_fraction=edge_fraction)
    elif baseline_mode == "linear":
        baseline = linear_baseline(x, y)
    else:
        raise ValueError("baseline_mode must be 'edge' or 'linear'.")

    y_corr = y - baseline
    sigma = robust_noise_sigma(y_corr)
    threshold = noise_sigma_multiplier * sigma

    y_integrand = y_corr.copy()
    y_integrand[y_integrand < threshold] = 0.0

    integral = float(np.trapezoid(y_integrand, x=x))
    return integral, baseline, y_corr, y_integrand, sigma, threshold


def build_title(dic: dict, filename: str, extras: list[str]) -> str:
    nucleus = dic.get(".OBSERVE NUCLEUS") or dic.get("NUC1") or dic.get("AXNUC")
    obsfreq = dic.get(".OBSERVE FREQUENCY") or dic.get("SFO1") or dic.get("SF")
    parts = [filename]
    if nucleus is not None:
        parts.append(str(nucleus))
    if obsfreq is not None:
        try:
            parts.append(f"{float(obsfreq):.3f} MHz")
        except Exception:
            parts.append(str(obsfreq))
    parts.extend([e for e in extras if e])
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-phase, re-reference, plot, and integrate a JCAMP-DX 1D NMR spectrum."
    )
    parser.add_argument("input", help="Input JCAMP-DX file (.dx or .jdx)")

    parser.add_argument("--ppm", action="store_true", help="Plot against ppm")
    parser.add_argument("--imag", action="store_true", help="Also plot imaginary channel")
    parser.add_argument("--linewidth", type=float, default=0.9)
    parser.add_argument("--title", default=None)
    parser.add_argument("-o", "--output", default=None)

    parser.add_argument("--autophase", action="store_true")
    parser.add_argument("--phase-method", choices=["acme", "peak_minima"], default="acme")
    parser.add_argument("--p0", type=float, default=0.0)
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--peak-width", type=int, default=100)

    parser.add_argument("--med-baseline", action="store_true")
    parser.add_argument("--bl-mw", type=float, default=24)
    parser.add_argument("--bl-sf", type=float, default=16)
    parser.add_argument("--bl-sigma", type=float, default=5.0)

    parser.add_argument("--ppm-offset", type=float, default=0.0)
    parser.add_argument("--ref-current", type=float, default=None)
    parser.add_argument("--ref-target", type=float, default=0.0)
    parser.add_argument("--auto-ref", action="store_true")
    parser.add_argument("--ref-guess", type=float, default=None)
    parser.add_argument("--ref-window", type=float, default=0.3)
    parser.add_argument("--peak-sign", choices=["positive", "negative", "absolute"], default="positive")
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-polyorder", type=int, default=3)
    parser.add_argument("--prominence-fraction", type=float, default=0.10)
    parser.add_argument("--rel-height", type=float, default=0.5)
    parser.add_argument("--mark-ref-peak", action="store_true")

    parser.add_argument("--integrate", action="store_true")
    parser.add_argument("--int-min", type=float, default=None)
    parser.add_argument("--int-max", type=float, default=None)

    parser.add_argument(
        "--standard-regions-csv",
        type=str,
        default="data/metadata/peak_regions_standard.csv",
        help="CSV containing integration bounds for standard NMR",
    )
    parser.add_argument(
        "--pure-shift-regions-csv",
        type=str,
        default="data/metadata/peak_regions_pure_shift.csv",
        help="CSV containing integration bounds for pure shift NMR",
    )
    parser.add_argument(
        "--compound",
        type=str,
        default=None,
        help="Compound name. If omitted, inferred from parent folder.",
    )
    parser.add_argument(
        "--nmr-mode",
        choices=["standard", "pure_shift"],
        default=None,
        help="NMR mode. If omitted, inferred from file name.",
    )

    parser.add_argument("--baseline-correct", action="store_true")
    parser.add_argument("--baseline-mode", choices=["linear", "edge"], default="edge")
    parser.add_argument("--edge-fraction", type=float, default=0.15)
    parser.add_argument("--noise-aware-integration", action="store_true")
    parser.add_argument("--noise-sigma-multiplier", type=float, default=2.0)
    parser.add_argument("--shade-integral", action="store_true")

    parser.add_argument("--xscale", default="linear")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: file not found: {input_path}\n")
        sys.exit(1)
    if input_path.suffix.lower() not in {".dx", ".jdx"}:
        sys.stderr.write("Error: this script expects a JCAMP-DX file (.dx or .jdx).\n")
        sys.exit(1)

    compound = args.compound if args.compound is not None else infer_compound_from_path(input_path)
    nmr_mode = args.nmr_mode if args.nmr_mode is not None else infer_nmr_mode_from_filename(input_path)

    standard_regions = read_peak_regions(Path(args.standard_regions_csv))
    pure_shift_regions = read_peak_regions(Path(args.pure_shift_regions_csv))

    if args.integrate and (args.int_min is None or args.int_max is None):
        if nmr_mode == "standard":
            if compound not in standard_regions:
                raise ValueError(f"Compound '{compound}' not found in standard regions CSV.")
            args.int_min = float(standard_regions[compound]["ppm_min"])
            args.int_max = float(standard_regions[compound]["ppm_max"])
        elif nmr_mode == "pure_shift":
            if compound not in pure_shift_regions:
                raise ValueError(f"Compound '{compound}' not found in pure shift regions CSV.")
            args.int_min = float(pure_shift_regions[compound]["ppm_min"])
            args.int_max = float(pure_shift_regions[compound]["ppm_max"])

    dic, real, imag = load_jcamp_spectrum(input_path)
    extras = [f"compound={compound}", f"mode={nmr_mode}"]
    marked_x = None
    marked_y = None
    x_region = None
    y_region = None
    baseline = None
    y_to_fill = None

    if args.autophase:
        if imag is None:
            print("Warning: no imaginary channel found; autophase skipped.")
            extras.append("autophase skipped")
        else:
            complex_data = real + 1j * imag
            autops_kwargs = {"disp": False}
            if args.phase_method == "peak_minima":
                autops_kwargs["peak_width"] = args.peak_width
            phased, phases = ng.proc_autophase.autops(
                complex_data,
                args.phase_method,
                p0=args.p0,
                p1=args.p1,
                return_phases=True,
                **autops_kwargs,
            )
            real = phased.real.astype(float)
            imag = phased.imag.astype(float)
            extras.append(f"autophase p0={phases[0]:.2f} p1={phases[1]:.2f}")
            print(f"Auto phase: p0 = {phases[0]:.6f} deg, p1 = {phases[1]:.6f} deg")

    if args.med_baseline:
        real = ng.proc_bl.med(real, mw=args.bl_mw, sf=args.bl_sf, sigma=args.bl_sigma)
        extras.append("median baseline corrected")

    if args.ppm:
        x = make_ppm_axis(dic, real)
        xlabel = "ppm"

        if args.ppm_offset != 0.0:
            x = x + args.ppm_offset
            extras.append(f"offset {args.ppm_offset:+.4f} ppm")

        if args.auto_ref:
            if args.ref_guess is None:
                sys.stderr.write("Error: --auto-ref requires --ref-guess.\n")
                sys.exit(1)
            try:
                center_ppm, peak_height = robust_reference_peak(
                    x, real, args.ref_guess, args.ref_window,
                    peak_sign=args.peak_sign,
                    smooth_window=args.smooth_window,
                    smooth_polyorder=args.smooth_polyorder,
                    prominence_fraction=args.prominence_fraction,
                    rel_height=args.rel_height,
                )
                delta = args.ref_target - center_ppm
                x = x + delta
                extras.append(f"auto ref {center_ppm:.4f}->{args.ref_target:.4f} ppm")
                marked_x = args.ref_target
                marked_y = peak_height
                print(f"Auto reference: detected center = {center_ppm:.6f} ppm, applied shift = {delta:+.6f} ppm")
            except Exception as e:
                print(f"Warning: auto reference skipped: {e}")
                extras.append("auto ref skipped")
        elif args.ref_current is not None:
            delta = args.ref_target - args.ref_current
            x = x + delta
            extras.append(f"manual ref {args.ref_current:.4f}->{args.ref_target:.4f} ppm")
            marked_x = args.ref_target
            idx_closest = int(np.argmin(np.abs(x - args.ref_target)))
            marked_y = real[idx_closest]
            print(f"Manual reference: applied shift = {delta:+.6f} ppm")

        if args.integrate:
            if args.int_min is None or args.int_max is None:
                raise ValueError("Integration requested but no bounds could be determined.")

            x_region, y_region = extract_region(x, real, args.int_min, args.int_max)

            if args.baseline_correct:
                if args.noise_aware_integration:
                    integral, baseline, y_corr, y_integrand, sigma, threshold = integrate_with_noise_rejection(
                        x_region,
                        y_region,
                        baseline_mode=args.baseline_mode,
                        edge_fraction=args.edge_fraction,
                        noise_sigma_multiplier=args.noise_sigma_multiplier,
                    )
                    y_to_fill = baseline + y_integrand
                    extras.append(f"I[{min(args.int_min,args.int_max):.2f},{max(args.int_min,args.int_max):.2f}]={integral:.6g}")
                    extras.append(f"noise-thr={threshold:.3g}")
                    print(f"Integration interval : {x_region[0]:.6f} to {x_region[-1]:.6f} ppm")
                    print(f"Number of points     : {x_region.size}")
                    print(f"Baseline corrected   : True")
                    print(f"Noise-aware mode     : True")
                    print(f"Estimated sigma      : {sigma:.10g}")
                    print(f"Threshold            : {threshold:.10g}")
                    print(f"Integral             : {integral:.10g}")
                else:
                    if args.baseline_mode == "edge":
                        baseline = edge_baseline(x_region, y_region, edge_fraction=args.edge_fraction)
                    else:
                        baseline = linear_baseline(x_region, y_region)

                    y_to_integrate = y_region - baseline
                    y_to_integrate = np.clip(y_to_integrate, 0.0, None)
                    y_to_fill = baseline + y_to_integrate
                    integral = float(np.trapezoid(y_to_integrate, x=x_region))

                    extras.append(f"I[{min(args.int_min,args.int_max):.2f},{max(args.int_min,args.int_max):.2f}]={integral:.6g}")
                    print(f"Integration interval : {x_region[0]:.6f} to {x_region[-1]:.6f} ppm")
                    print(f"Number of points     : {x_region.size}")
                    print(f"Baseline corrected   : True")
                    print(f"Noise-aware mode     : False")
                    print(f"Integral             : {integral:.10g}")
            else:
                y_to_integrate = np.clip(y_region, 0.0, None)
                y_to_fill = y_to_integrate
                integral = float(np.trapezoid(y_to_integrate, x=x_region))

                extras.append(f"I[{min(args.int_min,args.int_max):.2f},{max(args.int_min,args.int_max):.2f}]={integral:.6g}")
                print(f"Integration interval : {x_region[0]:.6f} to {x_region[-1]:.6f} ppm")
                print(f"Number of points     : {x_region.size}")
                print(f"Baseline corrected   : False")
                print(f"Noise-aware mode     : False")
                print(f"Integral             : {integral:.10g}")
    else:
        x = np.arange(real.size)
        xlabel = "Point index"
        if args.autophase or args.integrate or args.ppm_offset != 0.0 or args.auto_ref or args.ref_current is not None:
            sys.stderr.write("Warning: phasing works, but referencing and integration need --ppm to be meaningful.\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, real, linewidth=args.linewidth, label="Real")

    if args.imag and imag is not None:
        ax.plot(x if args.ppm and imag.size == real.size else np.arange(imag.size),
                imag, linewidth=args.linewidth, label="Imag")
        ax.legend()

    if args.ppm and args.integrate and x_region is not None and args.shade_integral:
        if args.baseline_correct and baseline is not None:
            ax.plot(x_region, baseline, linewidth=1.0, linestyle="--", label="Baseline")
            if y_to_fill is not None:
                ax.fill_between(x_region, y_to_fill, baseline, alpha=0.3, label="Integrated region")
        else:
            ax.fill_between(x_region, y_region, 0.0, alpha=0.3, label="Integrated region")
        ax.legend()

    if args.ppm and args.mark_ref_peak and marked_x is not None and marked_y is not None:
        ax.plot([marked_x], [marked_y], marker="o", linestyle="None")
        ax.annotate(f"{marked_x:.3f} ppm", xy=(marked_x, marked_y), xytext=(8, 8),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(args.title if args.title else build_title(dic, input_path.name, extras))

    if args.ppm:
        ax.set_xlim(np.max(x), np.min(x))

    ax.set_xscale(args.xscale)

    if args.x_min is not None or args.x_max is not None:
        ax.set_xlim(left=args.x_min, right=args.x_max)
    elif args.ppm:
        ax.set_xlim(np.max(x), np.min(x))

    plt.ylim(-0.5, None)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()