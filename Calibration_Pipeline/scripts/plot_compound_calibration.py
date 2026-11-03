#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def fit_linear_model(x: np.ndarray, y: np.ndarray):
    x_2d = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_2d, y)

    y_pred = model.predict(x_2d)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    return slope, intercept, r2, rmse, y_pred


def make_empty_plot(compound, nmr_label, output_png, reason):
    plt.figure(figsize=(8, 6))
    plt.text(
        0.5, 0.5,
        f"No exploitable data\n{reason}",
        ha="center", va="center", fontsize=12
    )
    plt.axis("off")
    plt.title(f"Calibration curve of {compound} - {nmr_label}")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "compound": compound,
        "nmr_type": nmr_label,
        "title": f"Calibration curve of {compound} - {nmr_label}",
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "equation": "NA",
        "n_points": 0,
        "status": f"no_data: {reason}",
    }


def make_plot(df_sub, compound, nmr_label, output_png):
    x = df_sub["final_integral"].to_numpy(dtype=float)
    y = df_sub["concentration_used"].to_numpy(dtype=float)

    if len(df_sub) < 2:
        return make_empty_plot(compound, nmr_label, output_png, "fewer than 2 valid points")

    slope, intercept, r2, rmse, y_pred = fit_linear_model(x, y)

    order = np.argsort(x)
    x_sorted = x[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Experimental data")
    plt.plot(x_sorted, y_pred_sorted, label="Linear regression")

    plt.xlabel("Final integral = integral_log / (2 × integral_log_0)")
    plt.ylabel("Concentration")
    plt.title(f"Calibration curve of {compound} - {nmr_label}")
    plt.legend()

    txt = (
        f"Slope = {slope:.6g}\n"
        f"Intercept = {intercept:.6g}\n"
        f"R² = {r2:.6f}\n"
        f"RMSE = {rmse:.6g}"
    )

    plt.text(
        0.05, 0.95, txt,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "compound": compound,
        "nmr_type": nmr_label,
        "title": f"Calibration curve of {compound} - {nmr_label}",
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "rmse": rmse,
        "equation": f"y = {slope:.6g}x + {intercept:.6g}",
        "n_points": len(df_sub),
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compound", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = ["nmr_type", "concentration_used", "integral_log", "integral_log_0_x2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    df["concentration_used"] = pd.to_numeric(df["concentration_used"], errors="coerce")
    df["integral_log"] = pd.to_numeric(df["integral_log"], errors="coerce")
    df["integral_log_0_x2"] = pd.to_numeric(df["integral_log_0_x2"], errors="coerce")

    raw_df = df.copy()

    df = df[df["integral_log_0_x2"] != 0].copy()
    df["final_integral"] = df["integral_log"] / df["integral_log_0_x2"]

    df = df.dropna(
        subset=[
            "nmr_type",
            "concentration_used",
            "integral_log",
            "integral_log_0_x2",
            "final_integral"
        ]
    ).copy()

    df["nmr_type_clean"] = df["nmr_type"].astype(str).str.strip().str.lower()

    pure_df = df[df["nmr_type_clean"] == "pure shift"].copy()
    standard_df = df[df["nmr_type_clean"] == "standard"].copy()

    pure_df.to_csv(output_dir / f"{args.compound}_pure_shift_points.csv", index=False)
    standard_df.to_csv(output_dir / f"{args.compound}_standard_points.csv", index=False)

    results = []

    results.append(
        make_plot(
            pure_df,
            args.compound,
            "RMN Pure shift",
            output_dir / f"calibration_{args.compound}_pure_shift.png",
        )
    )

    results.append(
        make_plot(
            standard_df,
            args.compound,
            "RMN Standard",
            output_dir / f"calibration_{args.compound}_standard.png",
        )
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        output_dir / f"calibration_{args.compound}_summary.csv",
        index=False
    )

    df.to_csv(output_dir / f"{args.compound}_with_final_integral.csv", index=False)
    raw_df.to_csv(output_dir / f"{args.compound}_raw_input_copy.csv", index=False)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()