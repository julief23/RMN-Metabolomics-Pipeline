#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


def ppm_from_headers(ppm_cols):
    """
    Convert headers like ppm_1234.56 or ppm_1234,56 into numeric ppm values.

    This follows the same convention as the training script:
        ppm = float(header_without_ppm_) / 1000.0
    """

    ppm = []

    for c in ppm_cols:
        value = str(c).replace("ppm_", "").replace(",", ".")
        ppm.append(float(value))

    return np.array(ppm)


def bin_spectra(X, ppm, n_bins):
    """
    Apply the exact same binning logic as in the training script.

    The ppm axis is sorted, split into n_bins approximately equal groups,
    and each bin is represented by the mean intensity.
    """

    order = np.argsort(ppm)
    ppm_sorted = ppm[order]
    X_sorted = X[:, order]

    bins = np.array_split(np.arange(len(ppm_sorted)), n_bins)

    X_binned = []
    ppm_binned = []

    for b in bins:
        X_binned.append(X_sorted[:, b].mean(axis=1))
        ppm_binned.append(ppm_sorted[b].mean())

    X_binned = np.vstack(X_binned).T
    ppm_binned = np.array(ppm_binned)

    return X_binned, ppm_binned


def get_model_expected_features(model):
    """
    Recover the number of features expected by the trained sklearn model/pipeline.
    """

    if hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "n_features_in_"):
                return step.n_features_in_

    if hasattr(model, "n_features_in_"):
        return model.n_features_in_

    return None


def check_required_keys(bundle):
    required_keys = [
        "model",
        "target_cols",
        "original_ppm_cols",
        "n_bins",
        "void_mask",
    ]

    missing = [k for k in required_keys if k not in bundle]

    if missing:
        raise ValueError(
            "\nThe joblib file does not contain the required preprocessing metadata.\n"
            f"Missing keys: {missing}\n\n"
            "This usually means that you are using an old model file, or that the "
            "training script was not rerun after adding the saved void_mask.\n\n"
            "Please retrain with the modified training script and make sure that "
            "the prediction command points to the newly generated .joblib file."
        )


def load_and_preprocess_prediction_spectra(df, bundle):
    """
    Reproduce the training preprocessing:

        1. select original training ppm columns
        2. optional binning
        3. apply stored void mask
    """

    original_ppm_cols = list(bundle["original_ppm_cols"])
    n_bins = bundle["n_bins"]
    void_mask = np.asarray(bundle["void_mask"], dtype=bool)

    missing_cols = [c for c in original_ppm_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            "\nSome ppm columns used during training are missing from the prediction CSV.\n"
            f"Number of missing columns: {len(missing_cols)}\n"
            f"First missing columns: {missing_cols[:10]}\n\n"
            "The prediction CSV must have the same original ppm grid as the training CSV."
        )

    X = (
        df[original_ppm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    ppm = ppm_from_headers(original_ppm_cols)

    print(f"\nRaw prediction matrix shape: {X.shape}")

    if n_bins is not None and n_bins > 0 and n_bins < X.shape[1]:
        X, ppm_after_binning = bin_spectra(X, ppm, n_bins)
        print(f"After binning: {X.shape}")
    else:
        ppm_after_binning = ppm.copy()
        print("No binning applied.")

    print(f"Stored void mask length: {len(void_mask)}")

    if len(void_mask) != X.shape[1]:
        raise ValueError(
            "\nThe stored void_mask length does not match the prediction matrix "
            "after binning.\n\n"
            f"void_mask length: {len(void_mask)}\n"
            f"X features after binning/no-binning: {X.shape[1]}\n\n"
            "This means the prediction preprocessing does not match the training "
            "preprocessing. Check that you are using the correct model file and "
            "the same n_bins setting used during training."
        )

    X = X[:, void_mask]
    ppm_kept = ppm_after_binning[void_mask]

    print(f"After applying stored void mask: {X.shape}")

    return X, ppm_kept


def make_output_dataframe(input_df, pred_df):
    """
    Preserve non-spectral metadata columns if present.
    """

    metadata_cols = [c for c in input_df.columns if not str(c).startswith("ppm_")]

    if metadata_cols:
        output_df = pd.concat(
            [
                input_df[metadata_cols].reset_index(drop=True),
                pred_df.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        output_df = pred_df.copy()
        output_df.insert(0, "sample_index", np.arange(len(output_df)))

    return output_df


def plot_predictions(output_df, target_cols, output_prefix):
    """
    Save diagnostic plots:
      1. one figure with separate subplots, one subplot per metabolite
      2. mean ± standard deviation across spectra
    """

    available_targets = [c for c in target_cols if c in output_df.columns]

    if not available_targets:
        print("\nNo target concentration columns found for plotting.")
        return

    # ------------------------------------------------------------
    # 1. Separate subplot for each metabolite on the same figure
    # ------------------------------------------------------------

    n_targets = len(available_targets)

    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
    )

    axes = np.array(axes).reshape(-1)

    x = np.arange(len(output_df))

    for i, col in enumerate(available_targets):
        ax = axes[i]

        y = output_df[col].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1,
            markersize=3,
        )

        ax.set_title(col)
        ax.set_ylabel("Concentration")

        mean_val = np.mean(y)
        std_val = np.std(y)

        ax.axhline(
            mean_val,
            linestyle="--",
            linewidth=1,
        )

        ax.text(
            0.03,
            0.95,
            f"mean = {mean_val:.3g}\nstd = {std_val:.3g}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    for j in range(n_targets, len(axes)):
        fig.delaxes(axes[j])

    for ax in axes[:n_targets]:
        ax.set_xlabel("Spectrum index")

    fig.suptitle("Predicted concentrations per metabolite", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = f"{output_prefix}_per_metabolite_subplots.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved plot: {out_png}")

    # ------------------------------------------------------------
    # 2. Mean concentration bar plot
    # ------------------------------------------------------------

    means = output_df[available_targets].mean(axis=0)
    stds = output_df[available_targets].std(axis=0)

    xpos = np.arange(len(available_targets))

    plt.figure(figsize=(max(10, 0.45 * len(available_targets)), 6))
    plt.bar(xpos, means, yerr=stds, capsize=4)
    plt.xticks(xpos, available_targets, rotation=90)
    plt.ylabel("Predicted concentration")
    plt.title("Mean predicted concentrations")
    plt.tight_layout()

    out_png = f"{output_prefix}_mean_concentrations.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved plot: {out_png}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict concentrations from NMR spectra using a PLS model trained "
            "with saved binning and void-region preprocessing."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV containing spectra to predict.",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Joblib model file generated by the modified training script.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="predicted_concentrations.csv",
        help="Output CSV file with predicted concentrations.",
    )

    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Set negative predicted concentrations to zero.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable diagnostic plots.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bundle = joblib.load(args.model)

    if not isinstance(bundle, dict):
        raise ValueError(
            "\nThe loaded joblib object is not a dictionary bundle.\n"
            "This script expects the model file generated by the modified training script."
        )

    print("\nLoaded model bundle.")
    print("Available keys:")
    for k in bundle.keys():
        print(f"  {k}")

    check_required_keys(bundle)

    model = bundle["model"]
    target_cols = list(bundle["target_cols"])

    X_pred, ppm_kept = load_and_preprocess_prediction_spectra(
        df=df,
        bundle=bundle,
    )

    expected_n_features = get_model_expected_features(model)

    if expected_n_features is not None:
        print(f"Model expects: {expected_n_features} features")

        if X_pred.shape[1] != expected_n_features:
            raise ValueError(
                "\nThe processed prediction matrix still has the wrong number of features.\n"
                f"Processed prediction features: {X_pred.shape[1]}\n"
                f"Model expected features: {expected_n_features}\n\n"
                "This should not happen if the correct model bundle is used. "
                "Check that the model was retrained with the modified training script."
            )

    print("\nPredicting concentrations...")
    pred = model.predict(X_pred)

    pred = np.asarray(pred)

    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    if pred.shape[1] != len(target_cols):
        raise ValueError(
            "\nThe number of predicted columns does not match target_cols.\n"
            f"Prediction columns: {pred.shape[1]}\n"
            f"Target columns: {len(target_cols)}"
        )

    pred_df = pd.DataFrame(pred, columns=target_cols)

    if args.clip_negative:
        pred_df = pred_df.clip(lower=0.0)

    output_df = make_output_dataframe(df, pred_df)

    output_df.to_csv(args.output, index=False)

    print(f"\nSaved predictions to: {args.output}")

    ppm_debug_csv = os.path.splitext(args.output)[0] + "_ppm_used_for_prediction.csv"
    pd.DataFrame(
        {
            "feature_index": np.arange(len(ppm_kept)),
            "ppm": ppm_kept,
        }
    ).to_csv(ppm_debug_csv, index=False)

    print(f"Saved ppm features used for prediction to: {ppm_debug_csv}")

    if not args.no_plots:
        output_prefix = os.path.splitext(args.output)[0]
        plot_predictions(
            output_df=output_df,
            target_cols=target_cols,
            output_prefix=output_prefix,
        )


if __name__ == "__main__":
    main()
