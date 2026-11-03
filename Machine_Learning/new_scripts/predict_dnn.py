#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf


def check_required_keys(bundle):
    required_keys = [
        "keras_model_path",
        "target_cols",
        "original_ppm_cols",
        "void_mask",
        "x_scaler",
        "y_scaler",
    ]

    missing = [k for k in required_keys if k not in bundle]

    if missing:
        raise ValueError(
            "\nThe joblib bundle does not contain the required DNN preprocessing metadata.\n"
            f"Missing keys: {missing}\n\n"
            "Make sure the model was trained with the dense NN script that saves "
            "the void_mask, scalers, original_ppm_cols and keras_model_path."
        )


def resolve_model_path(bundle, joblib_path, override_model_path=None):
    """
    Locate the Keras .keras model file.

    Priority:
      1. --keras-model if provided
      2. keras_model_path stored in the joblib bundle
      3. same basename relative to the joblib directory
    """

    if override_model_path is not None:
        if not os.path.isfile(override_model_path):
            raise FileNotFoundError(
                f"Keras model file not found: {override_model_path}"
            )
        return override_model_path

    stored_path = bundle["keras_model_path"]

    if os.path.isfile(stored_path):
        return stored_path

    joblib_dir = os.path.dirname(os.path.abspath(joblib_path))
    relative_path = os.path.join(joblib_dir, os.path.basename(stored_path))

    if os.path.isfile(relative_path):
        return relative_path

    raise FileNotFoundError(
        "\nCould not find the Keras model file.\n"
        f"Stored path in joblib: {stored_path}\n"
        f"Tried relative path: {relative_path}\n\n"
        "Use --keras-model to provide the correct .keras model file."
    )


def load_and_preprocess_spectra(df, bundle):
    """
    Apply exactly the same preprocessing used during dense-NN training:

      1. select original ppm columns
      2. apply saved void_mask
      3. apply saved x_scaler
      4. return 2D matrix for Dense NN: samples x retained_features
    """

    original_ppm_cols = list(bundle["original_ppm_cols"])
    void_mask = np.asarray(bundle["void_mask"], dtype=bool)
    x_scaler = bundle["x_scaler"]

    missing_cols = [c for c in original_ppm_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            "\nSome ppm columns used during training are missing from the prediction CSV.\n"
            f"Number of missing columns: {len(missing_cols)}\n"
            f"First missing columns: {missing_cols[:10]}\n\n"
            "The prediction CSV must contain the same original ppm grid as the training CSV."
        )

    X = (
        df[original_ppm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    print(f"\nRaw prediction matrix shape: {X.shape}")
    print(f"Stored void_mask length: {len(void_mask)}")

    if len(void_mask) != X.shape[1]:
        raise ValueError(
            "\nThe stored void_mask length does not match the number of original ppm columns.\n"
            f"void_mask length: {len(void_mask)}\n"
            f"Raw prediction features: {X.shape[1]}\n\n"
            "This means the prediction CSV and the trained model bundle are not compatible."
        )

    X = X[:, void_mask]

    print(f"After void removal: {X.shape}")

    expected_features = getattr(x_scaler, "n_features_in_", None)

    if expected_features is not None and X.shape[1] != expected_features:
        raise ValueError(
            "\nThe processed prediction matrix has the wrong number of features "
            "for the saved x_scaler.\n"
            f"Processed features: {X.shape[1]}\n"
            f"x_scaler expected: {expected_features}"
        )

    X_scaled = x_scaler.transform(X)

    print(f"Dense NN input shape: {X_scaled.shape}")

    return X_scaled


def make_output_dataframe(input_df, pred_df):
    """
    Preserve non-ppm metadata columns from the input file if present.
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
    Save:
      1. one figure with one subplot per metabolite
      2. one mean concentration bar plot
    """

    available_targets = [c for c in target_cols if c in output_df.columns]

    if not available_targets:
        print("\nNo target concentration columns found for plotting.")
        return

    # ------------------------------------------------------------
    # One panel per metabolite
    # ------------------------------------------------------------

    n_targets = len(available_targets)
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=False,
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
            f"mean = {mean_val:.4g}\nstd = {std_val:.4g}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        ax.set_title(col)
        ax.set_xlabel("Spectrum index")
        ax.set_ylabel("Predicted concentration")

    for j in range(n_targets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Dense NN predicted concentrations per metabolite", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = f"{output_prefix}_per_metabolite_subplots.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved plot: {out_png}")

    # ------------------------------------------------------------
    # Mean concentration bar plot
    # ------------------------------------------------------------

    means = output_df[available_targets].mean(axis=0)
    stds = output_df[available_targets].std(axis=0)

    xpos = np.arange(len(available_targets))

    plt.figure(figsize=(max(10, 0.45 * len(available_targets)), 6))
    plt.bar(xpos, means, yerr=stds, capsize=4)
    plt.xticks(xpos, available_targets, rotation=90)
    plt.ylabel("Predicted concentration")
    plt.title("Dense NN mean predicted concentrations")
    plt.tight_layout()

    out_png = f"{output_prefix}_mean_concentrations.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved plot: {out_png}")


def save_ppm_used(bundle, output_prefix):
    """
    Save the ppm variables retained after void removal.
    """

    if "ppm_kept" not in bundle:
        return

    ppm_kept = np.asarray(bundle["ppm_kept"])

    out_csv = f"{output_prefix}_ppm_used_for_prediction.csv"

    pd.DataFrame(
        {
            "feature_index": np.arange(len(ppm_kept)),
            "ppm": ppm_kept,
        }
    ).to_csv(out_csv, index=False)

    print(f"Saved ppm features used for prediction: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict metabolite concentrations from NMR spectra using a trained "
            "Keras dense neural network with saved void-region preprocessing."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV containing spectra to predict.",
    )

    parser.add_argument(
        "-b",
        "--bundle",
        required=True,
        help="Joblib bundle saved during dense NN training.",
    )

    parser.add_argument(
        "--keras-model",
        default=None,
        help=(
            "Optional path to the .keras model file. "
            "If omitted, the script uses keras_model_path stored in the joblib bundle."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default="dense_predicted_concentrations.csv",
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

    bundle = joblib.load(args.bundle)

    if not isinstance(bundle, dict):
        raise ValueError(
            "\nThe loaded joblib object is not a dictionary bundle.\n"
            "Use the .joblib file saved by the dense NN training script."
        )

    print("\nLoaded dense NN joblib bundle.")
    print("Available keys:")
    for k in bundle.keys():
        print(f"  {k}")

    check_required_keys(bundle)

    model_type = bundle.get("model_type", "unknown")
    print(f"\nModel type in bundle: {model_type}")

    target_cols = list(bundle["target_cols"])
    y_scaler = bundle["y_scaler"]

    keras_model_path = resolve_model_path(
        bundle=bundle,
        joblib_path=args.bundle,
        override_model_path=args.keras_model,
    )

    print("\nLoading Keras model from:")
    print(keras_model_path)

    model = tf.keras.models.load_model(keras_model_path)

    X_dense = load_and_preprocess_spectra(
        df=df,
        bundle=bundle,
    )

    expected_shape = model.input_shape

    print(f"Model input shape: {expected_shape}")

    if expected_shape[-1] is not None and X_dense.shape[1] != expected_shape[-1]:
        raise ValueError(
            "\nThe processed input does not match the dense model input size.\n"
            f"Processed features: {X_dense.shape[1]}\n"
            f"Model expects: {expected_shape[-1]}"
        )

    print("\nPredicting concentrations with dense NN...")
    pred_scaled = model.predict(X_dense, verbose=0)

    pred = y_scaler.inverse_transform(pred_scaled)

    if args.clip_negative:
        pred = np.clip(pred, 0.0, None)

    if pred.shape[1] != len(target_cols):
        raise ValueError(
            "\nThe number of predicted columns does not match target_cols.\n"
            f"Prediction columns: {pred.shape[1]}\n"
            f"Target columns: {len(target_cols)}"
        )

    pred_df = pd.DataFrame(pred, columns=target_cols)

    output_df = make_output_dataframe(df, pred_df)
    output_df.to_csv(args.output, index=False)

    print(f"\nSaved dense NN predictions to: {args.output}")

    output_prefix = os.path.splitext(args.output)[0]

    save_ppm_used(
        bundle=bundle,
        output_prefix=output_prefix,
    )

    if not args.no_plots:
        plot_predictions(
            output_df=output_df,
            target_cols=target_cols,
            output_prefix=output_prefix,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
