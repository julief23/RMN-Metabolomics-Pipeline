#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def detect_columns(df):
    ppm_cols = [c for c in df.columns if str(c).startswith("ppm_")]
    non_ppm_cols = [c for c in df.columns if c not in ppm_cols]

    target_cols = []
    ignored_cols = []

    for c in non_ppm_cols:
        values = pd.to_numeric(df[c], errors="coerce")

        if values.notna().any():
            target_cols.append(c)
            df[c] = values.fillna(0.0)
        else:
            ignored_cols.append(c)

    if not ppm_cols:
        raise ValueError("No ppm_ spectral columns found.")

    if not target_cols:
        raise ValueError("No numeric concentration columns found.")

    print(f"Detected {len(ppm_cols)} spectral columns.")
    print(f"Detected {len(target_cols)} concentration columns.")

    if ignored_cols:
        print("Ignored metadata columns:")
        for c in ignored_cols:
            print(f"  {c}")

    return target_cols, ppm_cols


def ppm_from_headers(ppm_cols):
    """
    Convert ppm headers into numeric ppm values.

    Example:
        ppm_1250.0 -> 1.250
    """

    ppm = []

    for c in ppm_cols:
        value = str(c).replace("ppm_", "").replace(",", ".")
        ppm.append(float(value))

    return np.array(ppm)


def bin_spectra(X, ppm, n_bins):
    """
    Bin spectra using exactly the same deterministic logic that must also be
    used during prediction.

    The ppm axis is sorted, split into approximately equal index groups,
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


def make_void_mask_from_training(
    X,
    threshold_fraction=0.001,
    min_keep_points=20,
):
    """
    Create a boolean mask that removes spectral variables with almost no signal.

    A variable is kept if:

        mean(abs(X[:, j])) > threshold_fraction * max(mean(abs(X)))

    Important:
    this mask must be computed during training, saved in the joblib file,
    and reused unchanged for prediction.
    """

    mean_abs_signal = np.mean(np.abs(X), axis=0)
    global_max = np.max(mean_abs_signal)

    if global_max <= 0:
        raise ValueError("Spectral matrix contains no signal.")

    threshold = threshold_fraction * global_max
    mask = mean_abs_signal > threshold

    if mask.sum() < min_keep_points:
        raise ValueError(
            f"Void mask retained only {mask.sum()} points. "
            f"Decrease --void_threshold_fraction."
        )

    return mask


def parse_exclude_ranges(exclude_ranges):
    parsed = []

    if exclude_ranges is None:
        return parsed

    for r in exclude_ranges:
        if ":" not in r:
            raise ValueError(
                f"Invalid exclusion range '{r}'. Use format low:high, e.g. 4.65:4.90"
            )

        a, b = r.split(":")
        low = float(a)
        high = float(b)

        if low > high:
            low, high = high, low

        parsed.append((low, high))

    return parsed


def apply_manual_exclusions(mask, ppm, exclude_ranges):
    """
    Apply user-defined ppm exclusions to an existing mask.
    """

    parsed = parse_exclude_ranges(exclude_ranges)

    if not parsed:
        return mask

    new_mask = mask.copy()

    for low, high in parsed:
        exclude = (ppm >= low) & (ppm <= high)
        new_mask[exclude] = False
        print(f"Excluded ppm range: {low:g} to {high:g}")

    if new_mask.sum() == 0:
        raise ValueError("Manual exclusions removed all ppm points.")

    return new_mask


def plot_kept_regions(ppm, mask, output_png):
    plt.figure(figsize=(12, 3))
    plt.scatter(ppm, mask.astype(int), s=4)
    plt.gca().invert_xaxis()
    plt.xlabel("ppm")
    plt.ylabel("Kept point")
    plt.title("PPM points retained after void-region removal")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved kept-region plot to: {output_png}")


def save_kept_regions_csv(ppm_after_binning, void_mask, output_csv):
    table = pd.DataFrame(
        {
            "feature_index_after_binning": np.arange(len(ppm_after_binning)),
            "ppm": ppm_after_binning,
            "kept": void_mask.astype(int),
        }
    )

    table.to_csv(output_csv, index=False)
    print(f"Saved kept-region CSV to: {output_csv}")


def choose_best_n_components(X_train, y_train, max_components, seed):
    max_allowed = min(max_components, X_train.shape[0] - 1, X_train.shape[1])

    if max_allowed < 1:
        raise ValueError("Not enough samples/features to fit PLS.")

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    best_n = 1
    best_score = -np.inf
    scores = []

    print("\nCross-validating number of PLS components:")

    for n in range(1, max_allowed + 1):
        model = Pipeline(
            steps=[
                ("x_scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=n, scale=False)),
            ]
        )

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        ).mean()

        scores.append((n, score))
        print(f"  n_components={n:3d}  CV R2={score:.6f}")

        if score > best_score:
            best_score = score
            best_n = n

    print(f"\nBest number of components: {best_n}")
    print(f"Best CV R2: {best_score:.6f}")

    cv_df = pd.DataFrame(scores, columns=["n_components", "cv_r2"])

    return best_n, cv_df


def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)

    print("\nGlobal metrics:")
    print(f"MAE  = {mean_absolute_error(y_test, y_pred):.6f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.6f}")
    print(f"R2   = {r2_score(y_test, y_pred):.6f}")

    print("\nPer-compound metrics:")
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])

        print(f"{name}")
        print(f"  MAE  = {mae:.6f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  R2   = {r2:.6f}")

    return y_pred


def save_predictions(y_test, y_pred, target_names, output_csv):
    data = {}

    for i, name in enumerate(target_names):
        data[f"{name}_true"] = y_test[:, i]
        data[f"{name}_pred"] = y_pred[:, i]
        data[f"{name}_error"] = y_pred[:, i] - y_test[:, i]

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"Saved test predictions to: {output_csv}")


def plot_regression_results(y_test, y_pred, target_names, output_png):
    n_targets = len(target_names)

    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
    )

    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(target_names):
        ax = axes[i]

        true_vals = y_test[:, i]
        pred_vals = y_pred[:, i]

        ax.scatter(true_vals, pred_vals, alpha=0.55, s=18)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=1.5,
            color="red",
        )

        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        r2 = r2_score(true_vals, pred_vals)

        ax.set_xlabel("True concentration")
        ax.set_ylabel("Predicted concentration")
        ax.set_title(name)

        ax.text(
            0.05,
            0.95,
            f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    for j in range(n_targets, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved regression plot to: {output_png}")


def plot_cv_results(cv_df, output_png):
    plt.figure(figsize=(7, 5))
    plt.plot(
        cv_df["n_components"],
        cv_df["cv_r2"],
        marker="o",
        linewidth=1.5,
    )

    best_idx = cv_df["cv_r2"].idxmax()
    best_n = cv_df.loc[best_idx, "n_components"]
    best_r2 = cv_df.loc[best_idx, "cv_r2"]

    plt.scatter([best_n], [best_r2], s=80)
    plt.xlabel("Number of PLS components")
    plt.ylabel("Cross-validated R²")
    plt.title("PLS component selection")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Saved CV plot to: {output_png}")


def inspect_saved_joblib(model_file):
    """
    Immediate post-save check to verify that the model file really contains
    the required preprocessing metadata.
    """

    bundle = joblib.load(model_file)

    print("\nPost-save joblib inspection:")
    print(f"  file: {model_file}")
    print(f"  type: {type(bundle)}")

    if not isinstance(bundle, dict):
        raise ValueError("Saved joblib file is not a dictionary bundle.")

    required = [
        "model",
        "target_cols",
        "original_ppm_cols",
        "n_bins",
        "void_mask",
        "ppm_after_binning",
        "ppm_kept",
    ]

    for key in required:
        print(f"  contains {key}: {key in bundle}")

    if "void_mask" not in bundle:
        raise ValueError(
            "CRITICAL ERROR: saved model does not contain 'void_mask'."
        )

    void_mask = np.asarray(bundle["void_mask"], dtype=bool)

    print(f"  void_mask length: {len(void_mask)}")
    print(f"  void_mask retained: {void_mask.sum()}")
    print(f"  final feature count: {bundle.get('n_features_final', 'NA')}")


def main():
    parser = argparse.ArgumentParser(
        description="PLS regression for NMR concentration prediction with saved preprocessing mask."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV containing concentration columns and ppm_ spectral columns.",
    )

    parser.add_argument(
        "-m",
        "--model_out",
        default="pls_nmr_model_WITH_MASK.joblib",
        help="Output trained PLS model file.",
    )

    parser.add_argument(
        "-p",
        "--predictions_out",
        default="pls_predictions.csv",
        help="CSV file containing test predictions.",
    )

    parser.add_argument(
        "--plot_out",
        default="pls_regression.png",
        help="Output PNG file for regression plots.",
    )

    parser.add_argument(
        "--cv_plot_out",
        default="pls_cv_components.png",
        help="Output PNG file for PLS component cross-validation plot.",
    )

    parser.add_argument(
        "--cv_out",
        default="pls_cv_components.csv",
        help="Output CSV file with CV scores for PLS components.",
    )

    parser.add_argument(
        "--kept_regions_plot_out",
        default="pls_kept_ppm_regions.png",
        help="PNG showing which ppm points were kept after void removal.",
    )

    parser.add_argument(
        "--kept_regions_csv_out",
        default="pls_kept_ppm_regions.csv",
        help="CSV showing which ppm variables were kept after preprocessing.",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=500,
        help=(
            "Number of ppm bins used to reduce spectral dimensionality. "
            "Set to 0 to disable binning."
        ),
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of PLS components. If omitted, selected by CV.",
    )

    parser.add_argument(
        "--max_components",
        type=int,
        default=30,
        help="Maximum number of PLS components tested by CV.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used as test set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed.",
    )

    parser.add_argument(
        "--void_threshold_fraction",
        type=float,
        default=0.001,
        help=(
            "Remove ppm points whose mean absolute intensity is below this "
            "fraction of the maximum mean absolute intensity."
        ),
    )

    parser.add_argument(
        "--no_void_removal",
        action="store_true",
        help="Disable automatic void-region removal.",
    )

    parser.add_argument(
        "--exclude_ppm",
        nargs="*",
        default=None,
        help=(
            "Manual ppm ranges to exclude, e.g. "
            "--exclude_ppm 4.65:4.90 0.0:0.1"
        ),
    )

    args = parser.parse_args()

    print("\nReading input CSV...")
    df = pd.read_csv(args.input)

    target_cols, ppm_cols = detect_columns(df)

    original_ppm_cols = list(ppm_cols)
    ppm_original = ppm_from_headers(original_ppm_cols)

    X = (
        df[original_ppm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    y = (
        df[target_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    original_n_features = X.shape[1]

    print(f"\nOriginal spectral dimension: {original_n_features}")

    # ------------------------------------------------------------
    # 1. Optional binning
    # ------------------------------------------------------------

    if args.n_bins is not None and args.n_bins > 0 and args.n_bins < X.shape[1]:
        X, ppm_after_binning = bin_spectra(
            X=X,
            ppm=ppm_original,
            n_bins=args.n_bins,
        )

        binning_applied = True

        print(f"Binning applied.")
        print(f"Binned spectral dimension: {X.shape[1]}")

    else:
        ppm_after_binning = ppm_original.copy()
        binning_applied = False

        print("No binning applied.")

    n_features_after_binning = X.shape[1]

    # ------------------------------------------------------------
    # 2. Void-region mask and/or manual exclusions
    # ------------------------------------------------------------

    if not args.no_void_removal:
        void_mask = make_void_mask_from_training(
            X=X,
            threshold_fraction=args.void_threshold_fraction,
        )

        void_mask = apply_manual_exclusions(
            mask=void_mask,
            ppm=ppm_after_binning,
            exclude_ranges=args.exclude_ppm,
        )

        print(
            f"\nVoid-region removal retained {void_mask.sum()} / {len(void_mask)} ppm variables "
            f"({100 * void_mask.sum() / len(void_mask):.2f}%)."
        )

    else:
        print("\nVoid-region removal disabled.")

        void_mask = np.ones(len(ppm_after_binning), dtype=bool)

        if args.exclude_ppm:
            void_mask = apply_manual_exclusions(
                mask=void_mask,
                ppm=ppm_after_binning,
                exclude_ranges=args.exclude_ppm,
            )

            print(
                f"\nManual exclusion retained {void_mask.sum()} / {len(void_mask)} ppm variables "
                f"({100 * void_mask.sum() / len(void_mask):.2f}%)."
            )

    plot_kept_regions(
        ppm=ppm_after_binning,
        mask=void_mask,
        output_png=args.kept_regions_plot_out,
    )

    save_kept_regions_csv(
        ppm_after_binning=ppm_after_binning,
        void_mask=void_mask,
        output_csv=args.kept_regions_csv_out,
    )

    X = X[:, void_mask]
    ppm_kept = ppm_after_binning[void_mask]

    print(f"\nFinal spectral dimension used by PLS: {X.shape[1]}")

    # ------------------------------------------------------------
    # 3. Train/test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # ------------------------------------------------------------
    # 4. Choose PLS components
    # ------------------------------------------------------------

    if args.n_components is None:
        n_components, cv_df = choose_best_n_components(
            X_train=X_train,
            y_train=y_train,
            max_components=args.max_components,
            seed=args.seed,
        )

        cv_df.to_csv(args.cv_out, index=False)
        plot_cv_results(cv_df, args.cv_plot_out)

    else:
        n_components = args.n_components
        cv_df = None
        print(f"\nUsing user-defined n_components = {n_components}")

    max_allowed = min(X_train.shape[0] - 1, X_train.shape[1])

    if n_components > max_allowed:
        raise ValueError(
            f"n_components={n_components} is too large. "
            f"Maximum allowed is {max_allowed}."
        )

    # ------------------------------------------------------------
    # 5. Train final PLS model
    # ------------------------------------------------------------

    model = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components, scale=False)),
        ]
    )

    print("\nTraining final PLS model...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------

    y_pred = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        target_names=target_cols,
    )

    save_predictions(
        y_test=y_test,
        y_pred=y_pred,
        target_names=target_cols,
        output_csv=args.predictions_out,
    )

    plot_regression_results(
        y_test=y_test,
        y_pred=y_pred,
        target_names=target_cols,
        output_png=args.plot_out,
    )

    # ------------------------------------------------------------
    # 7. Save full model bundle WITH preprocessing
    # ------------------------------------------------------------

    model_bundle = {
        # Final trained sklearn model
        "model": model,

        # Concentration target columns
        "target_cols": target_cols,

        # Original spectral information before preprocessing
        "original_ppm_cols": original_ppm_cols,
        "original_ppm": ppm_original,
        "original_n_features": original_n_features,

        # Binning information
        "n_bins": args.n_bins,
        "binning_applied": binning_applied,
        "ppm_after_binning": ppm_after_binning,
        "n_features_after_binning": n_features_after_binning,

        # Critical preprocessing information
        "void_mask": void_mask,
        "ppm_kept": ppm_kept,
        "n_features_final": X.shape[1],
        "void_threshold_fraction": args.void_threshold_fraction,
        "void_removal": not args.no_void_removal,
        "exclude_ppm": args.exclude_ppm,

        # Model / CV information
        "n_components": n_components,
        "cv_scores": cv_df,

        # Reproducibility
        "test_size": args.test_size,
        "seed": args.seed,
    }

    print("\nDEBUG before saving model:")
    print(f"  model_out: {args.model_out}")
    print(f"  void_mask exists: {'void_mask' in model_bundle}")
    print(f"  void_mask length: {len(model_bundle['void_mask'])}")
    print(f"  void_mask retained: {model_bundle['void_mask'].sum()}")
    print(f"  final features: {model_bundle['n_features_final']}")

    joblib.dump(model_bundle, args.model_out)

    print(f"\nSaved model to: {args.model_out}")
    print(f"Saved predictions to: {args.predictions_out}")
    print(f"Saved regression plot to: {args.plot_out}")
    print(f"Saved kept-region plot to: {args.kept_regions_plot_out}")
    print(f"Saved kept-region CSV to: {args.kept_regions_csv_out}")

    inspect_saved_joblib(args.model_out)

    print("\nTraining completed successfully.")
    print("Use this exact model file for prediction:")
    print(f"  {args.model_out}")


if __name__ == "__main__":
    main()
