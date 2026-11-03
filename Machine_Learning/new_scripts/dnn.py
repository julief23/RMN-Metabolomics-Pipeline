#!/usr/bin/env python3

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses


def detect_columns(df):
    ppm_cols = [c for c in df.columns if str(c).startswith("ppm_")]
    target_cols = [c for c in df.columns if c not in ppm_cols]

    if not ppm_cols:
        raise ValueError("No ppm_ spectral columns found.")

    if not target_cols:
        raise ValueError("No concentration target columns found.")

    print(f"\nDetected {len(target_cols)} concentration columns:")
    for c in target_cols:
        print(f"  {c}")

    print(f"\nDetected {len(ppm_cols)} spectral columns.")

    return target_cols, ppm_cols


def ppm_from_headers(ppm_cols):
    ppm = []

    for c in ppm_cols:
        value = str(c).replace("ppm_", "").replace(",", ".")
        ppm.append(float(value) )

    return np.array(ppm)


def make_void_mask_from_training(
    X,
    threshold_fraction=0.001,
    min_keep_points=20,
):
    """
    Remove spectral variables with almost no signal.

    A ppm point is kept if:

        mean(abs(X[:, j])) > threshold_fraction * max(mean(abs(X)))

    This is computed once before training and saved in the joblib bundle.
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
            f"Decrease --void-threshold-fraction."
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


def save_kept_regions_csv(ppm, mask, output_csv):
    table = pd.DataFrame(
        {
            "feature_index": np.arange(len(ppm)),
            "ppm": ppm,
            "kept": mask.astype(int),
        }
    )

    table.to_csv(output_csv, index=False)
    print(f"Saved kept-region CSV to: {output_csv}")


def load_dataset(csv_file):
    df = pd.read_csv(csv_file)

    target_cols, ppm_cols = detect_columns(df)

    X = (
        df[ppm_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    y = (
        df[target_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    ppm = ppm_from_headers(ppm_cols)

    print(f"\nOriginal X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, target_cols, ppm_cols, ppm


def build_dense_nn(
    n_features,
    n_targets,
    learning_rate=1e-3,
    dropout=0.20,
    loss_name="huber",
):
    """
    Dense neural network / MLP for void-filtered NMR spectral variables.

    Unlike the previous CNN, this model does not assume local continuity
    along the spectral axis. It takes the retained ppm variables directly
    as a feature vector.
    """

    inp = layers.Input(shape=(n_features,), name="spectrum_features")

    x = layers.Dense(512, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(32, activation="relu")(x)

    out = layers.Dense(n_targets, activation="linear", name="concentrations")(x)

    model = models.Model(inputs=inp, outputs=out)

    if loss_name.lower() == "huber":
        loss_fn = losses.Huber()
    elif loss_name.lower() == "mae":
        loss_fn = "mae"
    elif loss_name.lower() == "mse":
        loss_fn = "mse"
    else:
        raise ValueError("--loss must be one of: huber, mae, mse")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["mae", "mse"],
    )

    return model


class R2Callback(callbacks.Callback):
    def __init__(self, X_train, y_train_scaled, X_val, y_val_scaled, y_scaler):
        super().__init__()

        self.X_train = X_train
        self.y_train_scaled = y_train_scaled
        self.X_val = X_val
        self.y_val_scaled = y_val_scaled
        self.y_scaler = y_scaler

        self.train_r2 = []
        self.val_r2 = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_train_pred_scaled = self.model.predict(self.X_train, verbose=0)
        y_val_pred_scaled = self.model.predict(self.X_val, verbose=0)

        y_train_true = self.y_scaler.inverse_transform(self.y_train_scaled)
        y_val_true = self.y_scaler.inverse_transform(self.y_val_scaled)

        y_train_pred = self.y_scaler.inverse_transform(y_train_pred_scaled)
        y_val_pred = self.y_scaler.inverse_transform(y_val_pred_scaled)

        train_r2 = r2_score(
            y_train_true,
            y_train_pred,
            multioutput="variance_weighted",
        )

        val_r2 = r2_score(
            y_val_true,
            y_val_pred,
            multioutput="variance_weighted",
        )

        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)

        logs["train_r2"] = train_r2
        logs["val_r2"] = val_r2

        print(
            f" — train_R2: {train_r2:.5f}"
            f" — test_R2: {val_r2:.5f}"
        )


def plot_history(history_df, output_prefix):
    epochs = history_df["epoch"].to_numpy()

    if "train_r2" in history_df.columns and "val_r2" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history_df["train_r2"], marker="o", label="Train R²")
        plt.plot(epochs, history_df["val_r2"], marker="o", label="Test R²")
        plt.xlabel("Epoch")
        plt.ylabel("R²")
        plt.title("Dense NN concentration prediction: R² evolution")
        plt.legend()
        plt.tight_layout()
        out_png = f"{output_prefix}_r2_evolution.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved: {out_png}")

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_df["loss"], marker="o", label="Train loss")
    plt.plot(epochs, history_df["val_loss"], marker="o", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dense NN training loss")
    plt.legend()
    plt.tight_layout()
    out_png = f"{output_prefix}_loss_evolution.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history_df["mae"], marker="o", label="Train MAE")
    plt.plot(epochs, history_df["val_mae"], marker="o", label="Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE, scaled units")
    plt.title("Dense NN MAE evolution")
    plt.legend()
    plt.tight_layout()
    out_png = f"{output_prefix}_mae_evolution.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")


def plot_regression(y_true, y_pred, target_cols, output_png):
    n_targets = len(target_cols)
    ncols = 3
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
    )

    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(target_cols):
        ax = axes[i]

        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        ax.scatter(true_vals, pred_vals, alpha=0.55, s=18)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="red",
            linewidth=1.5,
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
            f"MAE = {mae:.4g}\nRMSE = {rmse:.4g}\nR² = {r2:.4g}",
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

    print(f"Saved: {output_png}")


def save_predictions(y_true, y_pred, target_cols, output_csv):
    data = {}

    for i, name in enumerate(target_cols):
        data[f"{name}_true"] = y_true[:, i]
        data[f"{name}_pred"] = y_pred[:, i]
        data[f"{name}_error"] = y_pred[:, i] - y_true[:, i]

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a Keras dense neural network to predict NMR metabolite concentrations."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV with concentration columns and ppm_ spectral columns.",
    )

    parser.add_argument(
        "--model-out",
        default="dense_nmr_model.keras",
        help="Output Keras model file.",
    )

    parser.add_argument(
        "--joblib-out",
        default="dense_nmr_bundle.joblib",
        help="Output joblib bundle containing scalers, columns and model path.",
    )

    parser.add_argument(
        "--output-prefix",
        default="dense",
        help="Prefix for output plots and CSV files.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Number of epochs without validation-loss improvement before stopping.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-5,
        help="Minimum validation-loss improvement required to reset patience.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction used as test set.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
        help="Dropout rate.",
    )

    parser.add_argument(
        "--loss",
        choices=["huber", "mae", "mse"],
        default="huber",
        help="Training loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed.",
    )

    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Clip negative predictions to zero when saving test predictions.",
    )

    parser.add_argument(
        "--void-threshold-fraction",
        type=float,
        default=0.001,
        help=(
            "Remove ppm variables whose mean absolute intensity is below this "
            "fraction of the maximum mean absolute intensity."
        ),
    )

    parser.add_argument(
        "--no-void-removal",
        action="store_true",
        help="Disable void-region removal.",
    )

    parser.add_argument(
        "--exclude-ppm",
        nargs="*",
        default=None,
        help=(
            "Manual ppm ranges to exclude after void detection, e.g. "
            "--exclude-ppm 4.65:4.90 0.0:0.1"
        ),
    )

    parser.add_argument(
        "--kept-regions-plot-out",
        default=None,
        help="Optional PNG showing retained ppm variables.",
    )

    parser.add_argument(
        "--kept-regions-csv-out",
        default=None,
        help="Optional CSV showing retained ppm variables.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X, y, target_cols, ppm_cols, ppm = load_dataset(args.input)

    original_ppm_cols = list(ppm_cols)
    original_ppm = ppm.copy()
    original_n_features = X.shape[1]

    # ------------------------------------------------------------
    # Void-region removal
    # ------------------------------------------------------------

    if not args.no_void_removal:
        void_mask = make_void_mask_from_training(
            X=X,
            threshold_fraction=args.void_threshold_fraction,
        )

        void_mask = apply_manual_exclusions(
            mask=void_mask,
            ppm=ppm,
            exclude_ranges=args.exclude_ppm,
        )

        print(
            f"\nVoid-region removal retained {void_mask.sum()} / {len(void_mask)} ppm variables "
            f"({100 * void_mask.sum() / len(void_mask):.2f}%)."
        )

    else:
        print("\nVoid-region removal disabled.")
        void_mask = np.ones(X.shape[1], dtype=bool)

        if args.exclude_ppm:
            void_mask = apply_manual_exclusions(
                mask=void_mask,
                ppm=ppm,
                exclude_ranges=args.exclude_ppm,
            )

            print(
                f"\nManual exclusion retained {void_mask.sum()} / {len(void_mask)} ppm variables "
                f"({100 * void_mask.sum() / len(void_mask):.2f}%)."
            )

    ppm_kept = ppm[void_mask]
    ppm_cols_kept = [c for c, keep in zip(ppm_cols, void_mask) if keep]

    X = X[:, void_mask]

    print(f"\nFinal dense NN spectral dimension: {X.shape[1]}")

    if args.kept_regions_plot_out is None:
        args.kept_regions_plot_out = f"{args.output_prefix}_kept_ppm_regions.png"

    if args.kept_regions_csv_out is None:
        args.kept_regions_csv_out = f"{args.output_prefix}_kept_ppm_regions.csv"

    plot_kept_regions(
        ppm=ppm,
        mask=void_mask,
        output_png=args.kept_regions_plot_out,
    )

    save_kept_regions_csv(
        ppm=ppm,
        mask=void_mask,
        output_csv=args.kept_regions_csv_out,
    )

    # ------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Scale X feature-wise after void removal
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Scale y target-wise
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    print("\nDense NN input shapes:")
    print(f"  X_train: {X_train_scaled.shape}")
    print(f"  X_test:  {X_test_scaled.shape}")
    print(f"  y_train: {y_train_scaled.shape}")
    print(f"  y_test:  {y_test_scaled.shape}")

    model = build_dense_nn(
        n_features=X_train_scaled.shape[1],
        n_targets=y_train_scaled.shape[1],
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        loss_name=args.loss,
    )

    print("\nModel summary:")
    model.summary()

    r2_callback = R2Callback(
        X_train=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_val=X_test_scaled,
        y_val_scaled=y_test_scaled,
        y_scaler=y_scaler,
    )

    print("\nTraining dense NN...\n")

    fit_callbacks = [r2_callback]

    if args.early_stopping:
        early_stop_callback = callbacks.EarlyStopping(
            monitor="val_r2",
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        )

        fit_callbacks.append(early_stop_callback)

        print("\nEarly stopping enabled:")
        print(f"  monitor: val_loss")
        print(f"  patience: {args.early_stopping_patience}")
        print(f"  min_delta: {args.early_stopping_min_delta}")
        print("  restore_best_weights: True")

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=fit_callbacks,
        shuffle=True,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    history_df["train_r2"] = r2_callback.train_r2
    history_df["val_r2"] = r2_callback.val_r2

    history_csv = f"{args.output_prefix}_training_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"\nSaved: {history_csv}")

    # Final prediction on test set
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

    if args.clip_negative:
        y_test_pred = np.clip(y_test_pred, 0.0, None)

    global_r2 = r2_score(y_test, y_test_pred, multioutput="variance_weighted")
    global_mae = mean_absolute_error(y_test, y_test_pred)
    global_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\nFinal test-set performance in original concentration units:")
    print(f"  Global MAE  = {global_mae:.6g}")
    print(f"  Global RMSE = {global_rmse:.6g}")
    print(f"  Global R²   = {global_r2:.6g}")

    print("\nPer-metabolite test-set performance:")
    for i, name in enumerate(target_cols):
        mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_test_pred[:, i])

        print(f"{name}")
        print(f"  MAE  = {mae:.6g}")
        print(f"  RMSE = {rmse:.6g}")
        print(f"  R²   = {r2:.6g}")

    plot_history(
        history_df=history_df,
        output_prefix=args.output_prefix,
    )

    plot_regression(
        y_true=y_test,
        y_pred=y_test_pred,
        target_cols=target_cols,
        output_png=f"{args.output_prefix}_regression.png",
    )

    save_predictions(
        y_true=y_test,
        y_pred=y_test_pred,
        target_cols=target_cols,
        output_csv=f"{args.output_prefix}_test_predictions.csv",
    )

    # Save Keras model
    model.save(args.model_out)
    print(f"\nSaved Keras model to: {args.model_out}")

    # Save joblib bundle, including void mask
    bundle = {
        "model_type": "keras_dense_nn",
        "keras_model_path": args.model_out,

        "target_cols": target_cols,

        "original_ppm_cols": original_ppm_cols,
        "original_ppm": original_ppm,
        "original_n_features": original_n_features,

        "ppm_cols_kept": ppm_cols_kept,
        "ppm_kept": ppm_kept,
        "void_mask": void_mask,
        "void_removal": not args.no_void_removal,
        "void_threshold_fraction": args.void_threshold_fraction,
        "exclude_ppm": args.exclude_ppm,
        "n_features_final": X.shape[1],

        "x_scaler": x_scaler,
        "y_scaler": y_scaler,

        "n_targets": y.shape[1],
        "test_size": args.test_size,
        "seed": args.seed,
        "loss": args.loss,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,

        "final_test_global_mae": global_mae,
        "final_test_global_rmse": global_rmse,
        "final_test_global_r2": global_r2,
    }

    joblib.dump(bundle, args.joblib_out)

    print(f"Saved joblib bundle to: {args.joblib_out}")
    print("\nSaved preprocessing information:")
    print(f"  original features: {original_n_features}")
    print(f"  final features:    {X.shape[1]}")
    print(f"  void mask length:  {len(void_mask)}")
    print(f"  retained points:   {void_mask.sum()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
