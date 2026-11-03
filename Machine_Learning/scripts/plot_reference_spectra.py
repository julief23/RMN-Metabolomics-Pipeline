"""
plot_reference_spectra.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "../results"
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "spectra")
FILE_STD    = os.path.join(RESULTS_DIR, "reference_standard.csv")
FILE_PS     = os.path.join(RESULTS_DIR, "reference_pure_shift.csv")

STYLE = {
    "standard":   {"color": "#2563EB", "lw": 1.0, "label": "Standard"},
    "pure_shift": {"color": "#DC2626", "lw": 1.0, "label": "Pure Shift"},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_std = pd.read_csv(FILE_STD).copy()
df_ps  = pd.read_csv(FILE_PS).copy()

ppm_cols_std = [c for c in df_std.columns if c.startswith("ppm_")]
ppm_cols_ps  = [c for c in df_ps.columns  if c.startswith("ppm_")]

def ppm_from_cols(cols):
    """Gère ppm_0.0000 (nouveau) et ppm_0000 (ancien, /1000)."""
    vals = [float(c.replace("ppm_", "")) for c in cols]
    arr = np.array(vals)
    if arr.max() > 100:   # ancien format entier
        arr /= 1000.0
    return arr

ppm_axis_std = ppm_from_cols(ppm_cols_std)
ppm_axis_ps  = ppm_from_cols(ppm_cols_ps)

# Plage commune pour xlim
ppm_min = min(ppm_axis_std.min(), ppm_axis_ps.min())
ppm_max = max(ppm_axis_std.max(), ppm_axis_ps.max())

print(f"Axe ppm std : {ppm_axis_std.min():.3f} → {ppm_axis_std.max():.3f} ({len(ppm_cols_std)} pts)")
print(f"Axe ppm ps  : {ppm_axis_ps.min():.3f} → {ppm_axis_ps.max():.3f} ({len(ppm_cols_ps)} pts)\n")

def get_molecule(sample_name: str) -> str:
    parts = sample_name.split("-")
    return parts[2] if len(parts) >= 3 else sample_name

df_std["molecule"] = df_std["sample_name"].apply(get_molecule)
df_ps["molecule"]  = df_ps["sample_name"].apply(get_molecule)

molecules = sorted(df_std["molecule"].unique())
print(f"Molécules : {molecules}\n")

for mol in molecules:
    row_std = df_std[df_std["molecule"] == mol]
    row_ps  = df_ps[df_ps["molecule"]  == mol]

    if row_std.empty or row_ps.empty:
        print(f"  ⚠  Données manquantes pour {mol}, ignoré.")
        continue

    y_std = row_std.iloc[0][ppm_cols_std].values.astype(float)
    y_ps  = row_ps.iloc[0][ppm_cols_ps].values.astype(float)

    y_std_n = y_std / (y_std.max() or 1.0)
    y_ps_n  = y_ps  / (y_ps.max()  or 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    for ax, y, ppm_ax, nmr_type in zip(
        axes,
        [y_std_n, y_ps_n],
        [ppm_axis_std, ppm_axis_ps],
        ["standard", "pure_shift"]
    ):
        s = STYLE[nmr_type]
        ax.plot(ppm_ax, y, color=s["color"], linewidth=s["lw"])
        ax.fill_between(ppm_ax, 0, y, color=s["color"], alpha=0.10)
        ax.set_xlim(ppm_max, ppm_min)   # axe RMN inversé
        ax.set_ylim(-0.02, 1.12)
        ax.set_ylabel(f"{s['label']}\n(norm.)", fontsize=9, labelpad=6)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))   # tick tous les 1 ppm
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("δ (ppm)", fontsize=10)
    fig.suptitle(f"Référence {mol} — Standard vs Pure Shift",
                 fontsize=12, fontweight="bold", y=1.01)

    out_path = os.path.join(OUTPUT_DIR, f"{mol}_standard_vs_pure_shift.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")

print(f"\n✅  {len(molecules)} figures sauvegardées dans '{OUTPUT_DIR}/'")