#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
import numpy as np


EXCEL_PATH = Path("2_Analytical Chemistry Data.xlsx")
if not EXCEL_PATH.exists():
    EXCEL_PATH = Path("data/2_Analytical Chemistry Data.xlsx")

OUTPUT_PATH = Path("results/final_concentration_table.csv")

MIXTURE_SHEET = "Model Mixture Data Analysis"
REFERENCE_SHEET = "Reference Library Data Analysis"

EXCLUDED_REFERENCE_COMPOUNDS = set()


def clean_value(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
        x = re.sub(r"[^\d\.\-eE]", "", x)
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_compound_name(name):
    if name is None:
        return None
    name = str(name).strip()
    mapping = {
        "IIe": "Ile",
        "ILE": "Ile",
        "ile": "Ile",
    }
    return mapping.get(name, name)


def extract_compound_from_reference_name(sample_name):
    match = re.search(r"Xi-Reference-([A-Za-z]+)-?\d*", sample_name)
    if not match:
        return None
    return normalize_compound_name(match.group(1))


def extract_mixtures(mix_df):
    rows = []
    compounds = set()

    for start_row in range(len(mix_df)):
        marker_std = clean_value(mix_df.iloc[start_row, 0])
        marker_ps = clean_value(mix_df.iloc[start_row, 9])

        if marker_std != "Our laboratory compounds":
            continue

        data_rows = []
        r = start_row + 1

        while r < len(mix_df):
            compound = clean_value(mix_df.iloc[r, 0])
            if compound is None:
                break
            if compound.lower() in {"nmr data name", "data name", "tsp"}:
                break
            compound = normalize_compound_name(compound)
            data_rows.append(r)
            compounds.add(compound)
            r += 1

        data_name_row = None
        for rr in range(r, min(r + 10, len(mix_df))):
            value = clean_value(mix_df.iloc[rr, 0])
            if value is not None and value.lower() in {"nmr data name", "data name"}:
                data_name_row = rr
                break

        if data_name_row is None:
            continue

        for col in [2, 3, 4]:
            sample_name = clean_value(mix_df.iloc[data_name_row, col])
            if sample_name is None or not sample_name.startswith("Xi-Calsample"):
                continue
            row = {
                "sample_name": sample_name,
                "sample_kind": "mixture",
                "nmr_type": "standard",
            }
            for rr in data_rows:
                compound = normalize_compound_name(clean_value(mix_df.iloc[rr, 0]))
                row[compound] = to_float(mix_df.iloc[rr, col])
            rows.append(row)

        for col in [11, 12, 13]:
            sample_name = clean_value(mix_df.iloc[data_name_row, col])
            if sample_name is None or not sample_name.startswith("Xi-Calsample"):
                continue
            row = {
                "sample_name": sample_name,
                "sample_kind": "mixture",
                "nmr_type": "pure_shift",
            }
            for rr in data_rows:
                compound = normalize_compound_name(clean_value(mix_df.iloc[rr, 9]))
                row[compound] = to_float(mix_df.iloc[rr, col])
            rows.append(row)

    return rows, compounds


def extract_references(ref_df, all_compounds):
    rows = []

    for r in range(len(ref_df)):
        for c in range(len(ref_df.columns)):
            value = clean_value(ref_df.iloc[r, c])
            if value is None:
                continue
            if not value.startswith("Xi-Reference-"):
                continue
            if not re.search(r"Xi-Reference-[A-Za-z]+-\d+", value):
                continue

            sample_name = value
            compound = extract_compound_from_reference_name(sample_name)

            if compound is None:
                continue
            if compound.lower() in EXCLUDED_REFERENCE_COMPOUNDS:
                continue

            if c <= 8:
                nmr_type = "pure_shift"
                possible_conc_cols = [7, 3]
                concentration = np.nan
                for conc_col in possible_conc_cols:
                    if conc_col < len(ref_df.columns):
                        concentration = to_float(ref_df.iloc[r, conc_col])
                        if not np.isnan(concentration):
                            break
            else:
                nmr_type = "standard"
                possible_conc_cols = [17, 18, c + 4, c + 5]
                concentration = np.nan
                for conc_col in possible_conc_cols:
                    if conc_col < len(ref_df.columns):
                        concentration = to_float(ref_df.iloc[r, conc_col])
                        if not np.isnan(concentration):
                            break

            if np.isnan(concentration):
                continue

            row = {
                "sample_name": sample_name,
                "sample_kind": "reference",
                "nmr_type": nmr_type,
            }
            for comp in all_compounds:
                row[comp] = 0.0
            if compound not in row:
                row[compound] = 0.0
            row[compound] = concentration
            rows.append(row)

    return rows


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            "Excel file not found. Put it either in the current folder "
            "or in data/2_Analytical Chemistry Data.xlsx"
        )

    mix_df = pd.read_excel(EXCEL_PATH, sheet_name=MIXTURE_SHEET, header=None)
    ref_df = pd.read_excel(EXCEL_PATH, sheet_name=REFERENCE_SHEET, header=None)

    mixture_rows, mixture_compounds = extract_mixtures(mix_df)
    all_compounds = set(mixture_compounds)

    for r in range(len(ref_df)):
        for c in range(len(ref_df.columns)):
            value = clean_value(ref_df.iloc[r, c])
            if value and value.startswith("Xi-Reference-"):
                compound = extract_compound_from_reference_name(value)
                if compound is not None:
                    all_compounds.add(compound)

    all_compounds = sorted(all_compounds)
    reference_rows = extract_references(ref_df, all_compounds)

    final_rows = []

    for row in mixture_rows:
        clean_row = {
            "sample_name": row["sample_name"],
            "sample_kind": row["sample_kind"],
            "nmr_type": row["nmr_type"],
        }
        for compound in all_compounds:
            clean_row[compound] = to_float(row.get(compound, 0.0))
        final_rows.append(clean_row)

    for row in reference_rows:
        clean_row = {
            "sample_name": row["sample_name"],
            "sample_kind": row["sample_kind"],
            "nmr_type": row["nmr_type"],
        }
        for compound in all_compounds:
            clean_row[compound] = to_float(row.get(compound, 0.0))
        final_rows.append(clean_row)

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(
        by=["sample_kind", "nmr_type", "sample_name"],
        ascending=[True, True, True],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"CSV created: {OUTPUT_PATH}")
    print(f"Number of rows: {len(final_df)}")
    print(f"Number of compounds: {len(all_compounds)}")
    print("All reference compounds included.")


if __name__ == "__main__":
    main()