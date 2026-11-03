#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-standard", required=True)
    parser.add_argument("--output-pure-shift", required=True)
    args = parser.parse_args()

    all_rows = []

    for file in args.input_files:
        path = Path(file)

        if not path.exists():
            print(f"Warning: fichier absent ignoré : {path}")
            continue

        df = pd.read_csv(path)

        if "nmr_type" not in df.columns:
            print(f"Warning: colonne nmr_type absente dans {path}")
            continue

        all_rows.append(df)

    if all_rows:
        merged = pd.concat(all_rows, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=[
            "compound",
            "nmr_type",
            "title",
            "slope",
            "intercept",
            "r2",
            "rmse",
            "equation",
            "n_points",
            "status",
        ])

    merged["nmr_type_clean"] = (
        merged["nmr_type"]
        .astype(str)
        .str.lower()
        .str.replace("rmn", "", regex=False)
        .str.strip()
    )

    standard_df = merged[
        merged["nmr_type_clean"].str.contains("standard", regex=False, na=False)
    ].copy()

    pure_shift_df = merged[
        merged["nmr_type_clean"].str.contains("pure shift", regex=False, na=False)
    ].copy()

    standard_df = standard_df.drop(columns=["nmr_type_clean"], errors="ignore")
    pure_shift_df = pure_shift_df.drop(columns=["nmr_type_clean"], errors="ignore")

    output_standard = Path(args.output_standard)
    output_pure_shift = Path(args.output_pure_shift)

    output_standard.parent.mkdir(parents=True, exist_ok=True)
    output_pure_shift.parent.mkdir(parents=True, exist_ok=True)

    standard_df.to_csv(output_standard, index=False)
    pure_shift_df.to_csv(output_pure_shift, index=False)

    print(f"CSV Standard créé : {output_standard}")
    print(f"CSV Pure shift créé : {output_pure_shift}")


if __name__ == "__main__":
    main()