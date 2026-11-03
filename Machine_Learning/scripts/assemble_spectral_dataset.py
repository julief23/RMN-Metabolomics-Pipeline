#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(".")
SPECTRA_DIR = BASE_DIR / "data" / "normalized_csv"
CONCENTRATION_TABLE = BASE_DIR / "results" / "final_concentration_table.csv"

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "assembled_spectra_with_concentrations.csv"


COMPOUND_FOLDER_TO_CODE = {
    "Alanine": "Ala",
    "Arginine": "Arg",
    "Asparagine": "Asn",
    "Aspartate": "Asp",
    "Glucose": "Glc",
    "Glutamine": "Gln",
    "Glutamate": "Glu",
    "Isoleucine": "Ile",
    "Lactate": "Lac",
    "Leucine": "Leu",
    "Methionine": "Met",
    "Phenylalanine": "Phe",
    "Tyrosine": "Tyr",
    "Valine": "Val",
}


def extract_number_from_file(path: Path) -> int:
    return int(path.stem.strip())


def infer_nmr_type_from_file(path: Path) -> str:
    number = str(extract_number_from_file(path))
    if number.endswith("1"):
        return "standard"
    if number.endswith("0"):
        return "pure_shift"
    return "unknown"


def match_mixture_sample(conc_df: pd.DataFrame, number: int) -> pd.DataFrame:
    pattern = rf"Xi-Calsample-H2O-D2O-{number}$"
    return conc_df[
        conc_df["sample_name"].astype(str).str.contains(pattern, regex=True)
    ]


def match_reference_sample(conc_df: pd.DataFrame, spectrum_path: Path) -> pd.DataFrame:
    file_number = extract_number_from_file(spectrum_path)
    nmr_type = infer_nmr_type_from_file(spectrum_path)

    compound_folder = spectrum_path.parent.name.replace("-Reference", "")
    compound_code = COMPOUND_FOLDER_TO_CODE.get(compound_folder)

    if compound_code is None:
        return pd.DataFrame()

    candidates = [
        file_number,
        file_number - 2,
        (file_number // 10) * 100 + 1,
        (file_number // 10) * 100 - 1,
    ]

    candidates = sorted(set([n for n in candidates if n > 0]))

    mask = False
    for n in candidates:
        pattern = rf"Xi-Reference-{compound_code}-{n}$"
        mask = mask | conc_df["sample_name"].astype(str).str.contains(pattern, regex=True)

    matches = conc_df[
        mask
        & (conc_df["sample_kind"].astype(str).str.lower() == "reference")
        & (conc_df["nmr_type"].astype(str).str.lower() == nmr_type)
        & (pd.to_numeric(conc_df[compound_code], errors="coerce").fillna(0.0) > 0)
    ]

    return matches


def load_spectrum_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "ppm",
        "intensity",
        "tsp_integral",
        "intensity_divided_by_tsp_integral",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    return df


def main():
    if not CONCENTRATION_TABLE.exists():
        raise FileNotFoundError(f"Missing concentration table: {CONCENTRATION_TABLE}")

    conc_df = pd.read_csv(CONCENTRATION_TABLE)

    metadata_cols = ["sample_name", "sample_kind", "nmr_type"]
    concentration_cols = [c for c in conc_df.columns if c not in metadata_cols]

    for c in concentration_cols:
        conc_df[c] = pd.to_numeric(conc_df[c], errors="coerce").fillna(0.0)

    all_rows = []
    unmatched = []

    spectrum_files = sorted(SPECTRA_DIR.glob("mixtures/*.csv"))
    spectrum_files += sorted(SPECTRA_DIR.glob("references/*/*.csv"))

    for spectrum_path in spectrum_files:
        if spectrum_path.name.startswith("._"):
            continue

        spectrum_kind = "mixture" if "mixtures" in spectrum_path.parts else "reference"

        if spectrum_kind == "mixture":
            number = extract_number_from_file(spectrum_path)
            matches = match_mixture_sample(conc_df, number)
        else:
            matches = match_reference_sample(conc_df, spectrum_path)

        if matches.empty:
            unmatched.append(str(spectrum_path))
            continue

        spectrum_df = load_spectrum_csv(spectrum_path)

        for _, concentration_row in matches.iterrows():
            temp = spectrum_df.copy()

            temp.insert(0, "spectrum_file", str(spectrum_path))
            temp.insert(1, "spectrum_name", spectrum_path.stem)
            temp.insert(2, "sample_name", concentration_row["sample_name"])
            temp.insert(3, "sample_kind", concentration_row["sample_kind"])
            temp.insert(4, "nmr_type", concentration_row["nmr_type"])

            for compound in concentration_cols:
                temp[compound] = concentration_row[compound]

            all_rows.append(temp)

    if not all_rows:
        raise RuntimeError("No spectrum could be matched with the concentration table.")

    final_df = pd.concat(all_rows, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"CSV created: {OUTPUT_FILE}")
    print(f"Number of rows: {len(final_df)}")
    print(f"Number of spectra matched: {final_df['sample_name'].nunique()}")

    print("\nMatched standard references:")
    ref = final_df[
        (final_df["sample_kind"].astype(str).str.lower() == "reference")
        & (final_df["nmr_type"].astype(str).str.lower() == "standard")
    ]

    summary = ref.groupby(["spectrum_file", "sample_name"])[concentration_cols].first().reset_index()

    for _, row in summary.iterrows():
        positives = [
            c for c in concentration_cols
            if pd.to_numeric(pd.Series([row[c]]), errors="coerce").fillna(0.0).iloc[0] > 0
        ]
        print(row["spectrum_file"], " -> ", row["sample_name"], positives)

    if unmatched:
        print("\nUnmatched spectra:")
        for item in unmatched:
            print(f"  - {item}")


if __name__ == "__main__":
    main()