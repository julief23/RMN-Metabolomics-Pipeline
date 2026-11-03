#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import pandas as pd


def extract_integral(log_file: Path):
    if not log_file.exists():
        return None

    txt = log_file.read_text(errors="ignore")
    m = re.search(r"Integral\s*:\s*([0-9eE+.\-]+)", txt)
    if m:
        return float(m.group(1))
    return None


def to_float_or_none(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def get_compound_block(df: pd.DataFrame, compound_code: str):
    aliases = [compound_code]
    if compound_code == "Ile":
        aliases.extend(["IIe", "Ile", "ILE"])

    for i in range(len(df)):
        val = df.iloc[i, 1] if df.shape[1] > 1 else None
        if isinstance(val, str) and val.strip() in aliases:
            return i

    raise ValueError(
        f"Bloc non trouvé pour le composé {compound_code} dans la feuille Excel. "
        f"Aliases testés: {aliases}"
    )


def find_first_reference_name(row_values, prefix):
    for v in row_values:
        if isinstance(v, str):
            vv = v.strip()
            if vv.startswith(prefix):
                return vv
    return None


def get_compound_rows(df: pd.DataFrame, compound_code: str, concentration_kind: str):
    """
    Retourne deux listes triées par concentration décroissante :
      pure_rows = [(excel_name, concentration), ...]
      standard_rows = [(excel_name, concentration), ...]

    Cela permet ensuite d'associer :
      - logs 10,20,30... -> pure_rows triées décroissantes
      - logs 11,21,31... -> standard_rows triées décroissantes
    """
    start_row = get_compound_block(df, compound_code)
    row_start = start_row
    row_end = min(start_row + 20, len(df))

    if concentration_kind == "sample_preparation":
        pure_conc_col = 3
        standard_conc_col = 17
    elif concentration_kind == "finally":
        pure_conc_col = 7
        # on garde la logique actuelle pour la partie standard
        standard_conc_col = 17
    else:
        raise ValueError("concentration_kind must be 'sample_preparation' or 'finally'")

    prefix = f"Xi-Reference-{compound_code}-"

    pure_rows = []
    standard_rows = []

    for r in range(row_start, row_end):
        row = df.iloc[r, :]

        # Partie gauche = Pure shift
        pure_name = find_first_reference_name(row.iloc[0:13], prefix)
        pure_conc = to_float_or_none(row.iloc[pure_conc_col]) if df.shape[1] > pure_conc_col else None
        if pure_name is not None and pure_conc is not None:
            pure_rows.append((pure_name, pure_conc))

        # Partie droite = Standard
        standard_name = find_first_reference_name(row.iloc[13:], prefix)
        standard_conc = to_float_or_none(row.iloc[standard_conc_col]) if df.shape[1] > standard_conc_col else None
        if standard_name is not None and standard_conc is not None:
            standard_rows.append((standard_name, standard_conc))

    # Tri décroissant par concentration pour recaler avec :
    # Pure shift logs: 10,20,30...
    # Standard logs: 11,21,31...
    pure_rows = sorted(pure_rows, key=lambda x: x[1], reverse=True)
    standard_rows = sorted(standard_rows, key=lambda x: x[1], reverse=True)

    return pure_rows, standard_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compound", required=True, help="Nom du composé, ex: Glucose")
    parser.add_argument("--compound-code", required=True, help="Code Excel, ex: Glc")
    parser.add_argument("--excel", required=True, help="Fichier Excel")
    parser.add_argument("--logs-dir", required=True, help="Dossier de logs du composé")
    parser.add_argument("--output", required=True, help="CSV global de sortie")
    parser.add_argument(
        "--concentration-kind",
        choices=["sample_preparation", "finally"],
        default="sample_preparation",
        help="Type de concentration à utiliser"
    )
    args = parser.parse_args()

    excel_path = Path(args.excel)
    logs_dir = Path(args.logs_dir)
    output_csv = Path(args.output)

    df_excel = pd.read_excel(
        excel_path,
        sheet_name="Reference Library Data Analysis",
        header=None
    )

    pure_rows, standard_rows = get_compound_rows(
        df_excel,
        args.compound_code,
        args.concentration_kind
    )

    print(f"{args.compound}: {len(pure_rows)} références Pure shift trouvées")
    print(f"{args.compound}: {len(standard_rows)} références Standard trouvées")

    main_logs = []
    for log_file in sorted(logs_dir.glob("plot_*.log")):
        m = re.match(r"plot_(\d+)(_0)?$", log_file.stem)
        if not m:
            continue
        sample_num = int(m.group(1))
        is_zero = m.group(2) is not None
        if not is_zero:
            main_logs.append((sample_num, log_file))

    if not main_logs:
        raise ValueError(f"Aucun log principal trouvé dans {logs_dir}")

    pure_logs = sorted([(n, f) for n, f in main_logs if n % 10 == 0], key=lambda x: x[0])
    standard_logs = sorted([(n, f) for n, f in main_logs if n % 10 == 1], key=lambda x: x[0])

    if len(pure_logs) != len(pure_rows):
        print(f"Warning: {args.compound} Pure shift -> {len(pure_logs)} logs mais {len(pure_rows)} lignes Excel")
    if len(standard_logs) != len(standard_rows):
        print(f"Warning: {args.compound} Standard -> {len(standard_logs)} logs mais {len(standard_rows)} lignes Excel")

    rows = []

    for (sample_num, log_file), (excel_name, concentration) in zip(pure_logs, pure_rows):
        log0 = logs_dir / f"plot_{sample_num}_0.log"
        rows.append({
            "compound": args.compound,
            "compound_code": args.compound_code,
            "sample_number": sample_num,
            "excel_name": excel_name,
            "nmr_type": "Pure shift",
            "concentration_used": concentration,
            "integral_log": extract_integral(log_file),
            "integral_log_0": extract_integral(log0),
            "integral_log_0_x2": 2 * extract_integral(log0) if extract_integral(log0) is not None else None,
        })

    for (sample_num, log_file), (excel_name, concentration) in zip(standard_logs, standard_rows):
        log0 = logs_dir / f"plot_{sample_num}_0.log"
        rows.append({
            "compound": args.compound,
            "compound_code": args.compound_code,
            "sample_number": sample_num,
            "excel_name": excel_name,
            "nmr_type": "Standard",
            "concentration_used": concentration,
            "integral_log": extract_integral(log_file),
            "integral_log_0": extract_integral(log0),
            "integral_log_0_x2": 2 * extract_integral(log0) if extract_integral(log0) is not None else None,
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["nmr_type", "sample_number"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    pure_csv = output_csv.with_name(f"{output_csv.stem}_pure_shift.csv")
    standard_csv = output_csv.with_name(f"{output_csv.stem}_standard.csv")

    df_out[df_out["nmr_type"] == "Pure shift"].to_csv(pure_csv, index=False)
    df_out[df_out["nmr_type"] == "Standard"].to_csv(standard_csv, index=False)

    print(f"CSV global créé : {output_csv}")
    print(f"CSV Pure shift créé : {pure_csv}")
    print(f"CSV Standard créé : {standard_csv}")
    print(df_out)


if __name__ == "__main__":
    main()