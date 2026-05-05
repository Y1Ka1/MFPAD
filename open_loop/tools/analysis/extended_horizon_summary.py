import argparse
import csv
import os

import mmcv


def _load_json(path):
    return mmcv.load(path)


def _fmt_two_dec(val):
    return f"{val:.2f}"


def _method_row(name, data):
    return {
        "method": name,
        "avg_l2_7_12": data["avg_l2_7_12"],
        "avg_collision_7_12": data["avg_collision_7_12"],
        "rel_l2_6_to_12_pct": data["rel_l2_6_to_12_pct"],
        "rel_collision_6_to_12_pct": data["rel_collision_6_to_12_pct"],
    }


def _check_degrade(data):
    l2_6 = data["L2"]["6s"]
    l2_12 = data["L2"]["12s"]
    c_6 = data["collision"]["6s"]
    c_12 = data["collision"]["12s"]
    return l2_12 >= l2_6, c_12 >= c_6


def main():
    parser = argparse.ArgumentParser(description="Summarize 12s extended-horizon metrics")
    parser.add_argument("--sparsedrive-json", required=True)
    parser.add_argument("--momad-json", required=True)
    parser.add_argument("--mfpad-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-txt", required=True)
    parser.add_argument("--out-tex", required=True)
    args = parser.parse_args()

    sparse = _load_json(args.sparsedrive_json)
    momad = _load_json(args.momad_json)
    mfpad = _load_json(args.mfpad_json)

    rows = [
        _method_row("SparseDrive", sparse),
        _method_row("MomAD", momad),
        _method_row("MFPAD", mfpad),
    ]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "avg_l2_7_12",
            "avg_collision_7_12",
            "rel_l2_6_to_12_pct",
            "rel_collision_6_to_12_pct",
        ])
        for row in rows:
            writer.writerow([
                row["method"],
                f"{row['avg_l2_7_12']:.6f}",
                f"{row['avg_collision_7_12']:.6f}",
                f"{row['rel_l2_6_to_12_pct']:.6f}",
                f"{row['rel_collision_6_to_12_pct']:.6f}",
            ])

    sparse_l2_ok, sparse_c_ok = _check_degrade(sparse)
    momad_l2_ok, momad_c_ok = _check_degrade(momad)
    mfpad_l2_ok, mfpad_c_ok = _check_degrade(mfpad)

    all_l2_degrade = sparse_l2_ok and momad_l2_ok and mfpad_l2_ok
    all_c_degrade = sparse_c_ok and momad_c_ok and mfpad_c_ok

    mfpad_l2_best = mfpad["rel_l2_6_to_12_pct"] <= min(
        sparse["rel_l2_6_to_12_pct"], momad["rel_l2_6_to_12_pct"]
    )
    mfpad_c_best = mfpad["rel_collision_6_to_12_pct"] <= min(
        sparse["rel_collision_6_to_12_pct"], momad["rel_collision_6_to_12_pct"]
    )

    summary_lines = [
        f"All methods L2 degrade with horizon: {all_l2_degrade}",
        f"All methods collision degrade with horizon: {all_c_degrade}",
        f"MFPAD L2 degrades slowest: {mfpad_l2_best}",
        f"MFPAD collision degrades slowest: {mfpad_c_best}",
    ]

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w") as f:
        for row in rows:
            f.write(
                f"{row['method']} & "
                f"{_fmt_two_dec(row['avg_l2_7_12'])} & "
                f"{_fmt_two_dec(row['avg_collision_7_12'])} & "
                f"{_fmt_two_dec(row['rel_l2_6_to_12_pct'])} & "
                f"{_fmt_two_dec(row['rel_collision_6_to_12_pct'])} \\\\\n"
            )


if __name__ == "__main__":
    main()
