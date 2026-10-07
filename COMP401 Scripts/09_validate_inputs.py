#validate_inputs

import argparse
from pathlib import Path

from src.modelling_common import (
    CHECKPOINTS,
    TARGET,
    configure_logging,
    ensure_dirs,
    load_all,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    ensure_dirs(args.output_root)
    configure_logging(args.output_root)

    rows = []

    for checkpoint in CHECKPOINTS:
        train, validation, testing = load_all(
            args.data_root,
            checkpoint,
        )

        for split_name, df in [
            ("development", train),
            ("validation", validation),
            ("testing", testing),
        ]:
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "split": split_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "at_risk": int(df[TARGET].sum()),
                    "not_at_risk": int((df[TARGET] == 0).sum()),
                    "at_risk_rate_percent": round(
                        df[TARGET].mean() * 100,
                        4,
                    ),
                }
            )

    import pandas as pd

    summary = pd.DataFrame(rows)

    output_file = (
        args.output_root
        / "tables"
        / "input_validation_summary.csv"
    )
    summary.to_csv(output_file, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()