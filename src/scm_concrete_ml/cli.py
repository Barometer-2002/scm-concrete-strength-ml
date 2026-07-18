"""Command-line entry points for the public example workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import generate_synthetic_mixtures, load_dataset
from .evaluation import regression_metrics
from .features import TARGET_COLUMN
from .models import benchmark_models, get_model
from .uncertainty import KNNResidualScaleConformalRegressor, interval_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_demo(data_path: Path, output_dir: Path, *, random_state: int = 42) -> dict:
    """Run a small point-prediction and uncertainty workflow."""

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
    )

    model = get_model("rf", random_state=random_state)
    model.fit(X_train, y_train)
    point_prediction = model.predict(X_test)

    conformal = KNNResidualScaleConformalRegressor(
        estimator=get_model("rf", random_state=random_state),
        alpha=0.1,
        k_neighbors=min(15, max(2, len(X_train) // 8)),
        random_state=random_state,
    ).fit(X_train, y_train)
    prediction, lower, upper = conformal.predict_interval(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "observed": y_test.to_numpy(),
            "prediction": prediction,
            "lower_90": lower,
            "upper_90": upper,
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "data": str(data_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "point_metrics": regression_metrics(y_test, point_prediction),
        "interval_metrics": interval_metrics(y_test, lower, upper),
        "note": "Metrics come from synthetic example data, not the research dataset.",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scm-concrete-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="create a synthetic CSV")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=240)
    generate.add_argument("--seed", type=int, default=42)

    demo = subparsers.add_parser("demo", help="run an end-to-end example")
    demo.add_argument("--data", type=Path, required=True)
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark", help="compare model families")
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--models", nargs="+", default=["rf", "svr", "mlp"])
    benchmark.add_argument("--splits", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=2)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.csv"))
    benchmark.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-data":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_mixtures(args.samples, random_state=args.seed).to_csv(
            args.output,
            index=False,
        )
        print(f"Synthetic data written to {args.output}")
        return 0

    if args.command == "demo":
        summary = run_demo(args.data, args.output, random_state=args.seed)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    X, y = load_dataset(args.data, target_column=TARGET_COLUMN)
    result = benchmark_models(
        X,
        y,
        model_names=tuple(args.models),
        n_splits=args.splits,
        n_repeats=args.repeats,
        random_state=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
