"""Run the public synthetic-data workflow from a source checkout."""

from pathlib import Path

from scm_concrete_ml.cli import run_demo

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = run_demo(
        ROOT / "data" / "synthetic_example.csv",
        ROOT / "artifacts" / "demo",
    )
    print(result)
