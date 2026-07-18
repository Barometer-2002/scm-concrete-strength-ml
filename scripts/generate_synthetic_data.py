"""Regenerate the license-safe example dataset."""

from pathlib import Path

from scm_concrete_ml.data import generate_synthetic_mixtures


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "synthetic_example.csv"
    generate_synthetic_mixtures(n_samples=240, random_state=2026).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
