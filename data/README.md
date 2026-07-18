# Data policy

This repository does **not** redistribute the 1,456-record research dataset.
The study table was assembled from multiple third-party sources, and the
redistribution terms have not yet been documented source by source. Publishing
the merged CSV without that audit would create an avoidable licensing risk.

`synthetic_example.csv` is generated entirely by
`scripts/generate_synthetic_data.py`. It exists only for software tests,
examples, and UI demonstrations. It is not experimental evidence and must not
be used for scientific or engineering conclusions.

## Bring your own data

Provide one row per mixture with these dosage columns in consistent units:

| Column | Meaning |
|---|---|
| `Cement` | Portland cement dosage |
| `Water` | Mixing water dosage |
| `Coarse aggregate` | Coarse aggregate dosage |
| `Fine aggregate` | Fine aggregate dosage |
| `FA` | Fly ash dosage |
| `SF` | Silica fume dosage |
| `GGBFS` | Ground granulated blast-furnace slag dosage |
| `SP` | Superplasticizer dosage |
| `Cylinder compressive strength` | Target strength in MPa |

Before combining public datasets, retain the original URL, citation, license,
specimen geometry, curing age, test standard, and any unit conversion record.
Do not assume that a publicly downloadable file permits redistribution.
