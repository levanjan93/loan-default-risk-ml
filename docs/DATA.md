# Data notice

All datasets in this repository are **synthetic, course-provided samples** for education and portfolio demonstration. They are **not** real customer records and must not be used for production credit decisions.

| Field types in raw CSVs | Handling |
|-------------------------|----------|
| Names, SSN-like identifiers | Present only in source files; **dropped from model features** via `PII_COLUMNS` in gold-layer code |
| Customer IDs | Synthetic keys (e.g. `CUS_0x…`) |

If you fork this repo, do not substitute real PII without appropriate governance, consent, and security controls.
