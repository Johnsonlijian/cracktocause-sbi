# Historical reproducibility package audit — 2026-10-18

> **Superseded.** This file records the pre-publication local candidate. The
> current public-release audit is `PACKAGE_AUDIT_20261021.md`; the GitHub
> publication and remote-commit verification are recorded in the parent
> project's `results/github_publish_20261021.md`.

## Result

`python scripts/verify_package.py` passed after the final hash refresh.

- 131 files are hashed in `SHA256SUMS.txt`.
- JSON summaries parse successfully.
- Mesh posterior counts remain 804 training simulations, 188 paired simulations, and 4,079 paired maps.
- Required metadata, runbook, derived summaries, and figure files are present.
- The package text scan found no embedded Windows drive paths, WSL mount paths, home-directory paths, credentials, or author identity fields.

Package inventory at audit time: 25 source files, 44 example scripts, 10 portable scripts, 30 generated figures, and 21 derived result files; total size 11242209 bytes.

## Included classification

- **Code:** `src/`, `examples/`, and portable `scripts/` needed for forward checks, summaries, plotting, number generation, digitization, and reference checking.
- **Derived outputs:** selected JSON/TXT summaries, generated `numbers.tex`, and generated PNG/PDF figures.
- **Metadata:** `README.md`, `LICENSE`, `CITATION.cff`, `DATASETS_AND_LINKS.csv`, `DATA_AVAILABILITY.md`, `CODE_AVAILABILITY.md`, `REPRODUCIBLE_RUNBOOK.md`, and `SHA256SUMS.txt`.
- **Bibliography snapshot:** `refs.bib`, retained for audit and to be refreshed against authoritative metadata before submission.

## Excluded classification

- Raw third-party source figures and the four digitized experimental files: omitted because redistribution terms were not verified; names, sources, and hashes are listed in `DATASETS_AND_LINKS.csv`.
- Raw finite-element campaign trees, mesh files, NPE weights, and local run logs: omitted because they are large environment-specific artifacts and may expose private paths.
- Active manuscript source, PDF, cover letter, reviewer responses, internal rounds/logs, author/affiliation data, credentials, and submission metadata: omitted by design.

## Reproduction limits

The package can validate the audited summaries and regenerate lightweight analyses once authorized campaign inputs are supplied. A full FEniCSx campaign requires the stated dolfinx/MPI/PETSc environment and is not reproduced by this hash check. The current manuscript's NPE/conditional/noise/VOI summaries use validated prior estimator artifacts; this package does not claim a fresh 300-epoch CPU retraining. The no-opening causal counterfactual remains unavailable.

## Historical release gate

The release gate above applied before the 2026-10-21 public repository push. It
is retained for provenance and is not the current publication status.

