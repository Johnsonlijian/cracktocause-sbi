# CrackToCause-SBI reproducibility package

This is the public code and derived-output repository for the manuscript **Decoding foundation settlement from crack patterns: mechanisms revealed by simulation-based inference with phase-field fracture**. It contains the implementation, portable analysis scripts, generated simulation-only figures, and non-sensitive derived summaries used to audit the 2026-10-21 manuscript state.

The manuscript itself remains LOCAL_ONLY / NOT_SUBMITTED. This repository is intentionally separate from the submission files. The sole author is Lijian Ren, School of Civil Engineering, Inner Mongolia University of Technology (renlijian@imut.edu.cn).

## What is included

- src/ — NumPy/phase-field model and observation operators.
- examples/ — forward checks, calibration/posterior summaries, mesh and mechanism analyses, and plotting scripts.
- scripts/ — digitization helpers, NPE training entry point, number generation, reference checking, and environment launchers.
- figures/ — generated simulation-only PNG/PDF figures used by the manuscript. Figures that reproduce third-party experimental maps are withheld.
- results/ — derived JSON/TXT summaries and the generated numbers.tex; no raw campaign tree or trained model weights.
- refs.bib — manuscript bibliography snapshot; results/reference_audit_20261021.json records the online metadata checks.
- DATASETS_AND_LINKS.csv — provenance and redistribution boundary for every input and derived output.

## Evidence and reproducibility boundary

The supplied summaries document the completed 188-pair mesh posterior analysis and the downstream post-processing route. The NPE, conditional, noise-ablation and value-of-information summaries were reused from validated local estimator artifacts; this package does not claim a fresh 300-epoch CPU retraining. Full finite-element reruns require FEniCSx/dolfinx, MPI/PETSc, and campaign input data that are not redistributed here. The no-opening causal counterfactual is unavailable and is not reconstructed by this package.

The four digitized experimental files are not shipped because their source-specific redistribution terms were not verified. DATASETS_AND_LINKS.csv records the cited source and the local audit hash. A hash proves identity; it does not grant redistribution permission. Raw finite-element runs, solver logs, and trained estimator weights are also withheld.

## License

Source code is offered under the MIT terms in LICENSE. Author-generated derived summaries and simulation-only figures are offered under the Creative Commons Attribution 4.0 International terms in LICENSE-OUTPUTS.md. Third-party source material and digitized derivatives remain excluded and subject to the source terms recorded in DATASETS_AND_LINKS.csv.

## Quick verification

From this directory, run:

    python scripts/verify_package.py

The verifier parses the included JSON summaries, checks required files, scans for obvious private absolute paths, and compares every file with SHA256SUMS.txt. Recreate the hash file only after reviewing the package contents.

## Reproduction routes

See REPRODUCIBLE_RUNBOOK.md for the portable environment, input acquisition, deterministic post-processing, and the separate FEniCSx campaign route. A successful package verification proves file integrity and internal consistency; it does not prove field validity, publication acceptance, or permission to redistribute third-party inputs. DeepSeek was used for code-management and code-checking assistance and English-language polishing; the author made and verified all scientific decisions.

