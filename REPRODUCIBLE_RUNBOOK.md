# Reproducible runbook

This runbook describes what a clean user can reproduce from the public code release and which inputs must be obtained separately. All commands are run from this directory.

## 1. Environment

The lightweight analysis and plotting route uses Python 3.11 or newer with numpy, scipy, scikit-image, scikit-learn, matplotlib, PyMuPDF, torch, and sbi 0.27.0. The phase-field forward solver additionally requires a compatible FEniCSx/dolfinx stack with mpi4py, petsc4py, petsc, and ufl.

Install the public dependencies and run the integrity gate:

    python -m pip install -r requirements-public.txt
    python scripts/verify_package.py

If sbi is supplied by a managed environment, set CACAIE_SITE_PACKAGES to that environment''s site-packages directory before using scripts/run_with_cacaie_env.py.

## 2. External inputs and redistribution boundary

The complete finite-element campaign and NPE retraining need generated simulation records and mesh files. Those inputs are intentionally excluded because they are large local artifacts. Put an authorized campaign tree at a path such as runs/CrackToCause/ or set CACAIE_RUNS_DIR.

The four digitized experimental inputs are not shipped. Their filenames, source references, and local audit hashes are in DATASETS_AND_LINKS.csv. Users must obtain authorized copies from the cited sources before placing them under data_public/.

## 3. Portable checks and included outputs

The included derived JSON/TXT summaries and simulation-only figures are the outputs audited on 2026-10-21. Verify them first:

    python scripts/verify_package.py
    python scripts/verify_refs_online.py
    python -m json.tool results/mesh_posterior.json > NUL
    python -m json.tool results/mesh_mechanisms.json > NUL

The verifier checks the 188-pair posterior summary, required figure set, reference-audit JSON, package hashes, JSON parseability, and obvious local-path leakage. It does not rerun the finite-element campaign.

## 4. Rebuild the result chain

With authorized campaign data and a compatible Python environment:

    set CACAIE_RUNS_DIR=<absolute path to campaign>
    set CACAIE_NPE_OUT=results/npe_local
    bash scripts/make_results.sh

On PowerShell, set the same variables in the environment and invoke Bash or adapt the commands to python. The portable script exposes the core campaign statistics, estimator training, and number generation. The full local manuscript route also includes calibration, mesh, mechanism, posterior predictive, intervention, and figure stages; those stages require the excluded campaign tree and are recorded in the local project runbook.

A fresh 300-epoch, five-member CPU NPE run is computationally expensive. The manuscript result uses validated estimator artifacts for the NPE, conditional, noise-ablation, and value-of-information summaries. This package records that boundary rather than silently treating reuse as retraining.

## 5. Optional finite-element and mesh analyses

Run forward simulations only after installing FEniCSx and reviewing the solver parameters in src/crackcause/fx_pfczm.py and the relevant examples entry point. Run paired mesh analyses with the same simulation index in both mesh trees, keeping train and evaluation indices disjoint as specified in results/mesh_posterior_analysis_manifest_20261018.md. Report early-stopped runs and common-record filtering; do not fill missing records with fabricated values.

## 6. Manuscript and release checks

The active manuscript and submission materials are outside this package by design. The manuscript remains LOCAL_ONLY / NOT_SUBMITTED. The author profile is Lijian Ren, Inner Mongolia University of Technology, renlijian@imut.edu.cn.

DeepSeek was used for code-management and code-checking assistance and for English-language polishing. The author selected the data, made the scientific decisions, and verified the final calculations, references, claims, and files.

The public repository is https://github.com/Johnsonlijian/cracktocause-sbi. No archive DOI is claimed. This package does not claim field validation, a no-opening causal counterfactual, journal acceptance, or public indexing.

