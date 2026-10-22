# Public reproducibility package audit — 2026-10-21

## Result

`python scripts/verify_package.py` passes after the release hash refresh.

- The package contains code, author-generated derived summaries, simulation-only figures, provenance metadata, and a reproducible runbook.
- Raw finite-element campaign trees, trained estimator weights, and digitized derivatives of third-party experimental figures are excluded.
- The required mesh posterior counts remain 804 training simulations, 188 paired simulations, and 4,079 paired maps.
- The package contains no active manuscript source, submission files, reviewer material, credentials, or private absolute paths.
- The public repository is `https://github.com/Johnsonlijian/cracktocause-sbi`.
- Release tag: `v0.1.0-2026-10-21`.

## Verification boundary

The verifier proves file integrity, JSON parseability, required-file presence,
mesh-count consistency, and the recorded public/private boundary. It does not
rerun the finite-element campaign, certify field validity, certify the cited
literature, or grant redistribution rights for third-party material.

The manuscript uses validated prior NPE, conditional, noise-ablation, and
value-of-information artifacts for the reported summaries; this release does
not claim a fresh 300-epoch CPU retraining. The no-opening causal
counterfactual remains unavailable.

## Data and licensing boundary

Source code is MIT-licensed. Author-generated derived summaries and
simulation-only figures are CC BY 4.0 unless otherwise marked. The four
digitized experimental inputs remain withheld with `LICENSE_UNVERIFIED` status
in `DATASETS_AND_LINKS.csv`; their hashes identify local files but do not grant
permission to redistribute them.

## Manuscript boundary

The manuscript is maintained outside this repository and remains
`LOCAL_ONLY / NOT_SUBMITTED`. No journal upload, acceptance, archive DOI, or
public indexing is claimed by this package.
