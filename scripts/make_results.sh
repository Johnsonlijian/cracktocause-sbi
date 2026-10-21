#!/usr/bin/env bash
# Portable analysis route. Full campaigns require external generated runs; see the runbook.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${1:-${CACAIE_RUNS_DIR:-runs/CrackToCause}}"
PY="${PY:-python3}"
N="${CACAIE_NPE_OUT:-results/npe_local}"
mkdir -p results figures "$N"
run() {
    if [[ -n "${CACAIE_LAUNCHER_PY:-}" ]]; then
        "$PY" -u "$CACAIE_LAUNCHER_PY" "$@"
    else
        "$PY" -u "$@"
    fi
}
# The remaining steps mirror the manuscript's local make_results.sh route.
# They are intentionally explicit so a reviewer can inspect every input/output edge.
run examples/campaign_stats.py "$RUNS/sbi_facade" results/campaign_facade.json --facade
run examples/campaign_stats.py "$RUNS/sbi_den" results/campaign_den.json
run scripts/train_npe.py facade "$RUNS/sbi_facade" "$N/facade_final" --epochs "${NPE_EPOCHS:-300}" --ensemble "${NPE_ENSEMBLE:-5}"
run scripts/train_npe.py den "$RUNS/sbi_den" "$N/den_final" --epochs "${NPE_EPOCHS:-300}" --ensemble "${NPE_ENSEMBLE:-5}"
run scripts/make_numbers.py "$N/facade_final" "$N/den_final" results manuscript/numbers.tex
printf '%s\n' 'Portable core route complete; see runbook for optional figures and mesh/paired analyses.'
