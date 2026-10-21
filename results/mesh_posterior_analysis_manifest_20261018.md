# Fine-mesh posterior analysis contract — 2026-10-18

- **Question:** Does an estimator trained on the production mesh transfer to
  crack maps generated on the h=5 mm mesh, and do paired posterior summaries
  remain stable after the available fine-mesh completion?
- **Scientific unit:** one paired simulation index; map records within one
  simulation are repeated observations and remain in the same split.
- **Estimands:** 90% interval coverage, contraction for cracked maps, paired
  posterior-median differences, and paired crack-map Chamfer distance.
- **Inputs:** `<campaign-root>/sbi_facade` (production mesh),
  `<campaign-root>/sbi_facade_h5` (available indices 0--187),
  `examples/mesh_posterior.py`, and `src/crackcause/npe.py`.
- **Split rule:** train only on production-mesh simulations with index >=188;
  evaluate paired indices <188 on both meshes. No simulation index may occur
  in both training and paired evaluation.
- **Missing-data rule:** keep only records present at the same settlement on
  both meshes; retain early-stopped fine-mesh runs as partial common-record
  pairs and report their status.
- **Uncertainty:** five independently seeded posterior members; per-map 1,000
  posterior draws; 5th--95th percentile intervals.
- **Output:** `results/mesh_posterior.json`, followed by regenerated
  `manuscript/numbers.tex`, Section 5.7, and the reproducibility log.
- **Evidence boundary:** this is a simulation transfer/sensitivity result. It
  does not establish field validity or a causal no-opening counterfactual.

