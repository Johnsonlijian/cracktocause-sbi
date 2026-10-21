"""Integrity and release-boundary checks for the local reproducibility package."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = ROOT / "SHA256SUMS.txt"
REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "DATASETS_AND_LINKS.csv",
    "REPRODUCIBLE_RUNBOOK.md", "CODE_AVAILABILITY.md", "DATA_AVAILABILITY.md",
    "LICENSE-OUTPUTS.md", "AI_USE.md", "refs.bib", "data_public/README.md",
    "results/reference_audit_20261021.json", "results/mesh_posterior.json",
    "results/mesh_mechanisms.json", "results/numbers.tex",
    "figures/fig_mesh_widths.png", "figures/fig_mech_hinge.png",
]
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".py", ".sh", ".ps1", ".tex", ".cff", ".json"}
PRIVATE_PATTERNS = [
    re.compile(r"[A-Z]:[\\\\/]"),
    re.compile(r"/mnt/[A-Za-z]/"),
    re.compile(r"/home/[^\s/]+"),
    re.compile(r"C:\\Users\\", re.I),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def inventory() -> list[tuple[str, str]]:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.name == HASH_FILE.name:
            continue
        if any(part in {"__pycache__", ".git"} for part in p.parts):
            continue
        rows.append((p.relative_to(ROOT).as_posix(), digest(p)))
    return rows


def write_hashes() -> None:
    lines = [f"{sha}  {rel}" for rel, sha in inventory()]
    HASH_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_hashes(errors: list[str]) -> int:
    if not HASH_FILE.exists():
        errors.append("missing SHA256SUMS.txt")
        return 0
    expected = {}
    for line in HASH_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            sha, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed hash line: {line}")
            continue
        expected[rel] = sha.upper()
    actual = dict(inventory())
    for rel, sha in expected.items():
        if rel not in actual:
            errors.append(f"hash lists missing file: {rel}")
        elif actual[rel] != sha:
            errors.append(f"hash mismatch: {rel}")
    for rel in sorted(set(actual) - set(expected)):
        errors.append(f"unlisted file: {rel}")
    return len(expected)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-hashes", action="store_true")
    args = ap.parse_args()
    errors: list[str] = []
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    errors.extend(f"missing required file: {x}" for x in missing)
    for rel in ("results/mesh_posterior.json", "results/mesh_mechanisms.json"):
        if (ROOT / rel).is_file():
            try:
                json.loads((ROOT / rel).read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid JSON {rel}: {exc}")
    mesh = json.loads((ROOT / "results/mesh_posterior.json").read_text(encoding="utf-8")) if not missing else {}
    if mesh.get("n_train_sims") != 804 or mesh.get("n_pair_sims") != 188 or mesh.get("n_pair_maps") != 4079:
        errors.append("mesh posterior counts are not 804/188/4079")
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.name == HASH_FILE.name or p.name == Path(__file__).name or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible absolute/private path in {p.relative_to(ROOT).as_posix()}: {pattern.pattern}")
                break
    if args.write_hashes:
        write_hashes()
        print(f"wrote {HASH_FILE.name}")
        return 0
    n = check_hashes(errors)
    if errors:
        print("PACKAGE FAIL")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    print(f"PACKAGE PASS: {n} hashed files; JSON and boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



