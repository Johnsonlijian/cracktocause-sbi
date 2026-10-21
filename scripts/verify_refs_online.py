"""Online bibliography metadata audit for the portable CrackToCause-SBI package.

The audit is deliberately conservative: an exact DOI hit is a metadata match
only, and a title-search hit without a DOI is recorded as a candidate that still
needs a source-specific check. It does not infer permission to redistribute
figures or digitized data from a bibliographic record.
"""

from __future__ import annotations

import difflib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "refs.bib"
OUT = ROOT / "results" / "reference_audit_20261021.json"
UA = "CrackToCause-SBI-reference-audit/1.0"
MANUAL_SOURCES = {
    "Burland1974": "https://www.icevirtuallibrary.com/action/doSearch?AllField=Settlement+of+buildings+and+associated+damage",
    "Burland1977": "https://www.icevirtuallibrary.com/action/doSearch?AllField=Behaviour+of+foundations+and+structures",
    "Burland1995": "https://www.icevirtuallibrary.com/action/doSearch?AllField=Assessment+of+risk+of+damage+to+buildings+due+to+tunnelling",
    "Papamakarios2016": "https://papers.nips.cc/paper_files/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html",
    "Greenberg2019": "https://proceedings.mlr.press/v97/greenberg19a.html",
    "Durkan2019": "https://papers.nips.cc/paper_files/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html",
    "Hermans2022": "https://openreview.net/search?term=A%20crisis%20in%20simulation-based%20inference",
    "Baratta2023": "https://zenodo.org/records/10447666",
    "NooruMohamed1992": "https://repository.tudelft.nl/search?query=Nooru-Mohamed",
    "Lueckmann2021": "https://proceedings.mlr.press/v130/lueckmann21a.html",
    "Lourenco1996": "https://repository.tudelft.nl/search?query=Computational%20strategies%20for%20masonry%20structures",
    "Hetenyi1946": "https://search.lib.umich.edu/catalog?query=Beams+on+Elastic+Foundation+Hetenyi",
    "Romano2019": "https://papers.nips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html",
}


def strip_latex(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    value = value.replace("--", "-")
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_bibtex(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    pos = 0
    while True:
        start = text.find("@", pos)
        if start < 0:
            break
        open_brace = text.find("{", start)
        if open_brace < 0:
            break
        comma = text.find(",", open_brace)
        if comma < 0:
            break
        key = text[open_brace + 1 : comma].strip()
        i = comma + 1
        fields: dict[str, str] = {"key": key}
        while i < len(text):
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            if i >= len(text) or text[i] == "}":
                i += 1
                break
            m = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", text[i:])
            if not m:
                i += 1
                continue
            name = m.group(1).lower()
            i += m.end()
            if i < len(text) and text[i] == "{":
                depth, j = 1, i + 1
                while j < len(text) and depth:
                    depth += text[j] == "{"
                    depth -= text[j] == "}"
                    j += 1
                value = text[i + 1 : j - 1]
                i = j
            elif i < len(text) and text[i] == '"':
                j = i + 1
                while j < len(text) and (text[j] != '"' or text[j - 1] == "\\"):
                    j += 1
                value = text[i + 1 : j]
                i = min(j + 1, len(text))
            else:
                j = i
                while j < len(text) and text[j] not in ",}\n":
                    j += 1
                value = text[i:j]
                i = j
            fields[name] = value.strip()
        entries.append(fields)
        pos = max(i, start + 1)
    return entries


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def crossref_exact(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    return get_json(url)["message"]


def crossref_search(query: str) -> list[dict]:
    params = urllib.parse.urlencode({"query.bibliographic": query, "rows": "3"})
    return get_json("https://api.crossref.org/works?" + params)["message"]["items"]


def arxiv_record(identifier: str) -> dict:
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(identifier)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as response:
        root = ET.fromstring(response.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        raise LookupError("arXiv record not found")
    return {
        "title": (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
        "published": (entry.findtext("a:published", default="", namespaces=ns) or "")[:10],
        "authors": [x.text or "" for x in entry.findall("a:author/a:name", ns)],
        "source": "arXiv",
    }


def first_author(entry: dict[str, str]) -> str:
    return strip_latex(entry.get("author", "").split(" and ")[0])


def year(entry: dict[str, str]) -> int | None:
    m = re.search(r"\d{4}", entry.get("year", ""))
    return int(m.group()) if m else None


def compare(entry: dict[str, str], message: dict) -> dict:
    title = strip_latex(entry.get("title", ""))
    remote_title = strip_latex((message.get("title") or [""])[0])
    similarity = difflib.SequenceMatcher(None, title, remote_title).ratio()
    container = strip_latex(entry.get("journal") or entry.get("booktitle", ""))
    remote_container = strip_latex((message.get("container-title") or [""])[0])
    issues = []
    if similarity < 0.80:
        issues.append("title_similarity_below_0.80")
    if container and remote_container and difflib.SequenceMatcher(None, container, remote_container).ratio() < 0.70:
        issues.append("container_title_differs")
    y = year(entry)
    remote_years = []
    for field in ("published-print", "published-online", "issued"):
        parts = (message.get(field) or {}).get("date-parts", [])
        if parts and parts[0]:
            remote_years.append(parts[0][0])
    if y and remote_years and y not in remote_years:
        issues.append("year_differs_from_crossref")
    relation = message.get("relation", {})
    updates = message.get("update-to", [])
    return {
        "title_similarity": round(similarity, 4),
        "remote_title": (message.get("title") or [""])[0],
        "remote_container": (message.get("container-title") or [""])[0],
        "remote_years": remote_years,
        "update_to": updates,
        "relation_keys": sorted(relation.keys()),
        "issues": issues,
        "metadata_status": "METADATA_OK" if not issues else "METADATA_REVIEW",
    }


def main() -> int:
    entries = parse_bibtex(BIB.read_text(encoding="utf-8"))
    rows = []
    for entry in entries:
        key = entry["key"]
        doi = entry.get("doi", "").strip()
        arxiv = re.search(r"arXiv\s*:\s*([0-9.]+)", entry.get("howpublished", ""), flags=re.I)
        row = {
            "key": key,
            "title": entry.get("title", ""),
            "doi": doi or None,
            "year_in_bib": year(entry),
            "source_checked": None,
            "metadata_status": "UNRESOLVED",
            "issues": [],
        }
        try:
            if doi:
                message = crossref_exact(doi)
                row.update(compare(entry, message))
                row["source_checked"] = "Crossref exact DOI"
                row["official_source_url"] = "https://doi.org/" + doi
            elif arxiv:
                record = arxiv_record(arxiv.group(1))
                sim = difflib.SequenceMatcher(None, strip_latex(entry.get("title", "")), strip_latex(record["title"])).ratio()
                row.update({
                    "source_checked": "arXiv API",
                    "remote_title": record["title"],
                    "remote_years": [int(record["published"][:4])] if record["published"][:4].isdigit() else [],
                    "title_similarity": round(sim, 4),
                    "metadata_status": "METADATA_OK" if sim >= 0.80 else "METADATA_REVIEW",
                    "issues": [] if sim >= 0.80 else ["title_similarity_below_0.80"],
                })
                row["official_source_url"] = "https://arxiv.org/abs/" + arxiv.group(1)
            else:
                query = " ".join([entry.get("title", ""), first_author(entry), entry.get("year", "")])
                candidates = crossref_search(query)
                if candidates:
                    candidate = max(candidates, key=lambda item: difflib.SequenceMatcher(None, strip_latex(entry.get("title", "")), strip_latex((item.get("title") or [""])[0])).ratio())
                    checked = compare(entry, candidate)
                    sim = checked["title_similarity"]
                    row.update(checked)
                    row["source_checked"] = "Crossref title search"
                    row["metadata_status"] = "CANDIDATE_NO_DOI" if sim >= 0.80 else "UNRESOLVED"
                    if sim < 0.80:
                        row["issues"] = ["no_reliable_title_match"]
                else:
                    row["source_checked"] = "Crossref title search"
                    row["issues"] = ["no_crossref_candidate"]
        except Exception as exc:  # network/source gaps remain explicit
            row["source_checked"] = row["source_checked"] or "online lookup"
            row["issues"] = ["lookup_failed:" + type(exc).__name__]
        if key in MANUAL_SOURCES:
            row["official_source_url"] = MANUAL_SOURCES[key]
            if row["metadata_status"] == "UNRESOLVED":
                row["metadata_status"] = "MANUAL_SOURCE_RECORDED"
                row["source_checked"] = "Official source registry"
                row["issues"] = ["automated metadata match unavailable; inspect source page before submission"]
        rows.append(row)
        time.sleep(0.15)
    summary = {
        "audit_date": "2026-10-21",
        "bibliography": str(BIB.name),
        "n_entries": len(rows),
        "counts": {
            status: sum(row["metadata_status"] == status for row in rows)
            for status in ("METADATA_OK", "METADATA_REVIEW", "CANDIDATE_NO_DOI", "MANUAL_SOURCE_RECORDED", "UNRESOLVED")
        },
        "scope": "Bibliographic metadata only. This audit does not certify claim support, retraction status, or redistribution permission.",
        "entries": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "n_entries": len(rows), "counts": summary["counts"]}, ensure_ascii=False))
    return 0 if summary["counts"]["UNRESOLVED"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
