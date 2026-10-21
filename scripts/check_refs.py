"""Check the bibliography against CrossRef: title, year, journal, volume, pages and DOI of every cited entry.

usage: python check_refs.py manuscript/refs.bib manuscript/sections OUT.txt
Entries with a DOI are looked up directly; the others by a bibliographic query (best title match). The
report lists, per entry, the fields that differ from CrossRef and the DOI CrossRef suggests. No personal
data is sent (public API without a contact address).
"""
import difflib
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

bib_path, sec_dir, out = sys.argv[1:4]
text = open(bib_path, encoding='utf8').read()
cited = set()
for f in glob.glob(os.path.join(sec_dir, '*.tex')):
    for m in re.finditer(r'\\cite[a-z]*\{([^}]*)\}', open(f, encoding='utf8').read()):
        cited.update(k.strip() for k in m.group(1).split(','))


def parse(bib):
    entries = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, flags=re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}', body):
            fields[fm.group(1).lower()] = re.sub(r'\s+', ' ', fm.group(2)).strip()
        entries[key] = dict(kind=kind, **fields)
    return entries


def plain(s):
    s = re.sub(r'\\[a-zA-Z]+\s*', '', s or '')
    s = re.sub(r'[{}\\"\'`^~]', '', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'check_refs/1.0 (bibliography check)'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode('utf8'))


def crossref(e):
    if e.get('doi'):
        try:
            return get('https://api.crossref.org/works/' + urllib.parse.quote(e['doi'])).get('message'), 'doi'
        except Exception as exc:                     # noqa: BLE001
            return None, f'doi lookup failed: {exc}'
    first = plain(e.get('author', '').split(' and ')[0].split(',')[0])
    q = urllib.parse.quote(f"{plain(e.get('title'))} {first} {e.get('year', '')}")
    try:
        items = get(f'https://api.crossref.org/works?query.bibliographic={q}&rows=5').get('message', {}).get('items', [])
    except Exception as exc:                         # noqa: BLE001
        return None, f'search failed: {exc}'
    if not items:
        return None, 'no match'
    best = max(items, key=lambda it: difflib.SequenceMatcher(None, plain(e.get('title')), plain((it.get('title') or [''])[0])).ratio())
    return best, 'search'


entries = parse(text)
lines = []
for key in sorted(cited):
    e = entries.get(key)
    if e is None:
        lines.append(f'{key}: CITED BUT MISSING IN refs.bib')
        continue
    msg, how = crossref(e)
    time.sleep(1.0)
    if msg is None:
        lines.append(f'{key} [{e["kind"]}]: {how}')
        continue
    t_cr = (msg.get('title') or [''])[0]
    sim = difflib.SequenceMatcher(None, plain(e.get('title')), plain(t_cr)).ratio()
    issues = []
    if sim < 0.9:
        issues.append(f'title {sim:.2f} vs "{t_cr[:90]}"')
    y_cr = (msg.get('issued', {}).get('date-parts') or [[None]])[0][0]
    if y_cr and e.get('year') and str(y_cr) != e['year']:
        issues.append(f'year {e["year"]} vs {y_cr}')
    for fld, crk in (('volume', 'volume'), ('pages', 'page'), ('number', 'issue')):
        v_cr = msg.get(crk)
        if v_cr and e.get(fld) and plain(e[fld]).replace('--', '-') != plain(v_cr).replace('--', '-'):
            issues.append(f'{fld} {e[fld]} vs {v_cr}')
    j_cr = (msg.get('container-title') or [''])[0]
    j_bib = e.get('journal') or e.get('booktitle') or ''
    if j_cr and j_bib and difflib.SequenceMatcher(None, plain(j_bib), plain(j_cr)).ratio() < 0.8:
        issues.append(f'venue "{j_bib[:50]}" vs "{j_cr[:60]}"')
    a_cr = (msg.get('author') or [{}])[0].get('family', '')
    a_bib = plain(e.get('author', '').split(' and ')[0].split(',')[0])
    if a_cr and a_bib and plain(a_cr) != a_bib:
        issues.append(f'first author {a_bib} vs {plain(a_cr)}')
    doi_cr = msg.get('DOI', '')
    if not e.get('doi') and doi_cr and sim >= 0.9:
        issues.append(f'add doi {doi_cr}')
    status = 'OK' if not issues else '; '.join(issues)
    lines.append(f'{key} [{e["kind"]}, {how}]: {status}')
    print(lines[-1], flush=True)
open(out, 'w', encoding='utf8').write('\n'.join(lines) + '\n')
print(f'{len(cited)} cited entries checked -> {out}')
