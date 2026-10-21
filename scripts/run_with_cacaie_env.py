"""Run a project script with an optional external scientific Python site-packages path.

Set CACAIE_SITE_PACKAGES when sbi/nflows are supplied by a managed environment.
No machine-specific path is embedded in this release candidate.
"""
from __future__ import annotations
import os
import runpy
import sys

site = os.environ.get("CACAIE_SITE_PACKAGES")
if site and os.path.isdir(site) and site not in sys.path:
    sys.path.insert(0, site)
if len(sys.argv) < 2:
    raise SystemExit("usage: python run_with_cacaie_env.py SCRIPT [ARGS ...]")
script, *args = sys.argv[1:]
sys.argv = [script, *args]
runpy.run_path(script, run_name="__main__")
