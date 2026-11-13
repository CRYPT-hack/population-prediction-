#!/usr/bin/env python3
"""Run the whole analysis in one process (libraries are imported once) and report timings.

    python run_all.py               # closed model, open model, census-data model + all 15 figures
    python run_all.py --no-figures  # numbers only, about 1 s
"""
import argparse
import time
from pathlib import Path

import census_model
import open_system
import project_code

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--outdir", default=str(HERE))
    a = ap.parse_args()
    total = time.perf_counter()
    for name, mod in (("closed model (project_code)", project_code), ("open model (open_system)", open_system),
                      ("census-data model (census_model)", census_model)):   # order matters: census_model reads open_system.json
        t = time.perf_counter()
        R = mod.run(make_figures=not a.no_figures, outdir=a.outdir, quiet=True)
        print(f"{name:<34} {len(R['assertions_passed']):>3} checks passed, "
              f"{len(R['design_warnings'])} warnings   {time.perf_counter() - t:5.2f} s")
    print(f"{'total':<34} {'':>33}{time.perf_counter() - total:5.2f} s")
