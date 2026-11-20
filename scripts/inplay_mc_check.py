"""In-play Step 2 check: Monte Carlo simulator vs analytic pricer.

Prices a set of match states both ways and reports the differences in units
of Monte Carlo standard error. Writes reports/inplay_mc_check.md.

Usage:  python scripts/inplay_mc_check.py [--sims 200000] [--rho -0.08]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sportsq import inplay

STATES = [(0, 0, 0), (0, 0, 30), (1, 0, 30), (1, 0, 45), (0, 1, 60), (1, 1, 75),
          (2, 0, 45), (0, 0, 85)]


def main(n_sims, rho, lam_h=1.5, lam_a=1.1, seed=2026):
    rows, zs = [], []
    for i, (h, a, m) in enumerate(STATES):
        ex = inplay.price(lam_h, lam_a, h, a, m, rho=rho)
        mc = inplay.mc_price(lam_h, lam_a, h, a, m, n_sims, rho=rho, seed=seed + i)
        z = np.r_[(mc["1x2"] - ex["1x2"][0]) / mc["1x2_se"],
                  (mc["over"][2] - ex["over"][0][2]) / mc["over_se"][2]]
        zs.extend(z[np.isfinite(z)])
        rows.append({
            "minute": m, "score": f"{h}-{a}",
            "home (analytic / MC)": f"{ex['1x2'][0][0]:.4f} / {mc['1x2'][0]:.4f}",
            "draw": f"{ex['1x2'][0][1]:.4f} / {mc['1x2'][1]:.4f}",
            "away": f"{ex['1x2'][0][2]:.4f} / {mc['1x2'][2]:.4f}",
            "over 2.5": f"{ex['over'][0][2]:.4f} / {mc['over'][2]:.4f}",
            "max |z|": round(float(np.nanmax(np.abs(z))), 2),
        })
    zs = np.array(zs)
    lines = [
        "# In-play Step 2: Monte Carlo vs analytic", "",
        f"Pre-match rates: home {lam_h}, away {lam_a}; Dixon–Coles rho = {rho}; "
        f"{n_sims:,} simulated matches per state (fixed seeds).", "",
        "The simulator draws goal times as a point process (thinning) from the current "
        "state; the Dixon–Coles correction is applied by importance weighting. "
        "z = (MC − analytic) / MC standard error.", "",
        pd.DataFrame(rows).to_markdown(index=False), "",
        f"Across all {len(zs)} comparisons: mean z = {zs.mean():.3f}, "
        f"mean z² = {(zs ** 2).mean():.3f} (≈ 1 if the standard errors are honest), "
        f"max |z| = {np.abs(zs).max():.2f}.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/inplay_mc_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--rho", type=float, default=-0.08)
    a = ap.parse_args()
    main(a.sims, a.rho)
