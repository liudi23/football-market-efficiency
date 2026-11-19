"""Walk-forward forecasting and market blending.

`walk_forward_dc` refits Dixon-Coles every `refit_days` on matches strictly
before each block, so every prediction uses only information available at
the time. `MarketBlend` learns how much weight the model earns next to the
market: p ∝ exp(a·log p_market + b·log p_model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dixon_coles import DixonColes


def walk_forward_dc(df: pd.DataFrame, start_date, refit_days=7, window_days=730,
                    xi=0.0019, by_league=True) -> pd.DataFrame:
    """Out-of-sample 1X2 and over-2.5 probabilities for matches on/after start_date."""
    df = df.sort_values("date").reset_index(drop=True)
    out = []
    groups = df.groupby("league") if by_league else [("ALL", df)]
    for lg, d in groups:
        model = DixonColes(xi=xi)
        t = pd.Timestamp(start_date)
        end = d.date.max()
        while t <= end:
            t_next = t + pd.Timedelta(days=refit_days)
            test = d[(d.date >= t) & (d.date < t_next)]
            if len(test):
                train = d[(d.date < t) & (d.date >= t - pd.Timedelta(days=window_days))]
                model.fit(train, ref_date=t)
                p = model.predict_proba(test.home, test.away)
                po = model.predict_over(test.home, test.away)
                out.append(pd.DataFrame({"idx": test.index, "dc_h": p[:, 0], "dc_d": p[:, 1],
                                         "dc_a": p[:, 2], "dc_over": po}))
            t = t_next
    pred = pd.concat(out).set_index("idx")
    return df.join(pred, how="inner")


class MarketBlend:
    """Two-weight log-linear pool of market and model probabilities."""

    def fit(self, p_market, p_model, y):
        lm, lo = np.log(np.clip(p_market, 1e-9, 1)), np.log(np.clip(p_model, 1e-9, 1))
        y = np.asarray(y)

        def nll(w):
            z = w[0] * lm + w[1] * lo
            z -= z.max(1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -logp[np.arange(len(y)), y].sum()

        self.w_ = minimize(nll, [1.0, 0.0], method="BFGS").x
        return self

    def predict_proba(self, p_market, p_model):
        z = self.w_[0] * np.log(np.clip(p_market, 1e-9, 1)) + \
            self.w_[1] * np.log(np.clip(p_model, 1e-9, 1))
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
