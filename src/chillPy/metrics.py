"""Model-validation metrics translated from chillR."""

from __future__ import annotations

import numpy as np

from ._base import ensure_1d


def rmsep(predicted: object, observed: object, *, na_rm: bool = False) -> float:
    """Compute root mean squared error of prediction.

    Translates R ``RMSEP``. Missing values raise unless ``na_rm=True``.
    """
    pred = ensure_1d(predicted)
    obs = ensure_1d(observed)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed must have the same shape")
    if not na_rm and (np.isnan(pred).any() or np.isnan(obs).any()):
        raise ValueError(
            "Datasets include missing values. To override this error, set na_rm=True."
        )
    mask = ~(np.isnan(pred) | np.isnan(obs))
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))


def rpd(predicted: object, observed: object, *, na_rm: bool = False) -> float:
    """Compute residual prediction deviation.

    Translates R ``RPD`` as ``sd(observed) / RMSEP``.
    """
    obs = ensure_1d(observed)
    error = rmsep(predicted, observed, na_rm=na_rm)
    if error == 0:
        return float("inf")
    spread = np.nanstd(obs, ddof=1) if na_rm else np.std(obs, ddof=1)
    return float(spread / error)


def rpiq(predicted: object, observed: object, *, na_rm: bool = False) -> float:
    """Compute ratio of performance to interquartile distance.

    Translates R ``RPIQ`` as ``IQR(observed) / RMSEP`` using R's default
    linear quantile interpolation.
    """
    obs = ensure_1d(observed)
    error = rmsep(predicted, observed, na_rm=na_rm)
    if error == 0:
        return float("inf")
    if na_rm:
        q75, q25 = np.nanpercentile(obs, [75, 25])
    else:
        q75, q25 = np.percentile(obs, [75, 25])
    return float((q75 - q25) / error)


RMSEP = rmsep
RPD = rpd
RPIQ = rpiq
