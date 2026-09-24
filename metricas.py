"""Métricas en mm/h. NSE y R indefinidos se representan como None."""

import numpy as np


def extremos_fdc(observed, predicted, epsilon=1e-6):
    """FHV/FLV de curvas ordenadas; ceros regularizados solo para el logaritmo."""
    y = np.sort(np.maximum(np.asarray(observed, dtype=np.float64).ravel(), 0))[::-1]
    p = np.sort(np.maximum(np.asarray(predicted, dtype=np.float64).ravel(), 0))[::-1]
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Series finitas y de igual longitud requeridas")
    high = max(1, int(np.ceil(.02*len(y))))
    low = max(1, int(np.ceil(.30*len(y))))
    volume = y[:high].sum()
    yy = np.log(np.maximum(y[-low:], epsilon))
    pp = np.log(np.maximum(p[-low:], epsilon))
    observed_shape = (yy-yy[-1]).sum()
    simulated_shape = (pp-pp[-1]).sum()
    return {"FHV_pct": float(100*(p[:high].sum()-volume)/volume) if volume > 1e-12 else None,
            "FLV_pct": float(-100*(simulated_shape-observed_shape)/observed_shape) if observed_shape > 1e-12 else None,
            "FLV_epsilon_mm_h": epsilon}


def calcular(observed, predicted):
    y, p = np.asarray(observed, dtype=np.float64).ravel(), np.asarray(predicted, dtype=np.float64).ravel()
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Observaciones y predicciones deben coincidir y ser finitas")
    if not len(y):
        raise ValueError("No hay observaciones")
    error = p-y
    yc, pc = y-y.mean(), p-p.mean()
    vy, vp = float(yc@yc), float(pc@pc)
    return {"RMSE": float(np.sqrt(np.mean(error**2))), "MAE": float(np.mean(np.abs(error))),
            "NSE": float(1 - (error@error)/vy) if vy > 1e-20 else None,
            "R": float((yc@pc)/np.sqrt(vy*vp)) if vy > 1e-20 and vp > 1e-20 else None,
            "bias_mm_h": float(error.mean()), "n_values": len(y)}


def por_cuenca(y, p, basins):
    return [{"basin_id": int(b), **calcular(y[basins == b], p[basins == b])} for b in np.unique(basins)]


def resumen(y, p, basins):
    result = calcular(y, p)
    basin = por_cuenca(y, p, basins)
    valid = [r["NSE"] for r in basin if r["NSE"] is not None]
    result.update(NSE_basin_mean=float(np.mean(valid)) if valid else None,
                  NSE_basin_median=float(np.median(valid)) if valid else None,
                  basins_with_defined_NSE=len(valid), basins_total=len(basin),
                  negative_prediction_fraction=float(np.mean(p < 0)))
    # Diagnóstico condicional, distinto de los índices FHV/FLV de curvas ordenadas.
    for label, threshold, mask in [("high", float(np.quantile(y, .98)), y >= np.quantile(y, .98)),
                                    ("low", float(np.quantile(y, .30)), y <= np.quantile(y, .30))]:
        yy, pp = y[mask], p[mask]
        volume = float(yy.sum())
        result[label] = {"threshold_mm_h": threshold, **calcular(yy, pp),
                         "volume_bias_pct": float(100*(pp.sum()-volume)/volume) if abs(volume) > 1e-12 else None}
    return result, basin
