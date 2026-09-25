"""Comparación ampliada: robustez, coste comparable y sensibilidad frecuencial."""

import argparse
import json
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset

from datos import ROOT, CaudalDataset, guardar_json
from ejecutar import (cargador, entrenar, entorno, evaluar, inferir, modelo,
                      semilla, sincronizar)
from metricas import calcular, por_cuenca

PRINCIPALES = ["lstm", "am", "cnn", "gnn", "fdmlp"]


def perturbar(x, kind, level, seed, identifier, channels):
    """La misma realización aleatoria se aplica a todos los modelos y niveles."""
    out = x.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(identifier), int(kind == "faltantes")]))
    if kind == "ruido":
        noise = rng.standard_normal(x.shape).astype(np.float32)
        out[:, channels] += level*noise[:, channels]
    elif kind == "faltantes":
        missing = rng.random(x.shape) < level
        hours = np.arange(len(x))
        for channel in channels:
            known = ~missing[:, channel]
            # Interpolación exclusiva dentro de la historia, extremos por vecino.
            out[:, channel] = np.interp(hours, hours[known], x[known, channel]) if known.any() else 0
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(out)


class DatosPerturbados(Dataset):
    def __init__(self, base, kind, level, seed, channels):
        self.base, self.kind, self.level = base, kind, level
        self.seed, self.channels = seed, channels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y, identifier, basin, last = self.base[index]
        return perturbar(x, self.kind, self.level, self.seed, identifier, self.channels), y, identifier, basin, last


def cargar(kind, device):
    checkpoint = torch.load(ROOT / "resultados" / kind / "mejor.pt", map_location=device)
    net = modelo(kind, checkpoint["config"], checkpoint["prep"], device)
    net.load_state_dict(checkpoint["state_dict"])
    return net, checkpoint["prep"]


def robustez(config, device):
    out = ROOT / "resultados/robustez"
    out.mkdir(exist_ok=True)
    base_prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    coefficients = np.abs(base_prep["lasso"]["coefficients"])
    auxiliary = sorted((c for c in base_prep["selected"] if c != 11), key=lambda c: coefficients[c], reverse=True)[:3]
    guardar_json(out / "protocolo.json", {"models": PRINCIPALES, "samples": 18142,
                 "noise_std_normalized": [.1, .5, 1.0], "missing_rates": [.05, .1, .2],
                 "auxiliary_channels": auxiliary, "auxiliary_selection": "top 3 absolute training LASSO coefficients, excluding discharge",
                 "same_random_realizations_across_models_and_levels": True,
                 "imputation": "linear interpolation within history; nearest observed at edges; training mean if all missing",
                 "retraining_after_corruption": False, "seed": config["seed"]})
    for model_kind in PRINCIPALES:
        net, prep = cargar(model_kind, device)
        clean = dict(np.load(ROOT / "resultados" / model_kind / "validacion.npz"))
        base = CaudalDataset(config["train_path"], prep, 1)
        for scope, channels in [("todas", list(range(len(prep["selected"])))),
                                 ("auxiliares", [prep["selected"].index(c) for c in auxiliary])]:
            for kind, levels in [("ruido", [.1, .5, 1.0]), ("faltantes", [.05, .1, .2])]:
                for level in levels:
                    target = out / f"{model_kind}_{scope}_{kind}_{level:g}.json"
                    if target.exists():
                        continue
                    dataset = DatosPerturbados(base, kind, level, config["seed"], channels)
                    data, seconds = inferir(net, cargador(dataset, config), device,
                                           config["mixed_precision"] and device.type == "cuda", prep)
                    assert np.array_equal(data["Id"], clean["Id"])
                    rows = []
                    for horizon in (12, 24, 48):
                        original = calcular(clean["observed"][:, :horizon], clean["prediction"][:, :horizon])
                        metrics = calcular(clean["observed"][:, :horizon], data["prediction"][:, :horizon])
                        rows.append({"hours": horizon, **metrics,
                                     "delta_NSE": metrics["NSE"]-original["NSE"],
                                     "relative_delta_NSE_pct": 100*(metrics["NSE"]-original["NSE"])/abs(original["NSE"]) if original["NSE"] else None,
                                     "relative_delta_RMSE_pct": 100*(metrics["RMSE"]-original["RMSE"])/original["RMSE"]})
                    guardar_json(target, {"model": model_kind, "scope": scope, "perturbation": kind,
                                 "level": level, "seconds": seconds, "metrics": rows,
                                 "by_basin_48h": por_cuenca(clean["observed"], data["prediction"], clean["basin_id"])})
        base.close()
        print(f"Robustez completada: {model_kind}", flush=True)


def eficiencia(config, device):
    target = ROOT / "resultados/benchmark.json"
    if target.exists():
        return
    prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    ds = CaudalDataset(config["train_path"], prep, 0)
    ids = np.random.default_rng(config["seed"]).choice(len(ds), config["batch_size"], replace=False)
    batch = [ds[int(i)] for i in ids]
    x = torch.from_numpy(np.stack([v[0] for v in batch])).to(device)
    y = torch.from_numpy(np.stack([v[1] for v in batch])).to(device)
    ds.close()
    rows = []
    amp = config["mixed_precision"] and device.type == "cuda"
    for kind in config["models"]:
        semilla(config["seed"])
        net, pp = cargar(kind, device)
        xx = x[..., [prep["selected"].index(c) for c in pp["selected"]]]
        net.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            for _ in range(5):
                net(xx)
            samples = []
            for _ in range(20):
                sincronizar(device)
                start = time.perf_counter()
                net(xx)
                sincronizar(device)
                samples.append(time.perf_counter()-start)
        optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        net.train()
        training_times = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(25):
            sincronizar(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.mse_loss(net(xx), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            sincronizar(device)
            if step >= 5:
                training_times.append(time.perf_counter()-start)
        rows.append({"model": kind, "batch_size": len(x), "repeats": 20, "warmup": 5,
                     "inference_median_ms": 1000*float(np.median(samples)),
                     "training_median_ms": 1000*float(np.median(training_times)),
                     "inference_p90_ms": 1000*float(np.quantile(samples, .9)),
                     "training_p90_ms": 1000*float(np.quantile(training_times, .9)),
                     "parameters": sum(p.numel() for p in net.parameters()),
                     "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None})
        # Los pasos de medición modifican solo esta copia, jamás el checkpoint.
        del net, optimizer, scaler
    guardar_json(target, {"environment": entorno(), "includes_io": False,
                 "same_batch_ids": [int(v[2]) for v in batch], "results": rows,
                 "checkpoint_weights_modified": False})
    print("Benchmark completado", flush=True)


def interpretabilidad(config, device):
    target = ROOT / "resultados/interpretabilidad.json"
    if target.exists():
        return
    net, prep = cargar("fdmlp", device)
    ds = CaudalDataset(config["train_path"], prep, 1)
    # cuDNN requiere modo train para derivar una LSTM. No hay dropout ni
    # normalización con estado, y no se realiza ninguna actualización de pesos.
    net.train()
    net.caracteristicas.capturar_espectro = True
    total = np.zeros(len(prep["selected"])//2+1, dtype=np.float64)
    count = 0
    for x, y, *_ in cargador(ds, config):
        x = x.to(device).requires_grad_(True)
        y = y.to(device)
        net.zero_grad(set_to_none=True)
        # FP32 y pérdida por muestra, sin factor 1/B que altere el último lote.
        prediction = net(x)
        loss = torch.nn.functional.mse_loss(prediction, y, reduction="sum")/y.shape[1]
        loss.backward()
        grad = net.caracteristicas.ultimo_espectro.grad
        assert grad is not None and torch.isfinite(grad).all()
        total += grad.abs().sum(dim=(0, 1)).detach().cpu().numpy()
        count += x.shape[0]*x.shape[1]
    average = total/count
    ds.close()
    guardar_json(target, {"model": "fdmlp", "samples": len(ds), "feature_channels": prep["selected"],
                 "fft_axis": "variables", "precision": "float32", "loss": "per-sample mean squared normalized discharge error",
                 "mean_absolute_complex_gradient": average.tolist(),
                 "relative_importance": (average/average.sum()).tolist(),
                 "interpretation_limit": "feature frequencies depend on channel order, not temporal cycles or causal hydrological processes"})
    print("Interpretabilidad completada", flush=True)


def verificar(config):
    from hashlib import sha256
    import csv
    result = {}
    reference = None
    for kind in config["models"]:
        folder = ROOT / "resultados" / kind
        data = dict(np.load(folder / "validacion.npz"))
        if reference is not None:
            for key in ("Id", "observed", "basin_id"):
                assert np.array_equal(data[key], reference[key])
        reference = data
        metrics = calcular(data["observed"], data["prediction"])
        recorded = json.loads((folder / "metricas.json").read_text(encoding="utf-8"))["raw"]
        assert abs(metrics["RMSE"]-recorded["RMSE"]) < 1e-12
        checkpoint = torch.load(folder / "mejor.pt", map_location="cpu")
        with (folder / "historial.csv").open(encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        best = min(history, key=lambda r: float(r["validation_mse_normalized"]))
        assert checkpoint["epoch"] == int(best["epoch"])
        assert abs(metrics["RMSE"]-float(best["validation_rmse_mm_h"])) < 1e-6
        with (folder / "predicciones_test.csv").open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            assert next(reader) == ["Id"]+[f"q_{h:02d}" for h in range(1, 49)]
            rows = np.asarray(list(reader), dtype=np.float64)
        assert rows.shape == (27983, 49) and np.isfinite(rows).all()
        assert np.array_equal(rows[:, 0], np.arange(27983)) and (rows[:, 1:] >= 0).all()
        np.testing.assert_allclose(rows[:, 1:], np.maximum(np.load(folder / "test_predicciones_crudas.npy"), 0), rtol=1e-8, atol=1e-10)
        result[kind] = {"RMSE": metrics["RMSE"], "best_epoch": checkpoint["epoch"], "passed": True}
    cases = list((ROOT / "resultados/robustez").glob("*.json"))
    assert len(cases) == 61
    assert all(np.isfinite(v) for v in json.loads((ROOT / "resultados/interpretabilidad.json").read_text())["mean_absolute_complex_gradient"])
    files = list(ROOT.glob("*.py"))+[ROOT / "config.json", ROOT / "requirements.txt"]
    guardar_json(ROOT / "resultados/verificacion_final.json", {"passed": True, "models": result,
                 "robustness_cases": 60, "test_targets_used": False,
                 "same_validation_rows_and_targets": True, "test_export_checked": True,
                 "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=["todo", "evaluar", "robustez", "eficiencia", "interpretar", "informe", "verificar"], default="todo", nargs="?")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semilla(config["seed"])
    status = ROOT / "resultados/estado_estudio.json"
    try:
        if args.accion == "todo":
            for kind in config["models"]:
                guardar_json(status, {"status": "running", "stage": "training", "model": kind})
                entrenar(kind, config, prep, device)
        if args.accion in ("todo", "evaluar"):
            for kind in config["models"]:
                folder = ROOT / "resultados" / kind
                guardar_json(status, {"status": "running", "stage": "evaluation", "model": kind})
                if not (folder / "validacion.npz").exists():
                    evaluar(kind, device)
                if not (folder / "predicciones_test.csv").exists():
                    evaluar(kind, device, test=True)
        for action, label, function in [("eficiencia", "benchmark", eficiencia),
                                         ("robustez", "robustness", robustez),
                                         ("interpretar", "interpretability", interpretabilidad)]:
            if args.accion in ("todo", action):
                guardar_json(status, {"status": "running", "stage": label})
                function(config, device)
        if args.accion in ("todo", "informe"):
            import informe
            informe.main()
        if args.accion in ("todo", "verificar"):
            verificar(config)
        guardar_json(status, {"status": "complete", "action": args.accion})
    except Exception as exc:
        guardar_json(status, {"status": "failed", "action": args.accion, "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
