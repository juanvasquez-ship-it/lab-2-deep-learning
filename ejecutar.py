"""Experimentos reproducibles. Ejecutar desde terminal: python ejecutar.py --help."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import platform
import random
import time
import sys

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from datos import ROOT, CaudalDataset, guardar_json, preparar, ruta
from modelos import Pronosticador

MODELOS = ["lstm", "am", "cnn", "gnn", "fdmlp", "fdmlp_lineal", "lstm_uni", "mlp", "fdmlp_residual", "mlp_residual"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


def entorno():
    return {"python": platform.python_version(), "system": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_free_total": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None}


def cargador(dataset, config, shuffle=False, generator=None):
    workers = config["num_workers"]
    kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      generator=generator, **kwargs)


def modelo(kind, config, prep, device):
    return Pronosticador(kind, len(prep["selected"]), config["hidden_size"],
                          config["lstm_layers"], feature_config=config.get("feature_modules", {}).get(kind, {})).to(device)


def configuracion_equivalente(a, b, kind):
    # Cambiar la lista de experimentos no cambia un modelo ya entrenado.
    comunes_a = {k: v for k, v in a.items() if k not in ("models", "feature_modules")}
    comunes_b = {k: v for k, v in b.items() if k not in ("models", "feature_modules")}
    return (comunes_a == comunes_b and
            a.get("feature_modules", {}).get(kind, {}) == b.get("feature_modules", {}).get(kind, {}))


def sincronizar(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def inferir(net, loader, device, amp, prep):
    net.eval()
    predictions, observations, identifiers, basins, persistence = [], [], [], [], []
    sincronizar(device)
    start = time.perf_counter()
    for x, y, ids, basin, last in loader:
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            pred = net(x.to(device, non_blocking=True))
        pred = pred.float().cpu().numpy() * prep["y_std"] + prep["y_mean"]
        if not np.isfinite(pred).all():
            raise FloatingPointError("Predicciones no finitas")
        predictions.append(pred)
        observations.append(y.numpy() * prep["y_std"] + prep["y_mean"])
        identifiers.append(ids.numpy())
        basins.append(basin.numpy())
        persistence.append(last.numpy())
    sincronizar(device)
    return {"prediction": np.concatenate(predictions), "observed": np.concatenate(observations),
            "Id": np.concatenate(identifiers), "basin_id": np.concatenate(basins),
            "last_discharge": np.concatenate(persistence)}, time.perf_counter() - start


@torch.no_grad()
def perdida_validacion(net, loader, device, amp):
    net.eval()
    total, count = 0.0, 0
    for x, y, *_ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            prediction = net(x)
            loss = torch.nn.functional.mse_loss(prediction, y, reduction="sum")
        total += float(loss)
        count += y.numel()
    return total / count


def entrenar(kind, config, prep, device):
    if kind == "lstm_uni":
        prep = {**prep, "selected": [11], "input_variant": "discharge_only"}
    folder = ROOT / "resultados" / kind
    folder.mkdir(exist_ok=True)
    if (folder / "finalizado.json").exists():
        previous = json.loads((folder / "configuracion.json").read_text(encoding="utf-8"))
        if not configuracion_equivalente(previous["config"], config, kind) or previous["prep"] != prep:
            raise ValueError("Existen resultados de otra configuración. Preservarlos antes de iniciar un experimento nuevo")
        print(kind, "ya finalizado, se conserva", flush=True)
        return
    semilla(config["seed"])
    generator = torch.Generator()
    train = CaudalDataset(config["train_path"], prep, 0)
    val = CaudalDataset(config["train_path"], prep, 1)
    train_loader = cargador(train, config, True, generator)
    val_loader = cargador(val, config)
    net = modelo(kind, config, prep, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    amp = config["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history, best, stale, first, best_epoch = [], float("inf"), 0, 1, 0
    latest = folder / "ultimo.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        if not configuracion_equivalente(checkpoint["config"], config, kind) or checkpoint["prep"] != prep:
            raise ValueError("La configuración cambió. Conservar la ejecución previa en otra carpeta antes de repetir")
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best, stale = checkpoint["history"], checkpoint["best"], checkpoint["stale"]
        first, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_epoch"]
    guardar_json(folder / "configuracion.json", {"model": kind, "config": config, "prep": prep,
                                                  "parameters": sum(p.numel() for p in net.parameters())})
    with (folder / "entrenamiento.log").open("a", encoding="utf-8", buffering=1) as log:
        for epoch in range(first, config["max_epochs"] + 1):
            if stale >= config["patience"]:
                break
            generator.manual_seed(config["seed"] + epoch)
            net.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sincronizar(device)
            start = time.perf_counter()
            total, count, skipped = 0.0, 0, 0
            for step, (x, y, *_rest) in enumerate(train_loader, 1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                    prediction = net(x)
                    loss = torch.nn.functional.mse_loss(prediction, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Pérdida no finita, {kind}, época {epoch}, paso {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous_scale)
                total += float(loss.detach()) * y.numel()
                count += y.numel()
                if step % 500 == 0:
                    progress = {"model": kind, "epoch": epoch, "step": step,
                                "steps_per_epoch": len(train_loader), "train_mse_normalized": total/count,
                                "elapsed_seconds": time.perf_counter()-start}
                    guardar_json(folder / "progreso.json", progress)
                    print(f"{kind} época {epoch}: {step}/{len(train_loader)} pasos", flush=True)
            sincronizar(device)
            train_seconds = time.perf_counter() - start
            val_start = time.perf_counter()
            validation = perdida_validacion(net, val_loader, device, amp)
            if not np.isfinite(validation):
                raise FloatingPointError("Pérdida de validación no finita")
            if validation < best:
                best, stale, best_epoch = validation, 0, epoch
                torch.save({"state_dict": net.state_dict(), "config": config, "prep": prep,
                            "kind": kind, "epoch": epoch}, folder / "mejor.pt")
            else:
                stale += 1
            row = {"epoch": epoch, "train_mse_normalized": total/count,
                   "validation_mse_normalized": validation,
                   "validation_rmse_mm_h": float(np.sqrt(validation)*prep["y_std"]),
                   "train_seconds": train_seconds, "validation_seconds": time.perf_counter()-val_start,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0,
                   "skipped_amp_steps": skipped, "best_epoch": best_epoch}
            history.append(row)
            with (folder / "historial.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            temporary = folder / "ultimo.tmp"
            torch.save({"state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "best": best, "stale": stale,
                        "epoch": epoch, "best_epoch": best_epoch, "config": config, "prep": prep}, temporary)
            temporary.replace(latest)
            message = (f"{kind} época {epoch}: train={total/count:.6f}, val={validation:.6f}, "
                       f"RMSE={row['validation_rmse_mm_h']:.6f} mm/h, {train_seconds:.1f}s, paciencia={stale}")
            print(message, flush=True)
            log.write(message + "\n")
    guardar_json(folder / "finalizado.json", {"epochs": len(history), "best_epoch": best_epoch,
                   "best_validation_mse": best, "stop_reason": "early_stopping" if stale >= config["patience"] else "max_epochs",
                   "total_training_seconds": sum(r["train_seconds"] for r in history)})
    train.close()
    val.close()


def evaluar(kind, device, test=False):
    folder = ROOT / "resultados" / kind
    checkpoint = torch.load(folder / "mejor.pt", map_location=device)
    config, prep = checkpoint["config"], checkpoint["prep"]
    net = modelo(kind, config, prep, device)
    net.load_state_dict(checkpoint["state_dict"])
    dataset = CaudalDataset(config["test_path"] if test else config["train_path"], prep, None if test else 1)
    data, seconds = inferir(net, cargador(dataset, config), device,
                           config["mixed_precision"] and device.type == "cuda", prep)
    if test:
        if not np.array_equal(data["Id"], np.arange(len(dataset))):
            raise AssertionError("El orden de las muestras de test cambió")
        # Restricción física aplicada solo a la entrega, conservando predicciones crudas.
        np.save(folder / "test_predicciones_crudas.npy", data["prediction"])
        path = folder / "predicciones_test.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Id"] + [f"q_{i:02d}" for i in range(1, 49)])
            for i, pred in zip(data["Id"], np.maximum(data["prediction"], 0)):
                writer.writerow([int(i)] + [f"{float(x):.9g}" for x in pred])
        guardar_json(folder / "test_generacion.json", {"rows": len(dataset), "horizon": 48,
                       "units": "mm/h", "seconds": seconds,
                       "negative_raw_values_clipped": int((data["prediction"] < 0).sum()),
                       "test_targets_used": False})
    else:
        # Las observaciones de evaluación se recuperan exactamente, sin redondeo
        # de una ida y vuelta por el escalado usado por la función de pérdida.
        with h5py.File(ruta(config["train_path"]), "r") as source:
            data["observed"] = source["y"][:][data["Id"]]
        np.savez_compressed(folder / "validacion.npz", **data)
        guardar_json(folder / "inferencia.json", {"seconds": seconds, "rows": len(dataset),
                       "parameters": sum(p.numel() for p in net.parameters()), "best_epoch": checkpoint["epoch"]})
    dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=["preparar", "entrenar", "evaluar", "predecir", "todo"])
    parser.add_argument("--modelos", nargs="+", choices=MODELOS)
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semilla(config["seed"])
    guardar_json(ROOT / "resultados/entorno.json", entorno())
    if args.accion in ("preparar", "todo"):
        prep = preparar(config)
    else:
        prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    for kind in args.modelos or config["models"]:
        if args.accion in ("entrenar", "todo"):
            entrenar(kind, config, prep, device)
        if args.accion in ("evaluar", "todo"):
            evaluar(kind, device)
        if args.accion in ("predecir", "todo"):
            evaluar(kind, device, test=True)


if __name__ == "__main__":
    main()
