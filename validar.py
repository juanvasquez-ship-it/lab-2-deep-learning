"""Pruebas matemáticas, integridad de particiones y piloto de entrenamiento GPU."""

import json
import time
import argparse

import h5py
import numpy as np
import torch

from datos import ROOT, CaudalDataset, guardar_json, ruta
from ejecutar import semilla, entorno, modelo, sincronizar
from metricas import calcular
from modelos import CapaCompleja, FDMLP, AtencionVariables, ConvolucionVariables, GrafoVariables


def ampliacion():
    from metricas import extremos_fdc
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semilla(cfg["seed"])
    ds = CaudalDataset(cfg["train_path"], prep, 0)
    ids = np.random.default_rng(cfg["seed"]).choice(len(ds), cfg["batch_size"], replace=False)
    items = [ds[int(i)] for i in ids]
    x = torch.from_numpy(np.stack([v[0] for v in items])).to(device)
    y = torch.from_numpy(np.stack([v[1] for v in items])).to(device)
    trials = []
    for kind in ("am", "cnn", "gnn", "lstm_uni"):
        semilla(cfg["seed"])
        pp = {**prep, "selected": [11]} if kind == "lstm_uni" else prep
        xx = x[..., -1:] if kind == "lstm_uni" else x
        net = modelo(kind, cfg, pp, device)
        module = net.caracteristicas
        if kind == "am":
            weights = module.pesos(xx[:2])
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
            assert (weights >= 0).all()
        if kind == "gnn":
            a = module.adyacencia()
            torch.testing.assert_close(a, a.T)
            assert (a > 0).all()
            assert float(torch.linalg.eigvalsh(a).abs().max()) < 1.0001
        if kind in ("cnn", "gnn"):
            permutation = torch.randperm(xx.shape[1], device=device)
            torch.testing.assert_close(module(xx[:2, permutation]), module(xx[:2])[:, permutation], atol=1e-5, rtol=1e-5)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
        amp = device.type == "cuda" and cfg["mixed_precision"]
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        net.eval()
        with torch.no_grad():
            fp32 = net(xx[:8])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                fp16 = net(xx[:8])
        amp_error = float((fp32-fp16.float()).abs().max())
        assert amp_error < .02
        net.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(net(xx[:8]), y[:8]).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        if kind != "lstm_uni":
            assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
        original = net.salida.weight.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = net(xx)
                assert prediction.shape == y.shape
                loss = torch.nn.functional.mse_loss(prediction, y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss))
        assert not torch.equal(original, net.salida.weight)
        trials.append({"model": kind, "parameters": sum(p.numel() for p in net.parameters()),
                       "pilot_steps": 5, "losses": losses, "amp_max_abs_error": amp_error,
                       "peak_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None})
    perfect = extremos_fdc(np.arange(1, 101), np.arange(1, 101))
    assert perfect["FHV_pct"] == 0 and perfect["FLV_pct"] == 0
    assert extremos_fdc(np.zeros(100), np.zeros(100))["FLV_pct"] is None
    assert abs(extremos_fdc(np.arange(1, 101), 2*np.arange(1, 101))["FHV_pct"]-100) < 1e-10
    ds.close()
    # Confirma que añadir comparadores no cambió los pesos ni la inferencia previa.
    val = CaudalDataset(cfg["train_path"], prep, 1)
    vx = torch.from_numpy(np.stack([val[i][0] for i in range(64)])).to(device)
    preserved = []
    for kind in ("lstm", "fdmlp", "fdmlp_lineal"):
        if not (ROOT / "resultados" / kind / "mejor.pt").exists() or not (ROOT / "resultados" / kind / "validacion.npz").exists():
            continue
        ckpt = torch.load(ROOT / "resultados" / kind / "mejor.pt", map_location=device)
        net = modelo(kind, ckpt["config"], ckpt["prep"], device).eval()
        net.load_state_dict(ckpt["state_dict"])
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = net(vx)
        actual = pred.float().cpu().numpy()*prep["y_std"]+prep["y_mean"]
        saved = np.load(ROOT / "resultados" / kind / "validacion.npz")["prediction"][:64]
        np.testing.assert_allclose(actual, saved, atol=1e-6, rtol=1e-5)
        preserved.append(kind)
    val.close()
    guardar_json(ROOT / "resultados/pruebas_ampliacion.json", {"passed": True, "pilot": trials,
                  "original_checkpoints_verified": preserved, "FHV_FLV_known_cases": True,
                  "pilot_weights_used_in_final_training": False})
    print("AMPLIACIÓN VALIDADA", json.dumps(trials), flush=True)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((ROOT / "resultados/preprocesamiento.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semilla(config["seed"])
    checks = []
    m = calcular([1, 2, 3], [1, 2, 3])
    assert m["NSE"] == 1 and m["RMSE"] == 0 and abs(m["R"]-1) < 1e-12
    assert calcular([2, 2, 2], [1, 2, 3])["NSE"] is None
    assert abs(calcular([1, 2, 3], [2, 2, 2])["NSE"]) < 1e-12
    checks.append("Métricas: predicción perfecta, media y varianza nula")

    layer = CapaCompleja(7).to(device)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_()
    a, b = torch.randn(3, 7, device=device), torch.randn(3, 7, device=device)
    real, imag = layer(a, b)
    expected = torch.complex(a, b)*torch.complex(layer.peso_real, layer.peso_imag) + torch.complex(layer.sesgo_real, layer.sesgo_imag)
    torch.testing.assert_close(torch.complex(real, imag), expected)
    checks.append("Producto complejo contrastado con aritmética compleja nativa")

    for n in (5, 12):
        x = torch.randn(2, 17, n, device=device, requires_grad=True)
        identity = FDMLP(n, nonlinear=False).to(device)
        torch.testing.assert_close(identity(x), x, atol=1e-5, rtol=1e-5)
        module = FDMLP(n).to(device)
        permutation = torch.randperm(17, device=device)
        torch.testing.assert_close(module(x[:, permutation]), module(x)[:, permutation])
        module(x).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    checks.append("FFT inversa, longitudes par/impar, eje de variables y gradientes finitos")

    train = CaudalDataset(config["train_path"], prep, 0)
    val = CaudalDataset(config["train_path"], prep, 1)
    test = CaudalDataset(config["test_path"], prep)
    assert len(np.intersect1d(train.ids, val.ids)) == 0
    assert len(train) == 254000 and len(val) == 18142 and len(test) == 27983
    assert not test.has_target
    rng = np.random.default_rng(config["seed"])
    # Muestreo aleatorio solo para el piloto, los experimentos usan todas las filas.
    indices = rng.choice(len(train), 256, replace=False)
    batch = [train[int(i)] for i in indices]
    x = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
    y = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
    with h5py.File(ruta(config["train_path"]), "r") as f:
        first = batch[0]
        raw = f["X"][first[2]][:, prep["selected"]]
        restored = first[0] * np.asarray(prep["std"])[prep["selected"]] + np.asarray(prep["mean"])[prep["selected"]]
        np.testing.assert_allclose(restored, raw, atol=.03, rtol=1e-5)
    checks.append("Particiones oficiales disjuntas, test sin etiquetas e inversión de normalización")

    trials = []
    for kind in ("lstm", "fdmlp", "fdmlp_lineal"):
        semilla(config["seed"])
        net = modelo(kind, config, prep, device)
        optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
        amp = config["mixed_precision"] and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        # Se contrasta una inferencia FP32 y AMP antes del entrenamiento piloto.
        net.eval()
        with torch.no_grad():
            reference = net(x[:8]).float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                approximate = net(x[:8]).float()
        amp_error = float((reference-approximate).abs().max())
        assert amp_error < .02
        net.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(net(x[:8]), y[:8]).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        losses = []
        batch_size = config["batch_size"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sincronizar(device)
        start = time.perf_counter()
        for step in range(20):
            begin = (step*batch_size) % len(x)
            xx, yy = x[begin:begin+batch_size], y[begin:begin+batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = net(xx)
                assert prediction.shape == (len(xx), 48)
                loss = torch.nn.functional.mse_loss(prediction, yy)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        sincronizar(device)
        trials.append({"model": kind, "steps": 20, "losses": losses,
                       "seconds": time.perf_counter()-start,
                       "peak_mib": torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else None,
                       "amp_max_abs_error_normalized": amp_error,
                       "parameters": sum(p.numel() for p in net.parameters())})
        print(json.dumps({k: v for k, v in trials[-1].items() if k != "losses"}), flush=True)
    checks.append("Tres modelos: 20 pasos, salida 48, pérdidas y gradientes finitos, sin OOM")
    guardar_json(ROOT / "resultados/pruebas.json", {"checks": checks, "passed": True, "environment": entorno(),
                                                    "pilot": trials, "pilot_weights_used_in_final_training": False})
    print("PRUEBAS COMPLETADAS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuevos", action="store_true")
    if parser.parse_args().nuevos:
        ampliacion()
    else:
        main()
        ampliacion()
