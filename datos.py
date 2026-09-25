"""Lectura HDF5, normalización y selección causal de variables con LASSO."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def ruta(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def guardar_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def preparar(config):
    """Todas las estadísticas y decisiones de selección se ajustan en split=0."""
    out = ROOT / "resultados"
    out.mkdir(exist_ok=True)
    meta = json.loads(ruta(config["metadata_path"]).read_text(encoding="utf-8"))
    with h5py.File(ruta(config["train_path"]), "r") as f:
        assert f["X"].shape[1:] == (336, 12)
        assert f["y"].shape == (len(f["X"]), 48)
        flags, basins = f["split"][:], f["basin_id"][:]
        if set(np.unique(flags)) != {0, 1}:
            raise ValueError("Las particiones no corresponden a train=0, validation=1")
        train_ids = np.flatnonzero(flags == 0)
        features = np.empty((len(train_ids), 12), dtype=np.float64)
        targets = np.empty(len(train_ids), dtype=np.float64)
        sums, squares = np.zeros(12), np.zeros(12)
        y_sum = y_squares = 0.0
        count = y_count = offset = 0
        for start in range(0, len(flags), 512):
            stop = min(start + 512, len(flags))
            mask = flags[start:stop] == 0
            if not mask.any():
                continue
            x = f["X"][start:stop][mask].astype(np.float64)
            y = f["y"][start:stop][mask].astype(np.float64)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Se requiere tratar explícitamente los valores no finitos")
            flat = x.reshape(-1, 12)
            sums += flat.sum(axis=0)
            squares += np.einsum("ij,ij->j", flat, flat)
            count += len(flat)
            y_sum += y.sum()
            y_squares += np.square(y).sum()
            y_count += y.size
            n = len(x)
            # Resumen causal por variable para LASSO, sin usar meteorología futura.
            features[offset:offset+n] = x.mean(axis=1)
            targets[offset:offset+n] = y.mean(axis=1)
            offset += n
        mean = sums / count
        std = np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 1e-8)
        y_mean = y_sum / y_count
        y_std = max(float(np.sqrt(max(y_squares / y_count - y_mean**2, 0))), 1e-8)

    alphas = config["lasso_alphas"]
    errors = [[] for _ in alphas]
    # Grupos por cuenca evitan repartir ventanas de una cuenca entre folds internos.
    folds = GroupKFold(n_splits=config["lasso_folds"])
    for fold, (a, b) in enumerate(folds.split(features, targets, basins[train_ids]), 1):
        xm, xs = features[a].mean(0), np.maximum(features[a].std(0), 1e-8)
        ym, ys = targets[a].mean(), max(targets[a].std(), 1e-8)
        xa = np.asfortranarray((features[a] - xm) / xs)
        xb = (features[b] - xm) / xs
        ya = (targets[a] - ym) / ys
        for j, alpha in enumerate(alphas):
            reg = Lasso(alpha=alpha, max_iter=10000, tol=1e-5).fit(xa, ya)
            pred = reg.predict(xb) * ys + ym
            errors[j].append(float(np.square(pred - targets[b]).mean()))
        print(f"LASSO fold {fold}/{config['lasso_folds']}", flush=True)
    alpha = alphas[int(np.argmin(np.mean(errors, axis=1)))]
    xm, xs = features.mean(0), np.maximum(features.std(0), 1e-8)
    ym, ys = targets.mean(), max(targets.std(), 1e-8)
    reg = Lasso(alpha=alpha, max_iter=10000, tol=1e-5).fit(
        np.asfortranarray((features - xm) / xs), (targets - ym) / ys)
    selected = sorted(set(np.flatnonzero(np.abs(reg.coef_) > 1e-8).tolist() + [meta["target_channel"]]))
    prep = {"mean": mean.tolist(), "std": std.tolist(), "y_mean": y_mean,
            "y_std": y_std, "selected": selected, "fit_split": 0,
            "train_samples": len(train_ids), "validation_samples": int((flags == 1).sum()),
            "lasso": {"alpha": alpha, "coefficients": reg.coef_.tolist(),
                      "alphas": alphas, "fold_mse": errors,
                      "feature_summary": "mean over 336 historical hours",
                      "target_summary": "mean over 48 future discharge hours",
                      "fold_group": "basin_id", "target_channel_always_retained": True},
            "source": str(ruta(config["train_path"])),
            "source_size": ruta(config["train_path"]).stat().st_size}
    guardar_json(out / "preprocesamiento.json", prep)
    print("Canales seleccionados:", selected, flush=True)
    return prep


class CaudalDataset(Dataset):
    def __init__(self, path, prep, split=None):
        self.path = str(ruta(path))
        self._file, self._pid = None, None
        self.selected = np.asarray(prep["selected"])
        self.mean = np.asarray(prep["mean"], dtype=np.float32)[self.selected]
        self.std = np.asarray(prep["std"], dtype=np.float32)[self.selected]
        self.y_mean, self.y_std = np.float32(prep["y_mean"]), np.float32(prep["y_std"])
        with h5py.File(self.path, "r") as f:
            self.has_target = "y" in f
            self.ids = np.arange(len(f["X"])) if split is None else np.flatnonzero(f["split"][:] == split)
            self.basins = f["basin_id"][:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r", rdcc_nbytes=4*1024*1024)
            self._pid = os.getpid()
        i = int(self.ids[index])
        raw = self._file["X"][i]
        x = np.ascontiguousarray((raw[:, self.selected] - self.mean) / self.std)
        y = (self._file["y"][i] - self.y_mean) / self.y_std if self.has_target else np.zeros(48, np.float32)
        return x, np.asarray(y, dtype=np.float32), i, int(self.basins[i]), np.float32(raw[-1, 11])

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"], state["_pid"] = None, None
        return state

    def __del__(self):
        self.close()
