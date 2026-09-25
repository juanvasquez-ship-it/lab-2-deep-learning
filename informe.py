"""Genera tablas, figuras e informe a partir de resultados medidos."""

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from datos import ROOT, guardar_json, ruta
from metricas import calcular, resumen, extremos_fdc

NAMES = {"lstm": "LSTM base", "am": "AM-LSTM", "cnn": "CNN-LSTM", "gnn": "GNN-LSTM", "fdmlp": "FDMLP-LSTM", "fdmlp_lineal": "FDMLP sin CReLU", "lstm_uni": "LSTM univariado", "persistencia": "Persistencia"}
COLORS = {"lstm": "#3465a4", "am": "#b67619", "cnn": "#8754a1", "gnn": "#008c95", "fdmlp": "#b43d38", "fdmlp_lineal": "#5d8a4a", "lstm_uni": "#bd6c96", "persistencia": "#777777"}
PRIMARY = ["lstm", "am", "cnn", "gnn", "fdmlp"]
NAMES.update(mlp="MLP-LSTM", fdmlp_residual="FDMLP residual", mlp_residual="MLP residual")
COLORS.update(mlp="#495d75", fdmlp_residual="#c47531", mlp_residual="#746751")


def tabla(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def number(x):
    return "no definido" if x is None else f"{x:.5f}"


def main():
    out = ROOT / "resultados"
    figures = out / "figuras"
    figures.mkdir(exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((out / "preprocesamiento.json").read_text(encoding="utf-8"))
    metrics, basins, arrays, timing, history, clipped_metrics = {}, {}, {}, {}, {}, {}
    for kind in config["models"]:
        folder = out / kind
        arrays[kind] = dict(np.load(folder / "validacion.npz"))
        a = arrays[kind]
        metrics[kind], basins[kind] = resumen(a["observed"], a["prediction"], a["basin_id"])
        clipped, _ = resumen(a["observed"], np.maximum(a["prediction"], 0), a["basin_id"])
        clipped_metrics[kind] = clipped
        for row in basins[kind]:
            mask = a["basin_id"] == row["basin_id"]
            row.update(extremos_fdc(a["observed"][mask], a["prediction"][mask]))
        guardar_json(folder / "metricas.json", {"raw": metrics[kind], "nonnegative": clipped,
                                                "fdc_nonnegative": extremos_fdc(a["observed"], a["prediction"])})
        tabla(folder / "metricas_por_cuenca.csv", basins[kind])
        tabla(folder / "metricas_por_horizonte.csv", [{"hour": h+1, **calcular(a["observed"][:, h], a["prediction"][:, h])} for h in range(48)])
        timing[kind] = json.loads((folder / "finalizado.json").read_text(encoding="utf-8"))
        timing[kind].update(json.loads((folder / "inferencia.json").read_text(encoding="utf-8")))
        with (folder / "historial.csv").open(encoding="utf-8") as f:
            history[kind] = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    a = arrays["fdmlp"]
    persistence = np.repeat(a["last_discharge"][:, None], 48, axis=1)
    metrics["persistencia"], basins["persistencia"] = resumen(a["observed"], persistence, a["basin_id"])
    arrays["persistencia"] = {**a, "prediction": persistence}
    guardar_json(out / "metricas_comparacion.json", metrics)
    rows = [{"model": kind, **{k: m[k] for k in ("RMSE", "MAE", "NSE", "R", "NSE_basin_mean", "NSE_basin_median", "negative_prediction_fraction")}} for kind, m in metrics.items()]
    tabla(out / "comparacion.csv", rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.dpi": 160})
    nrows = (len(config["models"])+3)//4
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 3.5*nrows), sharey=True, squeeze=False, constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, kind in zip(axes.flat, config["models"]):
        ax.set_visible(True)
        h = history[kind]
        ax.plot([r["epoch"] for r in h], [r["train_mse_normalized"] for r in h], label="Entrenamiento")
        ax.plot([r["epoch"] for r in h], [r["validation_mse_normalized"] for r in h], label="Validación")
        ax.axvline(timing[kind]["best_epoch"], color="gray", linestyle=":", label="Mejor época")
        ax.set(title=NAMES[kind], xlabel="Época", ylabel="MSE normalizado")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.legend(fontsize=8)
    fig.savefig(figures / "aprendizaje.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for kind, a in arrays.items():
        per_hour = [calcular(a["observed"][:, h], a["prediction"][:, h]) for h in range(48)]
        for ax, metric in zip(axes, ("RMSE", "NSE")):
            ax.plot(np.arange(1, 49), [m[metric] for m in per_hour], label=NAMES[kind], color=COLORS[kind])
            ax.set(xlabel="Horas futuras", ylabel="RMSE (mm/h)" if metric == "RMSE" else "NSE global")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.savefig(figures / "horizonte.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    distributions = [[r["NSE"] for r in basins[k] if r["NSE"] is not None] for k in arrays]
    for ax in axes:
        ax.boxplot(distributions, tick_labels=[NAMES[k] for k in arrays], showfliers=True)
        ax.axhline(0, color="gray", linestyle=":")
        ax.set(ylabel="NSE por cuenca")
        ax.tick_params(axis="x", labelsize=7, rotation=25)
        ax.grid(axis="y", alpha=.2)
    axes[0].set(ylim=(-1, 1.05), title="Detalle central: NSE entre −1 y 1")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set(ylim=(min(min(d) for d in distributions)*1.2, 1.05),
                title="Todas las cuencas: escala symlog")
    fig.savefig(figures / "cuencas.png")
    plt.close(fig)

    a = arrays["fdmlp"]
    errors = np.mean((a["prediction"] - a["observed"])**2, axis=1)
    order = np.argsort(errors)
    choices = [("Menor error", order[0]), ("Error mediano", order[len(order)//2]),
               ("Mayor error", order[-1]), ("Mayor caudal observado", int(np.argmax(a["observed"].max(axis=1))))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    chosen_rows = []
    with h5py.File(ruta(config["train_path"]), "r") as source:
        for ax, (label, row) in zip(axes.flat, choices):
            idx = int(a["Id"][row])
            past = source["X"][idx, -48:, 11]
            ax.plot(np.arange(-47, 1), past, color="black", label="Historia (últimas 48 h)")
            ax.plot(np.arange(1, 49), a["observed"][row], color="black", linestyle="--", label="Observado")
            for kind in PRIMARY:
                if kind not in arrays:
                    continue
                ax.plot(np.arange(1, 49), arrays[kind]["prediction"][row], color=COLORS[kind], label=NAMES[kind])
            ax.axvline(0, color="gray", linestyle=":")
            ax.set(title=f"{label}, Id {idx}, cuenca {int(a['basin_id'][row])}", xlabel="Hora relativa", ylabel="Caudal (mm/h)")
            ax.legend(fontsize=7)
            chosen_rows.append({"criterion": label, "Id": idx, "basin_id": int(a["basin_id"][row]), "RMSE": float(np.sqrt(errors[row]))})
    fig.savefig(figures / "ejemplos.png")
    plt.close(fig)
    guardar_json(out / "ejemplos_seleccionados.json", chosen_rows)

    flat_y, flat_p = a["observed"].ravel(), a["prediction"].ravel()
    chosen = np.random.default_rng(config["seed"]).choice(len(flat_y), min(20000, len(flat_y)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(flat_y[chosen], flat_p[chosen], s=3, alpha=.18, color=COLORS["fdmlp"], rasterized=True)
    maximum = max(float(flat_y[chosen].max()), float(flat_p[chosen].max()))
    ax.plot([0, maximum], [0, maximum], "k--", lw=1)
    ax.set(xlabel="Observado (mm/h)", ylabel="Predicho (mm/h)", title="FDMLP-LSTM, 20.000 pares de validación")
    fig.savefig(figures / "dispersion.png")
    plt.close(fig)

    baseline, full, ablated = metrics["lstm"], metrics["fdmlp"], metrics["fdmlp_lineal"]
    improvement = 100*(baseline["RMSE"]-full["RMSE"])/baseline["RMSE"]
    nonlinear = 100*(ablated["RMSE"]-full["RMSE"])/ablated["RMSE"]
    change_base = "reducción" if improvement >= 0 else "incremento"
    change_ablation = "reducción" if nonlinear >= 0 else "incremento"
    best = min(config["models"], key=lambda k: metrics[k]["RMSE"])
    best_mae = min(config["models"], key=lambda k: metrics[k]["MAE"])
    best_median = max(config["models"], key=lambda k: metrics[k]["NSE_basin_median"])
    report = ["# Resultados del Laboratorio 2", "",
              "Pronóstico horario de caudal, 336 horas históricas y 48 horas futuras, unidades mm/h.", "",
              "## Protocolo", "",
              f"Se utilizaron {prep['train_samples']:,} ventanas de entrenamiento y {prep['validation_samples']:,} de validación, respetando la partición entregada, las 508 cuencas están representadas en ambas. LASSO con validación cruzada de diez grupos de cuencas conservó {len(prep['selected'])} variables. La normalización y la selección se ajustaron únicamente en entrenamiento. Los objetivos de test no se utilizaron.", "",
              "Se ejecutó una semilla. Los cinco modelos principales y la ablación comparten tamaño e inicialización del tronco LSTM, orden de lotes y política de entrenamiento. El control univariado conserva el tamaño oculto, con una sola variable de entrada y una matriz de entrada necesariamente diferente. La ablación conserva las capas complejas y elimina CReLU, el LSTM base retira todo FDMLP. Persistencia repite el último caudal observado. Los resultados no constituyen una prueba de significancia entre semillas.", "",
              "## Comparación en validación", "",
              "Métricas de predicciones crudas, calculadas en unidades originales sobre las 48 salidas de cada ventana. NSE global y R mezclan cuencas, por ello se incluye la mediana del NSE por cuenca. Las ventanas pueden solaparse y no representan necesariamente horas independientes.", "",
              "| Modelo | RMSE (mm/h) | MAE (mm/h) | NSE global | R | Mediana NSE por cuenca |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, m in metrics.items():
        report.append(f"| {NAMES[k]} | {number(m['RMSE'])} | {number(m['MAE'])} | {number(m['NSE'])} | {number(m['R'])} | {number(m['NSE_basin_median'])} |")
    report += ["", f"El menor RMSE entre modelos entrenados corresponde a **{NAMES[best]}**. El FDMLP-LSTM presenta un **{change_base} del RMSE de {abs(improvement):.2f}%** respecto al LSTM base y un **{change_ablation} de {abs(nonlinear):.2f}%** respecto a la versión sin CReLU. Esta comparación evalúa la adaptación implementada, no reproduce los valores del dataset original del paper.", "",
               f"El ranking depende del criterio: {NAMES[best_mae]} obtiene el menor MAE y {NAMES[best_median]} la mayor mediana de NSE por cuenca. Una ventaja en RMSE global no implica ser mejor en todas las cuencas o para todos los tamaños de error.", "",
               "![Curvas de aprendizaje](resultados/figuras/aprendizaje.png)", "",
               "![Métricas por horizonte](resultados/figuras/horizonte.png)", "",
               "![Resultados por cuenca](resultados/figuras/cuencas.png)", "",
               "El NSE global puede ocultar errores en cuencas de caudal pequeño. La siguiente tabla conserva la media aritmética, sensible a cuencas con variabilidad casi nula, junto con el número de cuencas con NSE positivo. No se excluyen cuencas para mejorar el resumen.", "",
               "| Modelo | Media NSE por cuenca | Cuencas con NSE > 0 | Peor NSE por cuenca |",
               "|---|---:|---:|---:|"]
    for k, m in metrics.items():
        values = [r["NSE"] for r in basins[k] if r["NSE"] is not None]
        report.append(f"| {NAMES[k]} | {number(m['NSE_basin_mean'])} | {sum(v > 0 for v in values)}/{len(values)} | {number(min(values))} |")
    worst = min((r for r in basins["fdmlp"] if r["NSE"] is not None), key=lambda r: r["NSE"])
    yy = arrays["fdmlp"]["observed"][arrays["fdmlp"]["basin_id"] == worst["basin_id"]]
    wins_base = sum(f["RMSE"] < b["RMSE"] for f, b in zip(basins["fdmlp"], basins["lstm"]))
    wins_ablation = sum(f["RMSE"] < b["RMSE"] for f, b in zip(basins["fdmlp"], basins["fdmlp_lineal"]))
    report += ["", f"El peor NSE del FDMLP-LSTM corresponde a la cuenca {worst['basin_id']}: {worst['NSE']:.2f}, con RMSE {worst['RMSE']:.6f} mm/h y desviación observada {float(yy.std()):.8f} mm/h. Al dividir por una varianza pequeña, NSE amplifica errores que parecen pequeños en unidades absolutas; el resultado indica una predicción deficiente respecto a la media observada de esa cuenca.", "",
               f"FDMLP-LSTM obtiene menor RMSE que el LSTM base en {wins_base}/508 cuencas y que la ablación en {wins_ablation}/508. Son comparaciones descriptivas de esta ejecución, no pruebas estadísticas con muestras independientes.", "",
               "## Coste y selección de modelos", "",
               "| Modelo | Parámetros | Épocas ejecutadas | Mejor época | Entrenamiento total (min) | Inferencia validación (s) |",
               "|---|---:|---:|---:|---:|---:|"]
    for k, t in timing.items():
        report.append(f"| {NAMES[k]} | {t['parameters']:,} | {t['epochs']} | {t['best_epoch']} | {t['total_training_seconds']/60:.2f} | {t['seconds']:.2f} |")
    report += ["", "Los tiempos incluyen lectura y transferencias. Durante la ejecución hubo otras aplicaciones que compartieron RAM y GPU; la variación de carga impide interpretar diferencias de tiempo entre variantes como diferencias intrínsecas de eficiencia. Tampoco son una comparación controlada con la RTX 3090 del paper. Los pilotos tienen calentamiento desigual y no se usan para afirmar superioridad computacional.", "",
               "## Errores y extremos", "",
               "Se muestran automáticamente cuatro ventanas de validación, elegidas por menor, mediano y mayor error del FDMLP-LSTM, y por mayor caudal observado. Se visualizan las últimas 48 horas del historial, aunque el modelo recibe las 336. La selección se guarda con sus identificadores para poder reproducirla.", "",
               "![Ejemplos de predicción](resultados/figuras/ejemplos.png)", "",
               "![Dispersión](resultados/figuras/dispersion.png)", "",
               "| Modelo | RMSE caudales altos | Sesgo volumen altos (%) | RMSE caudales bajos | Sesgo volumen bajos (%) |",
               "|---|---:|---:|---:|---:|"]
    for k, m in metrics.items():
        report.append(f"| {NAMES[k]} | {number(m['high']['RMSE'])} | {number(m['high']['volume_bias_pct'])} | {number(m['low']['RMSE'])} | {number(m['low']['volume_bias_pct'])} |")
    full_array = arrays["fdmlp"]
    first_hour = calcular(full_array["observed"][:, 0], full_array["prediction"][:, 0])
    last_hour = calcular(full_array["observed"][:, -1], full_array["prediction"][:, -1])
    peak_row, peak_hour = np.unravel_index(np.argmax(full_array["observed"]), full_array["observed"].shape)
    direction = "subestima" if full["high"]["volume_bias_pct"] < 0 else "sobreestima"
    report += ["", "Altos y bajos se definen con los percentiles 98 y 30 de las observaciones de validación, respectivamente, y se conservan exactamente las mismas máscaras para todos los modelos. El sesgo es 100 × suma(predicción − observación) / suma(observación). La suma de caudales específicos de ventanas posiblemente solapadas no representa un volumen físico total en m³. Estos diagnósticos condicionales no se presentan como una reproducción exacta de FHV/FLV. Los valores indefinidos no se reemplazan por cero.", "",
               f"FDMLP-LSTM {direction} la suma de los caudales altos en un {abs(full['high']['volume_bias_pct']):.2f}%. Su RMSE a una hora es {first_hour['RMSE']:.5f} mm/h y a 48 horas es {last_hour['RMSE']:.5f} mm/h. El entrenamiento con MSE global pondera fuertemente los errores grandes; estas métricas complementarias permiten detectar limitaciones en extremos y en los plazos más largos.", "",
               f"En la ventana Id {int(full_array['Id'][peak_row])}, el mayor pico observado alcanza {full_array['observed'][peak_row, peak_hour]:.3f} mm/h a {peak_hour+1} horas, mientras FDMLP-LSTM predice {full_array['prediction'][peak_row, peak_hour]:.3f} mm/h en ese instante. La figura permite comparar la respuesta de los cinco modelos a esta crecida. No se dispone de meteorología futura, y el caudal histórico mostrado por sí solo no describe la magnitud del pico posterior; no se atribuye su causa física sin fechas y contexto meteorológico.", "",
               f"En el grupo de caudales bajos, el sesgo medio del FDMLP-LSTM es {full['low']['bias_mm_h']:.6f} mm/h y su sesgo relativo es {full['low']['volume_bias_pct']:.2f}%. Los denominadores pequeños amplifican los porcentajes, por lo que deben interpretarse junto con RMSE y sesgo absoluto. La mediana de NSE por cuenca y las ventanas casi constantes también evidencian limitaciones que el RMSE global atenúa.", "",
               "## Restricción de caudal no negativo", "",
               "| Modelo | Predicciones negativas (%) | RMSE crudo | RMSE después del recorte a cero |",
               "|---|---:|---:|---:|"]
    for k, m in clipped_metrics.items():
        report.append(f"| {NAMES[k]} | {100*metrics[k]['negative_prediction_fraction']:.2f} | {number(metrics[k]['RMSE'])} | {number(m['RMSE'])} |")
    report += ["", "El recorte físico se declara por separado y no interviene en la selección de checkpoints. Como los objetivos son no negativos, el recorte no puede aumentar el error cuadrático de una predicción negativa; tampoco resuelve los errores de magnitud o de anticipación de crecidas.", "",
               "## Predicciones finales y limitaciones", "",
               "El archivo principal es `resultados/fdmlp/predicciones_test.csv`, con Id y q_01…q_48, 27.983 filas en el orden original, en mm/h. Se aplica máximo(predicción, 0) únicamente para imponer caudales no negativos, se conservan los valores crudos y también las métricas de validación con y sin ese recorte. Las predicciones de los comparadores se conservan en sus respectivas carpetas.", "",
               "Las limitaciones principales son la ausencia de fechas y coordenadas, la falta de los hiperparámetros del suplemento y la realización de una sola semilla. No se puede certificar independencia temporal entre particiones ni comparar numéricamente RMSE en mm/h con el RMSE en m³/s del paper. La FFT describe el eje de variables, su frecuencia depende del orden de canales y no debe interpretarse directamente como periodicidad temporal o causalidad hidrológica.", "",
               "Las adaptaciones, ecuaciones y comandos de reproducción se documentan en README.md. No se generaron matrices de confusión porque este problema es de regresión."]
    report += estudio_ampliado(arrays, metrics, figures)
    if all(k in metrics for k in ("mlp", "fdmlp_residual", "mlp_residual")):
        report += controles_metodologicos(metrics, timing, figures)
    (ROOT / "RESULTADOS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Informe y figuras generados.")


def controles_metodologicos(metrics, timing, figures):
    kinds = ["lstm", "fdmlp", "mlp", "fdmlp_residual", "mlp_residual"]
    report = ["", "## Controles del módulo y de la inicialización", "",
              "Se conserva el FDMLP original. Se añaden un MLP real y dos variantes residuales con la semilla 42, las mismas muestras y el mismo criterio de parada. Son controles exploratorios planteados después de evaluar la versión inicial y reutilizan la misma validación, no constituyen una confirmación independiente. Las variantes residuales usan x + F(x), con la última transformación inicializada a 0,01 veces la identidad y sesgos cero. Se evalúa conjuntamente la conexión residual y el inicio cercano a identidad; este contraste no separa ambos efectos.", "",
              "El MLP real aplica Linear(12,12), ReLU y Linear(12,12), sin Fourier. Su módulo tiene 312 parámetros frente a 56 del FDMLP, una diferencia de 256 parámetros (aproximadamente 0,12% del modelo completo). Esta diferencia deriva de las conexiones densas reales frente a los productos complejos elemento a elemento de la ecuación 11. El control contrasta familias de módulos, pero no aísla exclusivamente la base de Fourier ni iguala la capacidad de sus módulos.", "",
              "| Modelo | RMSE (mm/h) | NSE | Parámetros totales | Mejor época | Épocas ejecutadas | Entrenamiento (min) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for kind in kinds:
        t = timing[kind]
        row = {"model": kind, "seed": 42, "RMSE": metrics[kind]["RMSE"], "NSE": metrics[kind]["NSE"],
               "parameters": t["parameters"], "best_epoch": t["best_epoch"], "epochs": t["epochs"],
               "training_minutes": t["total_training_seconds"]/60}
        rows.append(row)
        report.append(f"| {NAMES[kind]} | {row['RMSE']:.6f} | {row['NSE']:.5f} | {row['parameters']} | {row['best_epoch']} | {row['epochs']} | {row['training_minutes']:.2f} |")
    tabla(ROOT / "resultados/controles_metodologicos.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    vals = [metrics[k]["RMSE"] for k in kinds]
    ax.barh([NAMES[k] for k in kinds], vals, color=[COLORS[k] for k in kinds])
    ax.invert_yaxis()
    for i, value in enumerate(vals):
        ax.text(value+.0003, i, f"{value:.5f}", va="center", fontsize=10)
    ax.set(xlabel="RMSE de validación (mm/h), menor es mejor", xlim=(0, max(vals)*1.14))
    fig.savefig(figures / "controles_metodologicos.png")
    plt.close(fig)
    report += ["", "El tiempo acumulado de entrenamiento depende de las épocas y de la carga del equipo durante cada ejecución. Los experimentos originales y los nuevos se realizaron en momentos distintos; estos minutos no constituyen un benchmark comparable de velocidad por lote.", "",
               "![Controles metodológicos](resultados/figuras/controles_metodologicos.png)", ""]
    for original, revised in (("fdmlp", "fdmlp_residual"), ("mlp", "mlp_residual"), ("mlp_residual", "fdmlp_residual")):
        delta = 100*(metrics[revised]["RMSE"]/metrics[original]["RMSE"]-1)
        report.append(f"Respecto a {NAMES[original]}, {NAMES[revised]} cambia el RMSE en {delta:+.2f}% (negativo significa menor error).")
    report += ["", "La ablación sin CReLU conserva su finalidad original: medir el efecto de esa no linealidad. No demuestra por sí sola una ventaja del dominio frecuencial. El bloque completo con CReLU no empieza como identidad, aunque sus capas complejas aisladas sí tengan pesos identidad. Las pruebas de inicialización se conservan en `resultados/pruebas_controles.json`; la diferencia entre entrada y salida no es un porcentaje de información predictiva perdida.", "",
               "Las diferencias de RMSE son descriptivas. Un resultado favorable de estas variantes no demuestra por sí solo la superioridad de Fourier ni identifica la causa del resultado original. No se incorporaron proyecciones latentes de 128 dimensiones, cambios a multiplicación compleja densa ni rotaciones aleatorias, porque no son correcciones demostradas por el texto del paper y ampliarían los factores experimentales."]
    initialization = json.loads((ROOT / "resultados/pruebas_controles.json").read_text(encoding="utf-8"))
    report += ["", "### Cambio de representación al inicializar", "",
               "Medido en las mismas 64 ventanas de entrenamiento, seleccionadas con semilla 42 y normalizadas con las estadísticas de entrenamiento. Se informa 100 × norma(F(x) − x) / norma(x), antes de aprender. Este diagnóstico no es una métrica de pronóstico ni una medida de información perdida.", "",
               "| Módulo | Cambio relativo inicial (%) |", "|---|---:|"]
    for trial in initialization["trials"]:
        report.append(f"| {NAMES[trial['model']]} | {100*trial['initial_relative_input_change']:.3f} |")
    decision = ROOT / "resultados/presupuesto_semillas.json"
    if decision.exists():
        plan = json.loads(decision.read_text(encoding="utf-8"))
        report += ["", "### Presupuesto de repeticiones", "", plan["explanation"]]
    return report


def estudio_ampliado(arrays, metrics, figures):
    out = ROOT / "resultados"
    report = ["", "## Comparación experimental ampliada", "",
              "Los cinco modelos del artículo se adaptan al mismo pronóstico de 48 horas. Se evalúan prefijos de 12, 24 y 48 horas del mismo modelo, y cada hora por separado. No son tres modelos reentrenados con horizontes distintos, ni equivalen a los 1, 3 y 5 días del artículo. La ventana de 336 horas y la salida de 48 horas se mantienen por la consigna y los datos disponibles.", "",
              "Las arquitecturas, las adaptaciones y su justificación se detallan en ADAPTACIONES.md. No se hizo una búsqueda de hiperparámetros distinta para favorecer a un modelo, se compara una configuración común de referencia, sin afirmar que sea óptima para cada familia.", "",
              "### Precisión por plazo", "",
              "| Modelo | RMSE 1–12 h | RMSE 1–24 h | RMSE 1–48 h | NSE 1–48 h |",
              "|---|---:|---:|---:|---:|"]
    horizon_rows = []
    for kind, a in arrays.items():
        values = []
        for h in (12, 24, 48):
            m = calcular(a["observed"][:, :h], a["prediction"][:, :h])
            fdc = extremos_fdc(a["observed"][:, :h], a["prediction"][:, :h])
            horizon_rows.append({"model": kind, "hours": h, **m, **fdc})
            values.append(m["RMSE"])
        report.append(f"| {NAMES[kind]} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {number(metrics[kind]['NSE'])} |")
    tabla(out / "comparacion_plazos.csv", horizon_rows)
    report += ["", "### Caudales extremos: curvas de duración", "",
               "FHV compara el 2% superior de las curvas ordenadas de observaciones y predicciones. FLV compara la forma logarítmica del 30% inferior, con signo según la convención de Yilmaz et al. Se calculan sobre caudales no negativos, con un suelo de 10⁻⁶ mm/h para los logaritmos. No son los diagnósticos condicionales de la sección anterior y no requieren emparejar los picos por fecha.", "",
               "Definiciones: [NeuralHydrology, FHV y FLV](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html). Se implementaron las fórmulas sin incorporar esa biblioteca ni reutilizar modelos externos. Dado que faltan los detalles del suplemento, se declara esta convención explícita y no una identidad exacta con los valores de los autores.", "",
               "| Modelo | FHV global (%) | FLV global (%) | Mediana FHV por cuenca | Mediana FLV por cuenca | Cuencas con FLV definido |",
               "|---|---:|---:|---:|---:|---:|"]
    sensitivity = []
    for kind, a in arrays.items():
        fdc = extremos_fdc(a["observed"], a["prediction"])
        per_basin = [extremos_fdc(a["observed"][a["basin_id"] == b], a["prediction"][a["basin_id"] == b]) for b in np.unique(a["basin_id"])]
        high = [r["FHV_pct"] for r in per_basin if r["FHV_pct"] is not None]
        low = [r["FLV_pct"] for r in per_basin if r["FLV_pct"] is not None]
        report.append(f"| {NAMES[kind]} | {number(fdc['FHV_pct'])} | {number(fdc['FLV_pct'])} | {number(float(np.median(high)) if high else None)} | {number(float(np.median(low)) if low else None)} | {len(low)}/508 |")
        for eps in (1e-8, 1e-6, 1e-4):
            sensitivity.append({"model": kind, **extremos_fdc(a["observed"], a["prediction"], eps)})
    tabla(out / "sensibilidad_FLV.csv", sensitivity)
    report += ["", "Se conserva sensibilidad de FLV a su suelo logarítmico en `resultados/sensibilidad_FLV.csv`. Los casos con denominador cero se declaran indefinidos. Como existen ceros y ventanas posiblemente solapadas, estas curvas son diagnósticos de los ejemplos evaluados, no curvas de una serie continua reconstruida.", ""]
    if "lstm_uni" in metrics:
        gain = 100*(metrics["lstm_uni"]["RMSE"]-metrics["lstm"]["RMSE"])/metrics["lstm_uni"]["RMSE"]
        report += ["### Una variable frente a múltiples variables", "",
                   f"El LSTM univariado usa únicamente caudal histórico. Su RMSE es {metrics['lstm_uni']['RMSE']:.5f} mm/h, frente a {metrics['lstm']['RMSE']:.5f} mm/h con las doce variables. El cambio favorable al añadir meteorología es {gain:.2f}% (un valor negativo indica empeoramiento). Se mantienen datos, normalización del objetivo, tamaño oculto y parada temprana. La diferencia de parámetros de entrada es inherente a cambiar de una a doce variables.", ""]
    benchmark = out / "benchmark.json"
    if benchmark.exists():
        rows = json.loads(benchmark.read_text(encoding="utf-8"))["results"]
        tabla(out / "benchmark.csv", rows)
        report += ["### Coste con carga equivalente", "",
                   "Medianas de 20 repeticiones, después de cinco de calentamiento, lote fijo de 64, datos residentes en GPU y sin lectura de disco. Los pasos de entrenamiento se ejecutan sobre copias que se descartan, sin modificar checkpoints. Esta medición reduce la influencia de los distintos números de épocas y del acceso a archivos, aunque otras aplicaciones aún pueden afectar al hardware.", "",
                   "| Modelo | Parámetros | Entrenamiento por lote (ms) | Inferencia por lote (ms) | Memoria máxima asignada (MiB) |",
                   "|---|---:|---:|---:|---:|"]
        for r in rows:
            report.append(f"| {NAMES[r['model']]} | {r['parameters']} | {r['training_median_ms']:.3f} | {r['inference_median_ms']:.3f} | {number(r['peak_allocated_mib'])} |")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, key, label in zip(axes, ("training_median_ms", "inference_median_ms"), ("Entrenamiento (ms/lote)", "Inferencia (ms/lote)")):
            ax.barh([NAMES[r["model"]] for r in rows], [r[key] for r in rows], color=[COLORS[r["model"]] for r in rows])
            ax.set(xlabel=label)
        fig.savefig(figures / "eficiencia.png")
        plt.close(fig)
        report += ["", "![Eficiencia](resultados/figuras/eficiencia.png)", ""]
        cost = {r["model"]: r for r in rows}
        report += [f"En esta medición, FDMLP-LSTM requiere {cost['fdmlp']['training_median_ms']:.2f} ms por lote de entrenamiento y {cost['fdmlp']['inference_median_ms']:.2f} ms de inferencia, frente a {cost['lstm']['training_median_ms']:.2f} y {cost['lstm']['inference_median_ms']:.2f} ms del LSTM base. El pequeño incremento de parámetros no garantiza una ejecución más rápida: también intervienen FFT, conversiones de precisión y operaciones intermedias. Estos tiempos describen esta implementación y este equipo.", ""]
    robustness = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "robustez").glob("*.json")) if p.name != "protocolo.json"]
    if robustness:
        rows = [{"model": r["model"], "scope": r["scope"], "perturbation": r["perturbation"], "level": r["level"], **m} for r in robustness for m in r["metrics"]]
        tabla(out / "robustez.csv", rows)
        protocol = json.loads((out / "robustez/protocolo.json").read_text(encoding="utf-8"))
        report += ["### Robustez ante ruido y faltantes", "",
                   f"Se evalúan las 18.142 ventanas de validación, sin reentrenar los modelos. Se añade ruido gaussiano de desviación 0,1, 0,5 y 1,0 en unidades normalizadas, o se elimina el 5%, 10% y 20% de los valores históricos y se interpola dentro de la historia. Se repite para todos los canales y para los auxiliares {protocol['auxiliary_channels']}, elegidos por magnitud de coeficiente LASSO únicamente en entrenamiento. Las mismas realizaciones aleatorias se aplican a los cinco modelos, y los niveles comparten ruido y máscaras anidadas.", "",
                   "La interpolación utiliza exclusivamente las 336 horas de entrada, con el vecino más próximo en los extremos. Un caso completamente ausente se rellena con la media de entrenamiento. Las horas futuras nunca intervienen. El artículo estudia principalmente FDMLP, aquí se incluyen los cinco modelos para contrastar la robustez bajo idénticas perturbaciones.", ""]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row_index, scope in enumerate(("todas", "auxiliares")):
            for column, perturbation in enumerate(("ruido", "faltantes")):
                ax = axes[row_index, column]
                for kind in PRIMARY:
                    points = sorted([r for r in rows if r["model"] == kind and r["scope"] == scope and r["perturbation"] == perturbation and r["hours"] == 48], key=lambda r: r["level"])
                    ax.plot([p["level"] for p in points], [p["relative_delta_NSE_pct"] for p in points], "o-", label=NAMES[kind], color=COLORS[kind])
                ax.axhline(0, color="gray", linestyle=":")
                ax.set(title=f"{scope.capitalize()}: {perturbation}", xlabel="Desviación normalizada" if perturbation == "ruido" else "Fracción ausente", ylabel="Cambio relativo del NSE (%)")
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
        fig.savefig(figures / "robustez.png")
        plt.close(fig)
        report += ["![Robustez](resultados/figuras/robustez.png)", "",
                   "| Modelo | ΔNSE (%) ruido σ=1, todos | ΔNSE (%) faltantes 20%, todos |",
                   "|---|---:|---:|"]
        for kind in PRIMARY:
            noise = next(r for r in rows if r["model"] == kind and r["scope"] == "todas" and r["perturbation"] == "ruido" and r["level"] == 1 and r["hours"] == 48)
            missing = next(r for r in rows if r["model"] == kind and r["scope"] == "todas" and r["perturbation"] == "faltantes" and r["level"] == .2 and r["hours"] == 48)
            report.append(f"| {NAMES[kind]} | {number(noise['relative_delta_NSE_pct'])} | {number(missing['relative_delta_NSE_pct'])} |")
        worst_noise = [r for r in rows if r["scope"] == "todas" and r["perturbation"] == "ruido" and r["level"] == 1 and r["hours"] == 48]
        most_stable = max(worst_noise, key=lambda r: r["relative_delta_NSE_pct"])
        fd_noise = next(r for r in worst_noise if r["model"] == "fdmlp")
        fd_missing = next(r for r in rows if r["model"] == "fdmlp" and r["scope"] == "todas" and r["perturbation"] == "faltantes" and r["level"] == .2 and r["hours"] == 48)
        report += ["", f"Con ruido σ=1 en todos los canales, {NAMES[most_stable['model']]} presenta la menor caída relativa del NSE. Para FDMLP-LSTM, el cambio es {fd_noise['relative_delta_NSE_pct']:.2f}% con ese ruido y {fd_missing['relative_delta_NSE_pct']:.2f}% con un 20% de faltantes interpolados. La resistencia a faltantes interpolables no implica resistencia al ruido intenso. No se observa una superioridad general de FDMLP en robustez; las diferencias de escala del ruido, variables y protocolo impiden equiparar directamente estos porcentajes con los del artículo.", ""]
    interpretation = out / "interpretabilidad.json"
    if interpretation.exists():
        result = json.loads(interpretation.read_text(encoding="utf-8"))
        gradients = result["mean_absolute_complex_gradient"]
        tabla(out / "gradientes_frecuencia.csv", [{"bin": i, "mean_absolute_gradient": value, "relative_importance": result["relative_importance"][i]} for i, value in enumerate(gradients)])
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.bar(np.arange(len(gradients)), gradients, color=COLORS["fdmlp"])
        ax.set(xlabel="Componente rFFT sobre variables", ylabel="Gradiente complejo absoluto medio", title="Sensibilidad FDMLP-LSTM, validación completa")
        fig.savefig(figures / "interpretabilidad.png")
        plt.close(fig)
        report += ["", "### Sensibilidad de componentes frecuenciales", "",
                   "Se calcula el gradiente absoluto complejo de la pérdida respecto a cada componente rFFT, en FP32 y sobre toda la validación. La pérdida es MSE por muestra para evitar que el tamaño del último lote cambie la escala del gradiente. Es una medida de sensibilidad local del modelo entrenado, no una demostración de causalidad.", "",
                   "![Sensibilidad frecuencial](resultados/figuras/interpretabilidad.png)", "",
                   f"La componente con mayor sensibilidad es el bin {int(np.argmax(gradients))}. Hay siete componentes porque se transforman directamente doce variables reales. No deben interpretarse como ciclos temporales ni trasladarse automáticamente las interpretaciones físicas de frecuencias bajas del artículo, el orden y la representación de los canales son distintos.", ""]
    paper_path = ROOT / "data/paper_rmse.csv"
    if paper_path.exists():
        with paper_path.open(encoding="utf-8") as f:
            paper = list(csv.DictReader(f))
        wins = {kind: sum(min(PRIMARY, key=lambda k: float(r[k])) == kind for r in paper) for kind in PRIMARY}
        winner = min(PRIMARY, key=lambda k: metrics[k]["RMSE"])
        report += ["## Contraste con las conclusiones del paper", "",
                   "La tabla 3 del artículo se transcribió en `data/paper_rmse.csv`. Según sus valores de RMSE, FDMLP-LSTM obtiene el menor error en siete de nueve combinaciones estación–horizonte y CNN-LSTM en las otras dos. Ese recuento se refiere a los números de la tabla, dado que algunos pasajes narrativos del artículo presentan discrepancias.", "",
                   f"En nuestro conjunto, el menor RMSE global entre los cinco modelos principales corresponde a **{NAMES[winner]}**. FDMLP-LSTM cambia el RMSE del LSTM base en {100*(metrics['fdmlp']['RMSE']-metrics['lstm']['RMSE'])/metrics['lstm']['RMSE']:.2f}% (positivo significa mayor error). Por tanto, el ranking observado {'coincide con la ventaja global' if winner == 'fdmlp' else 'no reproduce la ventaja global'} del modelo propuesto en el artículo.", "",
                   "| Modelo | Casos del paper con menor RMSE (de 9) | Nuestro RMSE 48 h | Cambio RMSE frente a LSTM (%) |",
                   "|---|---:|---:|---:|"]
        for kind in PRIMARY:
            report.append(f"| {NAMES[kind]} | {wins[kind]} | {metrics[kind]['RMSE']:.5f} | {100*(metrics[kind]['RMSE']-metrics['lstm']['RMSE'])/metrics['lstm']['RMSE']:.2f} |")
        report += ["", "Se comparan rankings, cambios relativos y patrones de error, no valores absolutos entre m³/s diarios del paper y mm/h horarios del laboratorio. También difieren estaciones, número de cuencas, variables, tamaño de los modelos, ventanas, objetivos y protocolo de evaluación. Un resultado distinto no refuta el paper ni prueba que un modelo sea universalmente superior.", "",
                   "Las tablas anteriores separan precisión, extremos, robustez y coste: el mejor resultado en una dimensión no implica ser el mejor en las demás. El protocolo utiliza una semilla y una configuración fijada antes de ver los resultados nuevos, no una búsqueda exhaustiva de la mejor arquitectura de cada familia.", "",
                   "La validación también se empleó para seleccionar checkpoints y detener el entrenamiento, por lo que estos resultados pueden ser optimistas respecto a una evaluación independiente. Las predicciones de test se exportan, pero sus etiquetas no se han utilizado para obtener métricas. Sin fechas originales tampoco puede certificarse independencia temporal entre las particiones."]
        if "lstm_uni" in metrics:
            report += ["", f"El uso de múltiples variables reduce el RMSE del LSTM en {gain:.2f}% respecto al control univariado, lo que coincide en dirección con el beneficio de información adicional estudiado en el paper. Esa coincidencia no se extiende automáticamente a la ventaja del módulo frecuencial."]
    return report


if __name__ == "__main__":
    main()
