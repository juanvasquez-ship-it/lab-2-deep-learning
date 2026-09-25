# Laboratorio 2: pronóstico de caudal con FDMLP-LSTM

Implementación propia del núcleo metodológico de Jia et al., *Enhancing streamflow forecasting using an LSTM hybrid model with lightweight frequency-domain feature learning*, Expert Systems with Applications 297 (2026), 129418. DOI: https://doi.org/10.1016/j.eswa.2025.129418.

El objetivo es predecir 48 valores horarios de caudal a partir de 336 horas de 12 variables históricas de una cuenca. Los caudales y las predicciones se expresan en mm/h. Esta es una adaptación al dataset del laboratorio, no una reproducción numérica del experimento del río Amarillo.

## Archivos principales

| Archivo | Función |
|---|---|
| `datos.py` | Lectura por muestra, normalización y selección de variables con LASSO |
| `modelos.py` | FDMLP, comparadores, ablación y controles residuales |
| `ejecutar.py` | Preparación, entrenamiento, validación, reanudación y predicciones |
| `metricas.py` | RMSE, MAE, NSE, correlación y diagnóstico por cuenca |
| `estudios.py` | Orquestación ampliada, robustez, benchmark y gradientes frecuenciales |
| `validar.py` | Pruebas matemáticas, particiones y piloto de entrenamiento |
| `informe.py` | Tablas, figuras y `RESULTADOS.md` a partir de las ejecuciones |
| `config.json` | Rutas y configuración utilizada |
| `RESULTADOS.md` | Comparación medida, gráficos, errores y limitaciones |
| `GUION.md` | Organización de la exposición de 18 minutos y preguntas de comprensión |
| `ADAPTACIONES.md` | Referencia original, configuración adoptada y justificación de diferencias |
| `presentacion/Laboratorio_2_Revision_FDMLP.pptx` | Exposición de 18 diapositivas principales y tres de apoyo, con notas |
| `resultados/` | Evidencia de pruebas, curvas, modelos y predicciones |

No se necesita una API de OpenAI ni otro servicio externo para entrenar o ejecutar el modelo.

## Contenido del repositorio

El repositorio incluye código, documentación, métricas, figuras, checkpoints y predicciones de las ejecuciones realizadas. Los archivos HDF5 del dataset y el entorno `.venv` se conservan localmente y no se distribuyen en Git.

Los registros de consola, los estados transitorios de progreso y los archivos temporales de guardado se generan localmente cuando se ejecutan los scripts y se excluyen de Git. La evidencia permanente del entrenamiento se conserva en `historial.csv` y `finalizado.json` de cada modelo. Los checkpoints, las pruebas de validación y las predicciones crudas se mantienen para reanudación, verificación y análisis.

Para ejecutar el proyecto después de clonarlo, crear el entorno siguiendo las instrucciones de abajo, obtener los datos del laboratorio y ajustar `train_path`, `test_path` y `metadata_path` en `config.json` a sus ubicaciones locales. La ruta de entrenamiento guardada corresponde al equipo donde se realizaron los experimentos. Los checkpoints conservan su configuración original como evidencia de esas ejecuciones.

## Entorno y ejecución

En VS Code, abrir esta carpeta y usar el intérprete `.venv/Scripts/python.exe`. El entorno preparado utiliza Python 3.10.8, PyTorch 2.1.1 con CUDA 12.1, NumPy 1.26.4, h5py 3.11.0, Matplotlib 3.9.4 y scikit-learn 1.5.2. PyTorch coincide con la versión declarada en el paper. La rueda CUDA incluye los componentes de ejecución necesarios, no hace falta instalar un CUDA Toolkit para este código.

Desde una terminal PowerShell en esta carpeta:

```powershell
# Comprobar GPU
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Preparar normalización y selección, únicamente con entrenamiento
.\.venv\Scripts\python.exe ejecutar.py preparar

# Ejecutar pruebas antes de entrenar
.\.venv\Scripts\python.exe validar.py

# Entrenar todos los modelos, conservando y reanudando las ejecuciones
.\.venv\Scripts\python.exe ejecutar.py entrenar

# Evaluación y predicción, usando el mejor checkpoint de cada modelo
.\.venv\Scripts\python.exe ejecutar.py evaluar
.\.venv\Scripts\python.exe ejecutar.py predecir

# Regenerar tablas, figuras y análisis
.\.venv\Scripts\python.exe informe.py

# Estudio completo: entrenamientos pendientes, evaluación, test y análisis adicionales
.\.venv\Scripts\python.exe estudios.py todo
```

Para seleccionar experimentos se admite `--modelos` seguido de `lstm`, `am`, `cnn`, `gnn`, `fdmlp`, `fdmlp_lineal` o `lstm_uni`. `ejecutar.py todo` reúne preparación, entrenamiento, evaluación y predicción. `estudios.py todo` reutiliza el preprocesamiento guardado y añade robustez, benchmark, interpretabilidad, informe y verificación final. Su estado queda en `resultados/estado_estudio.json`; los historiales y checkpoints permiten revisar el progreso sin supervisión interactiva continua. Para una nueva configuración, preservar primero los resultados anteriores en otra ubicación y usar una carpeta de resultados nueva o vacía, evitando mezclar experimentos. Los estudios adicionales omiten resultados ya existentes, por lo que también deben preservarse y regenerarse si cambian sus protocolos o los checkpoints.

Para recrear el entorno en otra máquina Windows con GPU NVIDIA compatible:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las versiones completas, incluyendo dependencias transitivas, se conservan en `resultados/dependencias_instaladas.txt`. La instalación se realiza desde PyPI y el índice oficial https://download.pytorch.org/whl/cu121. La configuración puede utilizar CPU si CUDA no está disponible, aunque el entrenamiento final se ejecuta con GPU.

## Datos y prevención de fugas

`config.json` apunta al archivo original `C:/Users/msdro/Downloads/train-001.h5`, no se duplica su contenido de aproximadamente 5 GB. `data/test.h5` y `data/metadata.json` se extrajeron del ZIP suministrado. Para trasladar el proyecto basta con copiar el entrenamiento y actualizar `train_path`.

Los HDF5 se abren solo para lectura, cada proceso de carga mantiene su propia conexión. Las claves reales son `X`, `y`, `y_aux`, `split` y `basin_id`. Se conserva `split=0` como entrenamiento y `split=1` como validación. No se vuelve a dividir aleatoriamente el dataset, ni se construyen ejemplos concatenando cuencas. Las 508 cuencas están presentes en ambas particiones y en test, por lo que esto no evalúa generalización a cuencas completamente nuevas.

La auditoría completa está en `resultados/auditoria_datos.json`. Se revisaron todas las matrices de ambos HDF5, no se encontraron NaN, infinitos o caudales negativos, ni coincidencias de firmas de ventanas completas dentro de una misma cuenca. Esa última comprobación no descarta solapamientos parciales. No hay marcas temporales para verificar independencia cronológica, reconstruir series continuas, añadir calendarios o deducir coordenadas.

La media y desviación de cada canal se calculan sobre todas las horas históricas de las muestras de entrenamiento, la normalización del objetivo se calcula únicamente con `y` de entrenamiento. Las estadísticas de validación y test no intervienen. Los ceros de precipitación o caudal se conservan. Los valores negativos en temperatura y componentes de viento no son valores faltantes, la evaporación potencial también contiene negativos y se mantiene su convención original porque las unidades meteorológicas no están documentadas.

`y_aux` es supervisión futura opcional y no se utiliza, ni siquiera como entrada. Se predice únicamente caudal. El ZIP contiene además `test_targets.csv`, no documentado en las instrucciones, su contenido se auditó pero no se usa en ningún entrenamiento, ajuste, selección ni evaluación predictiva. Las predicciones de test se generan exclusivamente desde `X`.

## Metodología

### Selección de variables

Se conserva la etapa LASSO del paper. Para adaptarla a muestras ya divididas en ventanas, cada predictor es la media de una variable sobre sus 336 horas históricas y el objetivo de selección es la media de las 48 horas futuras de caudal. Este resumen solo sirve para decidir qué canales entran en la red, la red recibe posteriormente todas las horas de cada canal seleccionado.

Se comparan cuatro valores predefinidos de regularización y se utiliza validación cruzada de diez grupos definidos por cuenca, exclusivamente dentro de entrenamiento. En cada fold se recalculan tanto la estandarización de predictores como la de la respuesta con su subconjunto de ajuste, el criterio es el MSE fuera del fold en mm/h. Se retiene siempre el canal histórico de caudal. La rejilla, coeficientes y errores por fold se conservan en `resultados/preprocesamiento.json`. En esta ejecución se seleccionaron las doce variables.

### FDMLP-LSTM

La forma de entrada es `[lote,336,12]`. Para cada instante se aplica una transformada real de Fourier **sobre las variables**, generando siete componentes complejas. No se transforma el eje de horas.

La ecuación 11 del paper define productos complejos elemento a elemento. En cada componente, una capa calcula:

```text
real_salida = real_entrada * peso_real - imag_entrada * peso_imag + sesgo_real
imag_salida = real_entrada * peso_imag + imag_entrada * peso_real + sesgo_imag
```

El bloque completo es `rFFT → capa compleja → CReLU → capa compleja → irFFT`. CReLU aplica ReLU de forma independiente a las partes real e imaginaria. Los pesos se comparten entre horas y muestras. El uso de rFFT e irFFT conserva la simetría conjugada y garantiza una salida real de 12 variables, el artículo no especifica cómo resuelve este detalle. Las partes imaginarias de DC y Nyquist no contribuyen a la salida real. Se adopta normalización ortonormal de la transformada y se documenta como decisión de implementación.

La secuencia transformada entra en una LSTM unidireccional de dos capas y 128 unidades por capa. Su último estado oculto se proyecta con una capa lineal a 48 caudales futuros simultáneamente, no se necesita meteorología futura ni alimentación recursiva de predicciones. La capa de salida es lineal durante el aprendizaje.

### Comparadores y ablación

El LSTM base utiliza exactamente las mismas variables, normalización, capas recurrentes y salida, omitiendo FDMLP. La ablación `fdmlp_lineal` mantiene ambas capas complejas, las transformadas y el número de parámetros, pero retira CReLU, por lo que aísla la contribución de la no linealidad frecuencial. Se añade persistencia como referencia sin entrenamiento, repitiendo el último caudal observado durante 48 horas.

Sin CReLU, la composición completa del bloque es una transformación afín de las variables de cada hora. Esta puede absorberse matemáticamente en las proyecciones de entrada y sesgos de la LSTM. Por tanto, una diferencia respecto al modelo base puede deberse a la parametrización y a la trayectoria de optimización, no demuestra por sí sola una mayor capacidad de representación.

Se incluyen los cinco modelos de la comparación principal: LSTM, AM-LSTM, CNN-LSTM, GNN-LSTM y FDMLP-LSTM. Se conservan la ablación sin CReLU, el control LSTM univariado y persistencia. Los detalles de atención, convolución sobre variables y grafo aprendido se describen en `ADAPTACIONES.md`, junto con su relación con las ecuaciones del paper y las elecciones de tamaños auxiliares.

## Configuración y diferencias frente al paper

| Aspecto | Paper | Implementación y justificación |
|---|---|---|
| Datos | Tres estaciones del río Amarillo, diarios | 508 cuencas anónimas y registros horarios, exigencia del laboratorio |
| Historia | 7, 14, 30, 90, 180 o 365 días | 336 horas, sin reducir longitud |
| Horizonte | 1, 3 y 5 días | 48 salidas horarias directas |
| Variables | Caudal de estaciones y precipitación | Los 12 canales hidrometeorológicos entregados |
| Particiones | 60/20/20 | `split` entregado, 254.000/18.142, test separado de 27.983 |
| LASSO | Diez folds, detalles temporales no completos | Diez grupos por cuenca dentro de entrenamiento y resumen causal por ventana |
| Normalización | Estandarización e inversión | Z-score por canal y del objetivo, ajustado solo con entrenamiento |
| FDMLP | FFT, dos transformaciones complejas y CReLU | Productos elemento a elemento según ecuación 11, rFFT/irFFT para salida real |
| LSTM | Tamaños remitidos a suplemento S1-S5 | Dos capas de 128 unidades, decisión explícita al no disponer del suplemento |
| PyTorch | 2.1.1 | 2.1.1+cu121 |
| Loss y optimizador | MSE, Adam | MSE y Adam, sin pesos auxiliares |
| Learning rate | 0,0001 | 0,0001, constante |
| Lote y épocas | 64, hasta 200 | 64, hasta 200, todos los datos |
| Parada temprana | Paciencia 5 | Paciencia 5, restauración del mejor checkpoint de validación |
| Precisión | No detallada | AMP en LSTM, FFT siempre float32/complex64, contrastada con FP32 |
| Gradientes | No detallado | Recorte de norma a 1,0 como protección frente a gradientes extremos, sin cambiar la pérdida |
| Inicialización FDMLP | No detallada | Pesos reales uno, imaginarios y sesgos cero, la ablación lineal inicia como identidad |
| Salida no negativa | No detallado | Recorte a cero solo en exportación, métricas crudas y recortadas guardadas por separado |
| Semillas | No detallado en texto principal | Semilla 42, una ejecución por variante, sin afirmar significancia estadística |

El texto principal remite a un suplemento ausente de los archivos recibidos, su descarga pública no estuvo disponible. No se afirma reproducir sus anchuras ocultas, búsqueda completa de hiperparámetros o cifras de parámetros. La arquitectura adoptada conserva el método de la ecuación 11 sin introducir un embedding de dimensión desconocida. Las erratas de signos en las fórmulas continuas de Fourier y LASSO se resuelven con las definiciones estándar, LASSO usa penalización L1 positiva y la transformada inversa es la inversa matemática de la directa.

La elección de dos capas de 128 unidades fija un tronco de aproximadamente 211.000 parámetros para los modelos multivariados, compartido y validado con las secuencias completas en la GPU disponible. Permite una comparación acotada y explicable sin añadir una búsqueda de arquitectura distinta para cada modelo. No se presenta como el tamaño óptimo ni como una reducción de un tamaño original conocido. El piloto confirmó que no era necesario reducir el lote, la historia o el dataset.

## Validación, resultados y predicciones

`validar.py` contrasta las operaciones complejas con la aritmética compleja de PyTorch, verifica la inversión Fourier y el eje de transformación, prueba dimensiones pares e impares, comprueba gradientes, particiones y métricas en casos conocidos, y ejecuta 20 pasos de cada modelo. Los pesos del piloto no se reutilizan en el experimento final. La comparación AMP/FP32 y la memoria máxima quedan registradas en `resultados/pruebas.json`.

Cada época conserva pérdidas de entrenamiento y validación, RMSE en mm/h, tiempo, memoria y pasos omitidos por el escalado de AMP. `mejor.pt` contiene el modelo seleccionado por menor MSE de validación, `ultimo.pt` conserva además Adam, escalador y estado de parada temprana para reanudar. Los pesos son generados por este proyecto y no deben sustituirse por checkpoints de origen desconocido.

La semilla, el orden de lotes por época y las opciones deterministas de cuDNN se controlan para comparar ejecuciones completas en este entorno. La reanudación recupera el estado de aprendizaje, pero no garantiza identidad bit a bit: reconstruir los procesos de carga puede consumir de forma distinta el generador de aleatoriedad del primer lote. Tampoco se garantiza identidad numérica entre versiones o dispositivos diferentes.

El análisis final está en `RESULTADOS.md`. Las métricas principales son NSE, RMSE y R, como en el paper, además de MAE y resultados por horizonte y cuenca. No se usan métricas de clasificación. Se distinguen los diagnósticos condicionales de caudales altos y bajos de FHV/FLV calculados sobre curvas ordenadas. Las fórmulas, regularización de ceros y sensibilidad de FLV quedan documentadas en `ADAPTACIONES.md`.

Las pruebas originales se conservan en `resultados/pruebas.json`. `validar.py --nuevos` ejecuta únicamente las comprobaciones de los comparadores añadidos, incluyendo gradientes, cinco pasos de optimización por modelo, operaciones de grafo/atención y compatibilidad con los checkpoints previos. Su evidencia está en `resultados/pruebas_ampliacion.json`.

La entrega principal de predicciones es `resultados/fdmlp/predicciones_test.csv`, columnas `Id,q_01,...,q_48`. Las 27.983 filas mantienen el orden original. No se convierte de mm/h a m³/s, ni se utiliza el CSV de objetivos para elegir el modelo. Un menor error del baseline o de la ablación, si se observa, debe reportarse como resultado experimental y no ocultarse.

La comprobación final está en `resultados/verificacion_final.json`: verifica correspondencia de métricas con checkpoints, igualdad exacta de los objetivos de validación, dimensiones y orden de los CSV, valores finitos y recorte correcto a cero. También registra las huellas SHA-256 del código y la configuración. Las huellas de los datos y documentos de entrada se conservan en `resultados/fuentes_sha256.json`.

## Controles metodológicos adicionales

`mlp`, `fdmlp_residual` y `mlp_residual` son controles propios y no reemplazan el FDMLP original. Permiten contrastar un MLP real sin Fourier y una estrategia de conexión residual con inicio cercano a identidad. La arquitectura y los límites de comparación figuran en `ADAPTACIONES.md`, los resultados se incorporan a `RESULTADOS.md`.

```powershell
.\.venv\Scripts\python.exe validar.py --controles
.\.venv\Scripts\python.exe ejecutar.py entrenar --modelos mlp fdmlp_residual mlp_residual
.\.venv\Scripts\python.exe ejecutar.py evaluar --modelos mlp fdmlp_residual mlp_residual
.\.venv\Scripts\python.exe ejecutar.py predecir --modelos mlp fdmlp_residual mlp_residual
.\.venv\Scripts\python.exe informe.py
.\.venv\Scripts\python.exe estudios.py verificar
```

Las pruebas comprueban la inicialización cercana a identidad de ambos residuales, los mismos pesos iniciales del tronco LSTM y diez pasos de aprendizaje de cada modelo nuevo. Sus pesos se descartan. La revisión de resultados reutiliza las ejecuciones originales y no altera sus checkpoints. Los resultados con una semilla son descriptivos y no establecen significancia estadística. Las curvas de duración, predicciones crudas y pruebas previas se conservan como evidencia.

## Fuentes y observaciones de la consigna

- `Laboratorio 2.pdf`, instrucciones locales suministradas, tres páginas.
- `main (2).pdf`, artículo completo suministrado, quince páginas.
- `metadata.json`, descripción efectiva del dataset, canal objetivo 11 y particiones.
- Instalación PyTorch: https://pytorch.org/get-started/previous-versions/#v211.

La consigna contiene fechas contradictorias, el encabezado indica entrega el 1 de octubre y revisión el 3, mientras los títulos posteriores dicen jueves 3 y sábado 5. Esta diferencia no afecta el código y debe resolverse por el calendario oficial del curso. El video tiene un máximo de 18 minutos y la evaluación posterior es individual, el material de presentación se apoya en la metodología, implementación y resultados reales aquí conservados.
