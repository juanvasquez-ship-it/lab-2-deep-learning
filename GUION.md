# Guion de exposición, máximo 18 minutos

La exposición se organiza en tres bloques de seis minutos. Las cifras finales y las figuras provienen de `RESULTADOS.md`, no de resultados esperados ni de los números del dataset original del paper.

## 0:00–6:00, metodología

**0:00–1:00. Problema.** Presentar el pronóstico de 48 horas de caudal a partir de 336 horas históricas y 12 variables, aclarar que cada muestra pertenece a una sola cuenca y que la unidad es mm/h. Mostrar las formas `[B,336,12]` y `[B,48]`.

**1:00–2:00. Propuesta y comparadores.** Una LSTM modela dependencias temporales, la contribución FDMLP introduce un bloque para transformar las relaciones entre variables antes de esa LSTM. Presentar los cinco modelos: LSTM, AM-LSTM, CNN-LSTM, GNN-LSTM y FDMLP-LSTM. Distinguir atención entre canales, convolución sobre variables, grafo aprendido y transformaciones complejas. Todos reciben la misma historia y predicen las mismas 48 horas.

**2:00–4:00. Bloque frecuencial.** Mostrar `modelos.py`, seguir rFFT, primera transformación compleja, CReLU, segunda transformación e irFFT. Explicar con un ejemplo que `(a+ib)(c+id)=(ac−bd)+i(ad+bc)`. CReLU aplica ReLU por separado a ambas partes. La transformada se calcula para cada hora, sin incorporar horas futuras. La inversa vuelve a una representación real que puede recibir la LSTM.

**4:00–5:00. Selección y pronóstico.** Explicar LASSO y el ajuste exclusivo con entrenamiento, indicar que conservó los doce canales. La selección usa resúmenes históricos, pero la red sigue recibiendo la secuencia completa. El último estado oculto de la LSTM produce las 48 salidas de forma directa.

**5:00–6:00. Adaptaciones.** Contrastar datos diarios de tres estaciones con datos horarios de 508 cuencas. Justificar las 48 salidas, las particiones entregadas, rFFT para salida real y las dos capas de 128 unidades, cuyo tamaño se declara como una decisión porque el suplemento no estuvo disponible. La implementación del bloque complejo es propia.

## 6:00–12:00, implementación

**6:00–7:30. Datos.** Mostrar `metadata.json`, el canal 11 corresponde al caudal. Presentar los tamaños de entrenamiento, validación y test, y la auditoría de valores no finitos. Distinguir un cero hidrológico real de un valor faltante. No interpretar el identificador anónimo de cuenca como una cantidad física.

**7:30–9:00. Prevención de fugas.** Mostrar `datos.py`, las estadísticas y la selección se ajustan con `split=0`. Validación no determina las medias ni las desviaciones. Las variables meteorológicas futuras de `y_aux` no se usan como entradas. El CSV con posibles objetivos de test queda fuera del procedimiento. La ausencia de fechas impide garantizar independencia temporal entre particiones.

**9:00–10:30. Entrenamiento.** Explicar MSE, Adam con tasa 0,0001, lotes de 64, máximo de 200 épocas y paciencia de cinco. Mostrar cómo se guarda el mejor checkpoint y cómo se reanuda desde el último. Explicar la diferencia entre parámetros aprendidos e hiperparámetros fijados.

**10:30–12:00. Recursos y pruebas.** Mostrar `resultados/pruebas.json` y `resultados/entorno.json`, describir la GPU utilizada, lectura por lotes y AMP. La parte Fourier permanece en float32. Presentar la validación de formas, aritmética compleja, gradientes, particiones y 20 pasos por variante antes del experimento completo. Los pilotos no aportan pesos al entrenamiento final.

## 12:00–18:00, resultados

**12:00–13:30. Comparación principal.** Mostrar la tabla de `RESULTADOS.md`. Interpretar RMSE en mm/h, NSE respecto de predecir la media y correlación R. Indicar cuál obtuvo menor error según las mediciones, aunque no coincida con el ranking del paper. Distinguir NSE global de NSE por cuenca.

**13:30–15:00. Controles y contraste con el paper.** Presentar el MLP real sin Fourier y las variantes residuales. Separar el efecto conjunto de conexión residual e inicialización del contraste entre módulos. Indicar que MLP tiene 312 parámetros de módulo y FDMLP 56, por lo que no se aísla únicamente la base de Fourier. Recordar que retirar CReLU evalúa la activación y que la comparación univariada mide el aporte meteorológico. Contrastar con la tabla 3 del paper sin comparar RMSE absoluto entre unidades y datasets distintos. Las diferencias de una semilla son descriptivas.

**15:00–16:30. Errores.** Mostrar las curvas por horizonte y las ventanas seleccionadas automáticamente. Identificar suavizado de picos, retrasos o sesgos solo cuando las figuras los muestren. Los ejemplos abarcan error menor, mediano, mayor y el mayor caudal observado, su regla de selección está registrada. Describir los errores de caudales altos y bajos con las definiciones del informe.

**16:30–18:00. Robustez, coste y límites.** Mostrar el cambio de NSE con ruido y faltantes, el benchmark del mismo lote y los parámetros. Presentar la sensibilidad por componente Fourier sin interpretarla como causalidad o frecuencia temporal. Mostrar los CSV de 27.983 filas, señalar las diferencias de datos, horizonte, hardware y configuración frente al paper. Cerrar con la limitación de una semilla, fechas ausentes y test sin evaluación supervisada.

## Preguntas de comprensión

- **¿Qué aprende la FFT?** La transformada es fija, aprenden las capas complejas. La FFT cambia la representación del vector de variables de cada hora.
- **¿Por qué usar una LSTM después?** FDMLP se aplica independientemente a cada instante, la LSTM integra las dependencias a lo largo de las 336 horas.
- **¿Se usan datos futuros?** Los caudales futuros se usan como objetivos durante entrenamiento, nunca como entradas. No se usa meteorología futura para inferencia.
- **¿Por qué no basta una correlación alta?** Una predicción puede seguir la tendencia y mantener un sesgo importante, por eso se reportan error, NSE y sesgo.
- **¿Qué significa NSE negativo?** El error cuadrático supera al de una predicción constante igual a la media de las observaciones del conjunto evaluado.
- **¿Puede la ablación mejorar el modelo?** Sí, ese resultado puede indicar que la no linealidad no ayuda bajo esta adaptación, no autoriza a ocultar el experimento.
- **¿Es una reproducción exacta?** Se conserva el núcleo descrito en el artículo, pero el dataset, el horizonte, algunos detalles arquitectónicos no publicados en el texto principal y la precisión numérica requieren adaptaciones documentadas.
- **¿La frecuencia representa ciclos de lluvia?** No directamente, aquí la FFT se aplica al eje de variables, no al tiempo, y depende del orden de canales.
- **¿El FDMLP original empieza como identidad?** No, CReLU cambia la representación aunque las capas complejas aisladas tengan pesos identidad. Las variantes residuales conservan un camino directo y una corrección inicial pequeña.
- **¿Las variantes residuales pertenecen al paper?** Son controles propios de la adaptación, se mantienen separados del método original. No se añadieron proyecciones de 128 dimensiones que no se pudieron confirmar.
- **¿MLP frente a FDMLP prueba definitivamente la ventaja de Fourier?** No, también cambian conectividad y capacidad del módulo. La comparación aporta evidencia sobre estas implementaciones, no una atribución causal exclusiva a la base de Fourier.
