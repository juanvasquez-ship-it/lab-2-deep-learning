# Referencia metodológica y decisiones experimentales

La referencia es Jia et al., *Enhancing streamflow forecasting using an LSTM hybrid model with lightweight frequency-domain feature learning*, ESWA 297 (2026), 129418, DOI 10.1016/j.eswa.2025.129418. Se reproducen sus preguntas experimentales con los datos del laboratorio, sin afirmar identidad con sus configuraciones ni sus resultados numéricos.

## Protocolo comparable

Todos los modelos aprenden con las mismas 254.000 ventanas de entrenamiento y se seleccionan con las mismas 18.142 ventanas de validación. Cada muestra contiene 336 horas históricas y el objetivo tiene 48 caudales horarios, en mm/h. Las particiones entregadas se conservan. Se exportan las 27.983 predicciones de test, cuyos objetivos no se utilizan porque el papel de `test_targets.csv` no está confirmado.

La normalización, la selección LASSO y la selección de canales auxiliares para robustez se ajustan exclusivamente con entrenamiento. Los cinco modelos principales y la ablación usan los mismos doce canales. El control univariado utiliza únicamente el canal 11, conservando las estadísticas de ese canal y del objetivo.

Se mantiene MSE, Adam, tasa 10⁻⁴, lotes de 64, máximo de 200 épocas, paciencia de cinco épocas y selección del checkpoint con menor MSE de validación. Se usan todos los datos en cada época, semilla 42, el mismo orden de lotes por época y recorte de norma de gradiente a 1. Los números de épocas finales pueden diferir porque el criterio de parada es equivalente, no porque se asigne un presupuesto menor a algún modelo.

El tronco de pronóstico es una LSTM unidireccional de dos capas de 128 unidades y una proyección lineal a 48 salidas. Se inicializa antes de los módulos de características para que los cinco modelos principales y la ablación compartan exactamente los pesos iniciales de ese tronco. El LSTM univariado tiene una matriz de entrada de tamaño distinto, por lo que comparte semilla y tamaño oculto, pero no se afirma identidad de toda su inicialización.

La precisión mixta se usa donde es compatible. La FFT y la normalización del grafo permanecen en FP32 por estabilidad y compatibilidad. La inferencia final conserva predicciones crudas, y exporta máximo(predicción, 0) para imponer caudal no negativo. Las métricas crudas y recortadas se presentan separadamente.

## Modelos y arquitecturas

| Modelo | Referencia del paper | Implementación adaptada | Razón y alcance |
|---|---|---|---|
| LSTM | Sección 3.3, modelo base | Entrada directa de doce variables al tronco común | Aísla el aporte de los módulos de características |
| AM-LSTM | Ecuaciones 2–5, atención entre variables con estados recurrentes | Contexto BiLSTM de 32 unidades por dirección, proyecciones del contexto anterior y entrada anterior, tanh, softmax sobre canales, multiplicación por la entrada actual | El artículo menciona estados en ambos sentidos, pero no detalla su tamaño en el texto principal. El contexto adicional tiene tamaño acotado y solo procesa la historia disponible |
| CNN-LSTM | Ecuaciones 6–7, Conv1D sobre características y pooling global | Dos convoluciones de kernel 3 y padding 1, mapas 1→32→12, ReLU y promedio sobre el eje de variables en cada hora | Mantiene convolución y pooling del paper. Los doce mapas finales permiten un tronco de entrada común, sin confundir la convolución entre canales con una convolución temporal |
| GNN-LSTM | Ecuación 8, adyacencia aprendida, simetría, no negatividad y autoconexiones | Doce nodos, uno por variable, adyacencia softplus simetrizada, I y normalización D⁻¹ᐟ²AD⁻¹ᐟ², dos capas 1→16→1 con ReLU | No se inventan coordenadas ni enlaces geográficos entre cuencas. Se aprende un grafo de variables común a las ventanas, conservando una salida por variable |
| FDMLP-LSTM | Ecuaciones 11–15 | rFFT sobre variables, dos transformaciones complejas elemento a elemento, CReLU intermedia, irFFT y tronco LSTM | La ecuación 11 especifica productos elemento a elemento. La simetría conjugada implícita produce una salida real. La FFT opera sobre doce canales, no sobre las 336 horas |
| FDMLP sin CReLU | Ablación del componente no lineal para el laboratorio | Mismo bloque y parámetros que FDMLP, sin CReLU | Aísla la no linealidad. El bloque afín resultante puede absorberse en las proyecciones de entrada de la LSTM, por lo que diferencias frente al baseline pueden deberse a la optimización |
| LSTM univariado | Sección 5.1, entradas univariadas frente a multivariadas | Mismo tronco con solo caudal histórico | Mide el aporte conjunto de los once canales meteorológicos, con las mismas muestras y objetivos |

La atención bidireccional no accede a las 48 horas objetivo, solo resume la ventana histórica completa disponible al emitir el pronóstico. CNN y FDMLP dependen del orden de canales declarado en los metadatos, ese orden se conserva para todos los ejemplos. El grafo representa asociaciones aprendidas, no relaciones causales demostradas.

El texto principal de AM no especifica por completo la red que produce sus estados auxiliares, por ello el BiLSTM de contexto se declara como una adaptación. Las anchuras auxiliares se fijaron antes de observar los resultados nuevos, para mantener un coste moderado sobre doce variables y un tronco comparable, sin afirmar que sean los mejores valores de cada familia.

Los tamaños auxiliares figuran en `config.json`. El grafo inicializa sus conexiones con parámetros normales de media −2 y desviación 0,1 antes de softplus, y la proyección final con pesos 1/16 para evitar una salida completamente apagada por ReLU. Las demás capas auxiliares usan la inicialización predeterminada de PyTorch. FDMLP conserva pesos reales uno y partes imaginarias y sesgos cero. Estas elecciones son decisiones explícitas de implementación, no hiperparámetros atribuidos a los autores.

El suplemento con la búsqueda de hiperparámetros no estuvo disponible. No se ejecuta una búsqueda aleatoria distinta por modelo ni se presentan estas configuraciones como óptimos de cada arquitectura. El tamaño oculto común fija una comparación de referencia con aproximadamente 0,2 millones de parámetros, los tamaños publicados para los experimentos originales son distintos. Cambiar de dataset y representación impide transferir directamente sus conteos de parámetros.

## Correspondencia de los experimentos

### Controles del módulo y de su inicialización

Se mantiene el modelo FDMLP original y su ablación sin CReLU. Esta última evalúa la activación, no el beneficio específico de Fourier. El bloque sin CReLU es afín y puede absorberse en la primera proyección de entrada de la LSTM; el bloque con CReLU no inicia como identidad. Una diferencia entre entrada y salida mide cambio de representación, no una fracción de información predictiva perdida. Estos controles se plantean después de evaluar la versión inicial, por lo que constituyen análisis exploratorios sobre la misma validación y no una confirmación independiente.

Se añaden tres controles propios, separados de los cinco modelos atribuidos al paper:

| Control | Implementación | Qué permite estudiar |
|---|---|---|
| MLP-LSTM | Linear(12,12) → ReLU → Linear(12,12), aplicado a cada hora | Referencia no lineal en el dominio original, sin Fourier |
| FDMLP residual | x + FDMLP(x), segunda capa compleja inicializada con peso real 0,01, imaginario y sesgos cero | Estrategia conjunta de camino directo e inicio cercano a identidad |
| MLP residual | x + MLP(x), salida inicializada a 0,01 I y sesgos cero | Misma estrategia de preservación inicial con un módulo real |

Los MLP usan peso identidad y sesgo cero en su primera capa; sin residual, su segunda capa también inicia con identidad. Las capas se entrenan sin restringirse a matrices identidad o diagonales. La escala 0,01 se fija antes del entrenamiento, para que la corrección inicial sea pequeña y haya gradientes hacia la primera capa. No es un hiperparámetro atribuido a los autores ni ajustado a validación. La conexión residual y su inicialización cambian conjuntamente; no se atribuye el efecto observado solo a una de ellas.

Se conservan tronco LSTM, inicialización del tronco, normalización, datos, batch, tasa, pérdida, semilla 42 y parada temprana. El módulo real tiene 312 parámetros y el frecuencial 56, una diferencia de 256, alrededor del 0,12% del modelo completo. Esa cercanía global no implica capacidad idéntica de los módulos. Las conexiones densas reales son una referencia MLP convencional; el FDMLP conserva el producto elemento a elemento explícito de la ecuación 11. Su comparación no constituye una prueba aislada de la base de Fourier.

Los 65 componentes de la figura 15 no identifican de forma única una proyección Linear(12,128). El suplemento no disponible impide confirmar esa arquitectura. Por ello no se incorpora esa proyección como una supuesta corrección obligatoria. Tampoco se exige invariancia ante cualquier permutación: la dependencia del orden es una limitación de la parametrización, y un ensayo con pesos fijos no equivale a comparar modelos reentrenados con órdenes diferentes.

Se ejecuta primero una semilla, conservando las ejecuciones previas. La decisión sobre repeticiones adicionales se documenta con los tiempos medidos en `resultados/presupuesto_semillas.json`. El número de repeticiones se decide por coste, sin usar qué modelo gana como criterio de continuación. Las 60 condiciones de robustez y el benchmark original corresponden a los cinco comparadores originales y sus controles previos, no se atribuyen automáticamente a las nuevas variantes.

### Estudios del artículo

| Estudio del artículo | Estudio con este dataset | Adaptación necesaria |
|---|---|---|
| Cinco modelos, tabla 3 | LSTM, AM-LSTM, CNN-LSTM, GNN-LSTM y FDMLP-LSTM | Mismas ventanas horarias y mismo tronco, además de controles univariado, ablación y persistencia |
| Tres estaciones | Evaluación global y por cada una de las 508 cuencas | No hay equivalencia física entre estaciones originales y cuencas anónimas |
| Pronósticos a 1, 3 y 5 días | Métricas por cada hora y acumuladas para 1–12, 1–24 y 1–48 h | Solo existen 48 objetivos horarios. Se entrena una salida de 48 horas por modelo y se evalúan sus prefijos, sin construir objetivos posteriores ni fingir entrenamientos separados por horizonte |
| Historias de varios días y búsqueda de configuración | Historia fija de 336 horas | Respeta las entradas del laboratorio. No hay fechas para reconstruir ventanas de 30–365 días concatenando muestras |
| NSE, RMSE y R | Las mismas métricas, además de MAE, sesgo y resúmenes por cuenca | RMSE en mm/h, no en m³/s. Se explicitan métricas indefinidas y el efecto de cuencas casi constantes |
| FHV y FLV | Curvas ordenadas, 2% superior y 30% inferior | Se fija una convención publicada y suelo logarítmico, con análisis de sensibilidad; no se equiparan al sesgo de máscaras condicionales |
| Tiempo y parámetros, tabla 5 | Tiempos reales por época y benchmark del mismo lote | Hardware, carga externa y tamaño de dataset difieren. Se comparan medianas de 20 repeticiones tras cinco de calentamiento, sin I/O en el benchmark |
| Univariado/multivariado, figura 13 | LSTM con canal 11 frente a LSTM con doce canales | Mismo objetivo y mismo protocolo, matriz de entrada distinta por necesidad |
| Ruido y faltantes, figura 14 | Desviaciones 0,1/0,5/1 y tasas 5/10/20%, toda la validación | Ruido en unidades estandarizadas, realizaciones compartidas y máscaras anidadas, interpolación histórica, sin reentrenar |
| Perturbación de auxiliares relevantes | Mismas pruebas en los tres auxiliares de mayor coeficiente LASSO absoluto | Se usa una regla de selección calculada con entrenamiento, no el rendimiento en validación |
| Gradientes frecuenciales, figura 15 | Gradiente absoluto complejo medio en los siete bins rFFT, toda la validación | El espectro corresponde a doce variables y no a la representación de 65 bins mostrada en el artículo. No se infiere periodicidad temporal o causalidad física |

## Alcance de las perturbaciones

En robustez se fijan los pesos y se degradan solo las entradas de evaluación, para aislar la sensibilidad a sensores o datos de entrada alterados. El texto principal no detalla completamente la fase de aplicación de todas las perturbaciones, así que esta decisión se registra explícitamente. El ruido gaussiano no se recorta según restricciones físicas de cada canal, es una prueba artificial de estrés en unidades normalizadas, no una afirmación sobre la distribución real del error de los sensores.

## Extremos y regularización de FLV

Se ordenan por separado las observaciones y las predicciones no negativas. FHV es 100 × [suma(predichos superiores) − suma(observados superiores)] / suma(observados superiores). Se usa ceil(0,02 N), con al menos un valor.

Para FLV se toman ceil(0,30 N) valores inferiores de cada curva. Se calcula A = suma[log(Q observado) − log(Q observado mínimo)] y B de igual forma para la curva predicha. FLV = −100 × (B−A)/A. Si A es cero, FLV queda indefinido. El suelo para logaritmos es 10⁻⁶ mm/h, y se conserva una sensibilidad con 10⁻⁸ y 10⁻⁴ mm/h. Este suelo es una decisión numérica, no un límite de detección conocido del instrumento.

La referencia de estas fórmulas es [NeuralHydrology, métricas FDC](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.evaluation.metrics.html), basada en Yilmaz, Gupta y Wagener (2008), DOI 10.1029/2007WR006716. No se instala ni se reutiliza una implementación de redes de esa biblioteca. Un FLV grande puede ser muy sensible a los ceros, por lo que se presenta junto con métricas absolutas y sensibilidad al suelo logarítmico.

## Qué conclusiones permiten los experimentos

Los resultados permiten comparar las configuraciones entrenadas bajo las mismas condiciones y estudiar errores, coste y sensibilidad. No prueban superioridad universal de una familia, equivalencia con los resultados originales, independencia temporal entre ventanas ni significancia entre semillas. La falta de fechas, áreas y coordenadas limita la interpretación hidrológica y la conversión de unidades. El test permanece sin evaluación supervisada.

La validación también se utilizó para elegir checkpoints mediante parada temprana, por lo que sus métricas pueden ser optimistas y no sustituyen una evaluación externa independiente. Esta restricción se mantiene igual para todos los modelos y debe acompañar las conclusiones frente al conjunto de test del artículo.

Se mantienen los resultados originales de LSTM, FDMLP y su ablación. Añadir nombres de modelos a la lista de experimentos no cambia su configuración efectiva ni justifica repetirlos. Una prueba específica verificó que sus checkpoints siguen produciendo las mismas predicciones después de ampliar el código. Los resultados nuevos no se usan para ajustar retrospectivamente las configuraciones de sus comparadores.
