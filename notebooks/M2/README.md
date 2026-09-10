# Harness de Evaluación — Extractor de Perfil de Estilo (M2)

Harness en 3 dimensiones para evaluar al extractor de perfil de estilo del personal-shopper.
El punto de entrada es `harness(eval_set, system, judge_backend)`, que ejecuta las 3
dimensiones sobre un sistema y devuelve un **scorecard** consolidado.
El archivo `data/scorecard_baseline.csv` es el scorecard del **baseline** (Qwen2.5-1.5B-Instruct
zero-shot, sin LoRA) sobre el mismo `data/eval_set.json`.

| Archivo | Descripción |
|---|---|
| `harness.ipynb` | Notebook con el harness completo (3 dimensiones, juez local Qwen, modo demo) |
| `data/eval_set.json` | 12 casos de oro (8 `estandar`, 2 `ambigua`, 2 `trampa_out_of_domain`) |
| `data/scorecard_baseline.csv` | Scorecard del baseline |

---

## 1. Las 3 dimensiones

| Dimensión | Qué mide | Métricas |
|---|---|---|
| **D1** — Métricas clásicas | Coincidencia literal entre la salida y el gold | JSON válido, *exact-match*, F1 macro y F1 por campo |
| **D2** — LLM como juez | Calidad semántica con rúbrica 1–5 (con anclas) | Puntaje promedio y distribución por estrella |
| **D3** — Cumplimiento de criterios del campo | Fracción de respuestas que cumplen el *criterion* definido por caso en el eval_set | Cumplidos/total (con umbral `judge >= 4` y regla dura para `{}`) |

Detalles de diseño importantes:
- **Exact-match es estricto por diseño**: el JSON debe ser *igual* al gold. Omitir un campo no
  mencionado es **correcto**; inventar uno, aunque sea plausible, rompe el match (0).
- **D3 es la métrica del campo**: el *criterion* de cada caso es la definición operativa de "buena
  respuesta" que se fijó. Incluye una regla en contra de la alucinación para los casos ambiguos (Las trampas)
  out-of-domain: si el gold es `{}`, la respuesta debe ser JSON vacío (o `None`), sin importar lo
  que diga el juez.
- **El juez es local**: por defecto reutiliza el M1 afinado ya cargado para no tener que cargar un modelo nuevo,
  con opción de usar `Qwen/Qwen2.5-0.5B-Instruct` o el minijuez determinista.

---

## 2. Uso

1. Entrenar M1 con `notebooks/M1/01_finetuning_lora_qwen25_oficial.ipynb` y guardar el adaptador en
   `BASE_DIR/qwen25-1.5b-style-extractor-lora/adapter_final`.
2. En Colab (T4), abrir `harness.ipynb`, montar Drive y ejecutar el harness luego de tener todos los archivos requeridos en el Drive (adapter, eval_set)

---

## 3. Análisis de resultados: `scorecard_baseline.csv`

Resultados del baseline zero-shot (Qwen2.5-1.5B-Instruct **sin** LoRA, juez = M1 reutilizado) sobre
los 12 casos de `data/eval_set.json`.

### 3.1 Números agregados

| Dimensión | Resultado |
|---|---|
| **D1** JSON válido | **100%** |
| **D1** Exact-match | **8.3%** (1/12) |
| **D1** F1 macro | **0.644** |
| **D1** F1 por campo | estilo=0.583 · ocasion=0.476 · clima=0.750 · paleta=0.744 · fit=0.667 |
| **D2** Puntaje juez | **4.5 / 5** |
| **D3** Cumplimiento | **83.3%** (10/12) |

Contraste clave: el modelo produce **JSON sintácticamente perfecto el 100% de las veces** (supera
D1-JSON y parece "plausible" para el juez, D2=4.5), pero solo **1 de cada 12 respuestas coincide
literalmente con el esperado** (D1=8.3%). El formato nunca es el problema: lo es la *fidelidad del
contenido*.

### 3.2 ¿En qué casos falla?

El detalle por caso (`d1_exact_match`, `d2_judge`, `d3_cumple_criterio` en el CSV):

| Caso | Tipo | Error principal | D1 exact | D2 | D3 |
|---|---|---|---|---|---|
| eval_01 | estandar | Acierta los 4 campos del gold pero **inventa `clima=calido`** | 0 | 4 | ✓ |
| eval_02 | estandar | Acierta los 4 campos del gold pero **inventa `estilo=Formal`** | 0 | 5 | ✓ |
| eval_03 | estandar | **`ocasion=viaje`** en vez de `deporte` (acierta los otros 4) | 0 | 5 | ✓ |
| eval_04 | estandar | Acierta los 4 campos del gold pero **inventa `estilo=Bohemio`** | 0 | 5 | ✓ |
| eval_05 | estandar | Acierta los 3 campos del gold pero **inventa `clima=templado`** | 0 | 5 | ✓ |
| eval_06 | ambigua | **Inventa el JSON completo**: `ocasion=viaje` (¡ni siquiera `fin_de_semana`!) + 4 campos | 0 | 1 | ✗ |
| eval_07 | trampa | `{}` → `{}` (correcto) | **1** | 5 | ✓ |
| eval_08 | trampa | **Alucina un JSON de 5 campos** fuera de dominio (talla/presupuesto/descuentos) | 0 | 5 | ✗ |
| eval_09 | ambigua | Acierta `estilo=Urbano` pero falla `ocasion=viaje` (gold `deporte`) + inventa 3 campos | 0 | 5 | ✓ |
| eval_10 | estandar | **`fit=holgado`** en vez de `oversized` (texto: "muy holgada y ancha") + inventa `ocasion` | 0 | 5 | ✓ |
| eval_11 | estandar | **`paleta=neutros`** y **`estilo=Clasico`** (gold `oscuros`/`Formal`) + inventa `clima` y `fit` | 0 | 4 | ✓ |
| eval_12 | estandar | **`ocasion=viaje`** en vez de `deporte` + inventa `clima=calido` | 0 | 5 | ✓ |

**Categorías de error (por orden de frecuencia):**

1. **Relleno / alucinación de campos no señalados (10 de 12 casos).** El baseline emite *siempre*
   un JSON completo de 5 campos aunque el mensaje no mencione clima, estilo u ocasión (eval_01,
   eval_02, eval_04, eval_05, eval_06, eval_09, eval_10, eval_11, eval_12) o esté fuera del dominio
   (eval_08). Ningún caso estándar omitió un campo *porque el texto no daba señal*: el modelo no
   conoce la regla de "extracción mínima" que exige el eval_set.
2. **Confusión semántica entre valores del mismo campo.** `holgado` vs `oversized` (eval_10),
   `Clasico` vs `Formal` (eval_11), `neutros` vs `oscuros` (eval_11). Son estos deslices los que D1
   castiga y D2 perdona.
3. **Patrón de ocasión "por defecto": `viaje`.** El baseline asigna `ocasion=viaje` en eval_03,
   eval_06, eval_09 y eval_12 (donde el gold era `deporte` o `fin_de_semana`). Es su *prior* favorito
   cuando la señal es débil, y es la causa de que **ocasión sea el campo con peor F1 (0.476)**.
4. **Falta de abstención.** Hacer siempre un JSON válido (no `{}`, no respuesta mínima) es funcional
   aquí: los dos casos que exigen abstenerse fallan (eval_06 y eval_08).

### 3.3 ¿Qué pasó con los casos ambiguos? (eval_06 y eval_09)

Son la mayor brecha del baseline y muestran que **el problema no es detectar, sino abstenerse**:

- **eval_06** (señal mínima: "cómodo y lindo, no me importa el color"): el gold exige solo
  `{"ocasion": "fin_de_semana"}`. El modelo no solo rellena `estilo/clima/paleta/fit` — **falla
  incluso `fin_de_semana`** y emite `viaje`. Resultado D2=1/5 y D3=0 (los únicos en toda la tabla
  junto a eval_08). Es el caso más grave: cero extracción y cero abstención.
- **eval_09** (contradicción: "gimnasio" pero "no tan deportivo, más de calle"): acierta la intención
  corregida (`estilo=Urbano`) pero inventa `ocasion=viaje` y 3 campos extra. Aquí el **juez lo
  aprueba con 5/5**, pasando por alto la ocasión equivocada y los inventos — un ejemplo claro de la
  lenidad del juez frente a respuestas "plausibles".

**Conclusión sobre ambigüedad:** sin un mecanismo de confianza/abstención, el modelo degrada a su
*prior* (el "template" `viaje/Urbano/calido/monocromatico/regular` que se repite en casi todas las
respuestas inventadas). Además, el juez es **inestable** aquí: castiga bien eval_06 (1) y aprueba mal
eval_09 (5). Si solo se mirara D2 (4.5/5), la ambigüedad parecería resuelta.

### 3.4 Las trampas out-of-domain (eval_07 y eval_08)

- **eval_07** ("envío y talla 32"): responde `{}` correctamente (el único exact-match).
- **eval_08** ("chaqueta de cuero, presupuesto y descuentos"): **alucina un JSON completo** (diseña
  `estilo=Formal`, `paleta=neutros`, etc.). El juez lo aprueba 5/5, **pero la regla dura de D3 para
  `{}` lo captura** (cumple=0). Heredar de la palabra casual "mediana" o "presupuesto" campos del
  dominio de moda es exactamente el tipo de invención que la regla previene.

Este par demuestra el valor de combinar D1 + D3: el juez (D2) solo no basta.

### 3.5 ¿Qué dimensión es la más importante y por qué?

**D3 (cumplimiento de criterios del campo) es la dimensión principal**, por tres razones:

1. **Es la definición operativa del negocio.** Cada *criterion* del eval_set codifica lo que el
   equipo considera "buena respuesta" (extracción mínima, priorizar la intención corregida del
   usuario, respetar `{}` fuera de dominio, resolver cruces estilo-ocasión). D1 y D2 solo son
   aproximaciones a ese criterio.
2. **Incluye la regla dura anti-alucinación** (respuesta vacía para trampas `{}`), que es la
   salvaguarda que D2 no tiene: sin ella, eval_08 habría pasado como válido.
3. **Es comparable y accionable entre sistemas**: baseline 83.3% → objetivo M1 consistentemente
   mayor, con la misma regla y el mismo juez.

Pero D3 **no puede ir sola**: hereda la lenidad de su componente juez (eval_09 se aprueba con 5/5
pese a campos equivocados e inventados). Por eso:
- **D1 (exact-match 8.3%) es el complemento estricto**: detecta la sobre-generación y las difusas
  diferencias campo a campo que el juez perdona. Su dureza es intencional.
- **D2 (4.5/5) es diagnóstico, no veredicto**: es *consistente con D1* para condenar lo grave
  (eval_06 → 1/5), pero complaciente con lo "plausible". Debe leerse con sospecha cuando los campos
  inventados son estilísticamente aceptables.

---

## 4. Qué se puede mejorar, por qué ocurre y cómo ocurre

### 4.1 ¿Por qué ocurre?

1. **El modelo base maximiza el JSON plausible completo, no la fidelidad al texto.** En zero-shot,
   sin ejemplos de omisión, su distribución preferida es rellenar los 5 campos con los valores más
   probables. No tiene representado el patrón "omitir si no hay señal" que sí enseña el fine-tuning.
2. **Ontología cerrada sin salida de abstención.** Al tener una lista fija de valores por campo, el
   modelo "completa" cualquier mensaje dentro de una de las categorías; nunca emite `{}` ni un JSON
   mínimo, salvo cuando la entrada coincide con algo fuera de su distribución (eval_07).
3. **Un único juez LLM, reutilizado y complaciente.** El juez (M1 afinado) valora la *plausibilidad
   estilística*, no la *fidelidad campo a campo*, por eso D2 (4.5) y D1 (8.3) difieren tanto.
4. **No hay mecanismo de confianza ni verificación a posteriori.** Nada detiene ni corrige la
   alucinación de un campo no señalado (p. ej. inventar `clima` por inercia).

### 4.2 ¿Cómo ocurre (mecanismo)?

Durante la generación, ante una señal débil el modelo cae en el caso mas probable: el patrón repetido `ocasion=viaje · estilo=Urbano · clima=calido · paleta=monocromatico ·
fit=regular` aparece literalmente en eval_06, eval_09 y eval_12 (y parcialmente en otros). La abstención no ocurre porque no existe un token/ruta que la produzca:
el prompt no la pide y el modelo nunca la vio en sus datos.

### 4.3 Mejoras concretas

| # | Mejora | Por qué la ataca | Cómo implementarla |
|---|---|---|---|
| 1 | **Extracción mínima en el prompt** | Relleno de campos no señalados = el error más frecuente (10/12) | Instruir: *"incluye solo campos con señal explícita; si la señal es dudosa, omítelo; si es fuera de dominio, devuelve `{}`"* + 2–3 ejemplos few-shot de JSON mínimo y de `{}` |
| 2 | **Fine-tuning con golds parciales y `{}`** | La distribución del base no sabe omitir | Alimentar LoRA también con ejemplos cuyo gold tenga 1–2 campos y con trampas `{}` (el `data/eval_set.json` ya los tiene; ampliar en train/val) |
| 3 | **Predictor de confianza / selección selectiva** | No existe abstención | Medir la probabilidad token de cada campo o un clasificador auxiliar y descartar campos bajo un umbral calibrado **con este eval_set** |
| 4 | **Reglas duras de dominio post-generación** | El LLM autoconsistente no se detiene solo | Detectar señales de no-dominio (talla numérica, presupuesto, envío, descuento) y forzar `{}` o filtrar campos inventados antes de emitir el JSON |
| 5 | **Juez más estricto (D2)** | D2 es complaciente (4.5 vs 8.3 de D1) y aprobó eval_08/09 mal | Pedir al juez listar **campo a campo gold vs predicción** antes de puntuar, usar 2+ jueces con votación, y validación humana por muestreo (p. ej. 20% de los casos) |
| 6 | **Ampliar ambigüedad en el eval_set** | La ambigüedad es la mayor brecha y el juez es inestable ahí | Agregar más casos con señal mínima y contradicciones, y permitir golds múltiples aceptables para medir la abstención con menos ruido |

---

## 5. Lectura honesta y limitaciones

- **D1 del harness no es comparable con el del val set** En `val.jsonl` el
  baseline da ~20% exact-match/0.688 F1; aquí da 8.3%/0.644 porque el eval_set es intencionalmente
  más difícil (33% de casos trampa/ambiguos) y el exact-match exige *igualdad de JSON*, incluido el
  "JSON mínimo correcto" (omisión deliberada).
- **El juez evaluado es el propio M1 afinado**, no un anotador externo; su lenidad (sección 3.3)
  puede variar con el modelo juzgador elegido. Los resultados de D2 deben tratarse como opinión
  asistida, no como veredicto.
- **n=12 es una muestra pequeña** para distinguir mejoras finas; usar este harness para *diagnóstico
  cualitativo* (patrones de error) y complementar con la evaluación masiva de val.