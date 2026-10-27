# Harness de Evaluación — Extractor de Perfil de Estilo (M2)

Harness en 3 dimensiones para evaluar al extractor de perfil de estilo del personal-shopper.
El punto de entrada es `harness(eval_set, system, judge_backend)`, que ejecuta las 3
dimensiones sobre un sistema y devuelve un **scorecard** consolidado.
El archivo `data/scorecard_baseline.csv` es el scorecard del **baseline** (Qwen2.5-1.5B-Instruct
zero-shot, sin LoRA) sobre el mismo `data/eval_set.json`.

| Archivo | Descripción |
|---|---|
| `harness_3.ipynb` | Notebook con el harness completo (3 dimensiones, control de semillas, juez independiente 7B) |
| `data/eval_set.json` | 12 casos de oro (8 `estandar`, 2 `ambigua`, 2 `trampa_out_of_domain`) |
| `data/scorecard_baseline.csv` | Scorecard del baseline actualizado |

---

## 1. Las 3 dimensiones y mitigación de sesgos

| Dimensión | Qué mide | Métricas y Mitigaciones |
|---|---|---|
| **D1** — Métricas clásicas | Coincidencia literal entre la salida y el gold. | JSON válido, *exact-match*, F1 macro (calculado solo sobre etiquetas reales). |
| **D2** — LLM como juez | Calidad semántica con rúbrica 1–5 (con anclas). | **Mitigación de auto-preferencia:** Se usa un modelo distinto y superior (`Qwen2.5-7B-Instruct` en 4-bit) al evaluado para evitar complacencia. |
| **D3** — Cumplimiento de criterios del campo | Fracción de respuestas que cumplen el *criterion* definido en el eval_set. | **Validación Híbrida:** Exige puntaje del juez >= 4 **Y** validación estricta en Python (intersección de llaves) para penalizar cualquier campo inventado. |

**Garantía de Reproducibilidad:** Todo el arnés ejecuta un pipeline determinista fijando semillas globales (`torch.manual_seed(42)`) y utilizando *greedy decoding* (`do_sample=False`).

---

## 2. Mitigación del sesgo del Juez (Auto-preferencia)

Un error común en la evaluación LLM-as-a-judge es la **auto-preferencia**: usar el mismo modelo para generar y para evaluar. En versiones tempranas de este arnés, evaluar el baseline de 1.5B con el propio modelo de 1.5B arrojaba un puntaje promedio ilusorio de **4.5 / 5**. El modelo perdonaba sus propias alucinaciones (inventar campos no señalados).

**¿Cómo lo mitigamos?** 
Implementamos el switch `JUEZ_LOCAL_APARTE = True` que carga un modelo significativamente más capaz (`Qwen/Qwen2.5-7B-Instruct`) cuantizado en 4-bit (`BitsAndBytesConfig`) para que quepa en la RAM de Colab T4. Este nuevo juez redujo el puntaje del baseline a un realista **3.17 / 5**, demostrando ser mucho más estricto y menos sesgado hacia las salidas sobre-generadas del modelo base.

---

## 3. Análisis de resultados: `scorecard_baseline.csv`

Resultados del baseline zero-shot (Qwen2.5-1.5B-Instruct sin LoRA) evaluado por Qwen2.5-7B-Instruct.

### 3.1 Números agregados

| Dimensión | Resultado |
|---|---|
| **D1** JSON válido | **100.0%** |
| **D1** Exact-match | **16.7%** (2/12) |
| **D1** F1 macro | **0.792** |
| **D1** F1 por campo | estilo=0.667 · ocasion=0.733 · clima=1.000 · paleta=0.893 · fit=0.667 |
| **D2** Puntaje juez (7B) | **3.17 / 5** |
| **D3** Cumplimiento estricto | **16.7%** (2/12) |

**El hallazgo principal:** La Dimensión 3 pasó de un falso >80% a un **16.7%**. Al integrar validación dura en código (restando las llaves del gold contra las llaves de la salida), la métrica D3 por fin intercepta la falla sistemática del baseline: **la alucinación de campos por sobre-generación**. 

### 3.2 ¿En qué casos falla? (La caída de D3)

El baseline emite *siempre* un JSON completo de 5 campos. Antes, el juez pasaba esto por alto. Ahora, la validación estricta de D3 lo castiga automáticamente.

| Caso | Tipo | Error principal | D1 (Exact) | D2 (Juez) | D3 (Cumple) |
|---|---|---|---|---|---|
| eval_01 | estandar | Acierta 4 campos, **omite clima (inventa 'calido'/'templado' a veces)** pero aquí coincidió exacto por suerte | 1 | 5 | ✓ |
| eval_02 | estandar | Acierta los 4 campos del gold pero **alucina `estilo`** | 0 | 4 | ✗ FAIL (Python detectó invento) |
| eval_03 | estandar | Falla `ocasion` y el juez 7B lo castiga | 0 | 2 | ✗ FAIL |
| eval_04 | estandar | Acierta 4 campos pero **alucina `estilo`** | 0 | 4 | ✗ FAIL (Python detectó invento) |
| eval_06 | ambigua | El gold pide 1 campo, el modelo genera 5. | 0 | 4 | ✗ FAIL (Python detectó invento) |
| eval_07 | trampa | `{}` → `{}` (Se abstuvo correctamente) | 1 | 5 | ✓ |
| eval_08 | trampa | **Alucina un JSON completo** en vez de `{}`. El juez 7B lo liquida. | 0 | 1 | ✗ FAIL |

**Conclusión clave:** El juez 7B aprueba la semántica (D2=4) de casos como `eval_02` y `eval_04` porque los campos extra tienen sentido de moda, pero **D3 los rechaza** porque violan la regla de extracción mínima. D1 y D3 ahora están alineados en penalizar la sobre-generación.

### 3.3 Las trampas out-of-domain (eval_07 y eval_08)

El contraste definitivo:
- **eval_07** ("envío y talla 32"): Responde `{}` y logra el codiciado Exact-Match y D3.
- **eval_08** ("presupuesto y descuentos"): Inventa 5 campos basándose en ruido. 

Aquí el nuevo juez 7B demostró su valor: mientras el viejo juez M1 le daba 5/5 a `eval_08`, el nuevo juez lo castiga con un **1/5** por alucinar categorías a partir de palabras sobre presupuesto.

---

## 4. Qué se puede mejorar en M3 (RAG) y M5

El baseline falla sistemáticamente porque **maximiza el JSON plausible completo, no la fidelidad al texto**. No sabe abstenerse ni omitir campos. 

Otro problema identificado es la existencia de único juez LLM, reutilizado y complaciente. El juez (M1 afinado) valora la plausibilidad estilística, no la fidelidad campo a campo, por eso D2 (4.5) y D1 (8.3) difieren tanto.

**Mejoras concretas para las siguientes fases:**

1. **Fine-tuning con ejemplos negativos:** El M1 (LoRA) debe entrenarse con datos donde el JSON esperado solo tenga 1 o 2 llaves, para romper el *prior* de "responder siempre los 5 campos".
2. **Selector de Relevancia (RAG):** Para casos como eval_06 (ambiguo) o eval_08 (fuera de dominio), el sistema necesita un paso previo (un clasificador o retrieval) que dictamine si hay suficiente señal de moda antes de siquiera llamar al extractor.
3. **Mecanismo de fallback:** Implementar una regla dura (heurística) pre-generación que atrape palabras como "talla", "envío", "precio" y emita un `{}` inmediato sin gastar inferencia.

## 5. Lectura honesta del Scorecard

Construir una Dimensión 3 que combinara al Juez (D2) con reglas en Python fue doloroso porque desplomó nuestras métricas aparentes (de un cumplimiento >80% a solo 16.7%). Sin embargo, es el hito más importante del proyecto. 

Ahora tenemos una herramienta que **no se miente a sí misma**. D1 mide la forma estricta, D2 evalúa que los valores tengan sentido en el mundo real, y D3 vigila implacablemente que no se inventen cosas. Cualquier mejora de un 1% que logremos en D3 a partir de hoy, será una mejora real y defendible del producto.
