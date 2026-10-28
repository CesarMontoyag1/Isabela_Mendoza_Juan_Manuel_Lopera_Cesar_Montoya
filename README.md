# Personal Shopper de Moda — Proyecto SI4006

**¿De qué trata este proyecto?** Construimos, por módulos, el cerebro de un asistente de moda: un
sistema que lee lo que una persona pide con sus propias palabras ("quiero algo bien ancho y relajado
para el frío") y la ayuda a vestirse: primero entiende su intención, luego recomienda prendas reales
de un catálogo, y en cada paso se mide con una vara de evaluación honesta.

El proyecto se construye en capas que no se descartan: cada módulo se apoya en el anterior.

| Módulo | Qué aporta | Carpeta |
|---|---|---|
| **M1** | Modelo afinado que **extrae el perfil de estilo** del mensaje del usuario | `notebooks/M1/` |
| **M2** | **Harness de evaluación** (la vara con la que medimos todo lo demás) | `notebooks/M2/` |
| **M3** | **Sistema RAG agéntico** que recomienda outfits reales del catálogo | `notebooks/M3/` |

---

## M1 — Fine-Tuning Qwen2.5-1.5B para extracción de perfil de estilo

**Qué hace:** lee un mensaje de chat en español ("me voy a una boda y quiero verme elegante pero
clásico, con corte holgado y tonos neutros") y extrae un JSON con los atributos de moda (`estilo`,
`ocasion`, `clima`, `paleta`, `fit`) según una ontología cerrada.

- **Modelo base:** `Qwen/Qwen2.5-1.5B-Instruct` (familia Qwen), elegido por su capacidad multilingüe
  en español natural con abreviaciones y su rendimiento en seguimiento de instrucciones estructuradas.
- **Baseline:** el mismo modelo en *zero-shot* (sin afinar), para aislar el impacto de LoRA.
- **Técnica:** fine-tuning con LoRA (`r=16`, `alpha=32`, `dropout=0.05` sobre `q/k/v/o_proj`).

| Modelo | Exact-match | F1 macro |
|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | 20.0% | 0.688 |
| Qwen2.5-1.5B-Instruct + LoRA | **66.2%** | **0.752** |

**Lectura honesta:** el afinamiento triplicó el exact-match (20% → 66.2%). El modelo base
"alucinaba" rellenando categorías no pedidas (p. ej. `ocasion: viaje` por defecto); el fine-tuning le
enseñó a extraer solo lo presente en el texto. Sigue fallando en ~34% de los casos exactos ante
descripciones muy ambiguas o jerga marcada.

El dataset de entrenamiento es sintético, generado con Gemini (ver `data_generation/`).

---

## M2 — Harness de evaluación + scorecard del baseline

**Qué hace:** es la **vara** con la que medimos cualquier versión del sistema (el baseline del M1,
el RAG del M3, etc.). La función `harness(eval_set, sistema)` evalúa en **tres dimensiones** y
devuelve un *scorecard*. Está en `notebooks/M2/harness.ipynb`; el detalle está en
`notebooks/M2/README.md`.

| Dimensión | Qué mide |
|---|---|
| **D1 — métrica clásica** | Coincidencia literal (JSON válido, *exact-match*, F1 por campo) |
| **D2 — LLM como juez** | Calidad semántica con rúbrica 1–5 anclada (juez Qwen local) |
| **D3 — criterio de dominio** | Fracción de respuestas que cumplen el *criterion* del caso |

- **Eval set:** 12 casos de oro (8 estándar, 2 ambiguos, 2 trampa *out-of-domain*), curados por el
  equipo, con `input` / `esperado` / `criterio`.
- **Sesgo del juez mitigado:** la rúbrica está anclada por nivel y versionada; D3 añade una regla
  dura anti-alucinación (respuesta vacía obligatoria en las trampas), que corrige la lenidad del juez.

**Scorecard del baseline (Qwen zero-shot, sin LoRA):** D1 exact-match 8.3%, D2 juez 4.5/5, D3
cumplimiento 83.3%. El modelo produce JSON válido el 100% de las veces pero solo 1 de 12 coincide
literalmente: el formato nunca es el problema, lo es la *fidelidad del contenido* (relleno de campos
no señalados y abstención en casos fuera de dominio). Resultados en `data/scorecard_baseline.csv`.

> El harness se diseñó para re-ejecutarse sobre otro sistema. En el M3 reusamos la **misma vara**
> (3 dimensiones, misma firma, mismo eval set), re-instanciando cada dimensión para la tarea de QA
> del RAG, tal como el enunciado permite cuando la tarea del M1 era de extracción/clasificación.

---

## M3 — Sistema RAG agéntico: recomienda outfits reales del catálogo

**Qué hace:** recibe una petición en lenguaje natural y **recomienda conjuntos de ropa reales de un
catálogo** (los cita por su identificador), o activa una **válvula de escape** cuando la pregunta
está fuera del catálogo (precios, envíos, tallas). Reporte completo en `notebooks/M3/README.md`.

**Las cuatro versiones (sobre el mismo eval set, misma vara):**

| Sistema | D1 citación | D2 juez | D3 dominio | Latencia |
|---|---|---|---|---|
| Línea base M2 (sin RAG) | 0.167 | 4.75 | 0.75 | 4.65 s |
| RAG ingenuo | **0.917** | 4.33 | 0.75 | 8.29 s |
| RAG avanzado (híbrida + HyDE) | 0.833 | **4.67** | **0.833** | 13.81 s |
| Sistema final (con herramienta) | 0.75 | 4.00 | 0.667 | 12.23 s |

- **Corpus:** 800 outfits + 3796 prendas de moda (mujer y hombre), traducidos al español.
- **Técnicas por diagnóstico:** búsqueda **híbrida** (BM25 + dense, RRF) + **HyDE**. El **reranking**
  cross-encoder se evaluó y se **descartó** porque la ablación mostró que bajaba el puntaje.
- **Capa agéntica:** herramienta `filtrar_catalogo` (género/estación/ocasión) con esquema, parseo
  tolerante y tope de pasos.
- **Evaluación:** harness del M2 (3 dimensiones) + **RAGAS** (4 métricas, juez Qwen local).

**Lectura honesta:** el RAG aporta (D1 de 0.17 a 0.92: ancla al catálogo). El mejor sistema es el
**RAG avanzado**. La herramienta funciona y cumple el requisito, pero **no mejora el agregado**
porque su filtro reduce el *recall* (lo reportamos: "no todo necesita un agente"). RAGAS separa la
falla: *context precision* alta (recuperación relevante), *context recall* baja en el final
(recuperación limitada por el filtro) y *faithfulness* moderada (el generador pequeño añade detalles
no sustentados). Resultados en `notebooks/M3/resultados/`.

---

## Estructura del repositorio

```
.
├── data/                       # datos del M1/M2 (train, val, eval_set, scorecard baseline)
├── data_generation/            # generación del dataset (M1) y traducción del corpus (M3)
├── notebooks/
│   ├── M1/                     # tokenización + fine-tuning LoRA
│   ├── M2/                     # harness de evaluación + README
│   └── M3/                     # sistema RAG agéntico + README (reporte) + resultados
└── README.md                   # este archivo
```

## Reproducibilidad y uso de IA

Todo corre en **Colab gratuito sin claves de pago obligatorias**; semillas y versiones de modelos
fijadas. Los modelos del sistema (generador, juez y embeddings) son locales. Cada módulo documenta
en su propio README el detalle y la sección "Uso de IA".
