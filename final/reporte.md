# Reporte Técnico Final — Personal Shopper Multimodal de Moda (M1–M5)

* **Equipo:** Isabela Mendoza, Juan Manuel Lopera Soto, César Montoya
* **Curso:** SI4006 · Proyecto Integrador (Módulos 1 a 5)
* **Demo en línea (Hugging Face Spaces):** `<PEGAR_URL_DEL_SPACE_AQUI>`
* **Observabilidad (Weights & Biases):** `<PEGAR_URL_DEL_PROYECTO_WANDB_AQUI>`
* **Diapositivas de sustentación:** `final/presentacion_sustentacion.pdf` *(o `<PEGAR_ENLACE_DIAPOSITIVAS>`)*
* **Repositorio:** [GitHub — Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya](https://github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya)

---

## 1. El Problema y el Usuario

* **Usuario objetivo:** Compradores digitales de tiendas de moda online (*e-commerce*) que buscan armar conjuntos completos (*outfits*) a partir de una necesidad situacional descrita en lenguaje natural (ocasión, clima, estilo, corte) o de una fotografía de referencia de una prenda que ya poseen o desean combinar.
* **Decisión que toma con el sistema:** Seleccionar y combinar prendas reales disponibles en el catálogo (`corpus_look_es.csv` y `corpus_label_es.csv`, compuestos por **800 outfits** etiquetados en español) sin recorrer manualmente decenas de filtros ni recibir recomendaciones inventadas fuera de inventario.
* **Métrica de éxito conectada a la decisión:**
  1. **Fidelidad al inventario (`D1` — Citación de IDs reales):** Proporción de outfits recomendados que pertenecen realmente a los candidatos recuperados del catálogo (`[BC]_b\d+`), evitando que el usuario haga clic en prendas inexistentes.
  2. **Cumplimiento operativo y contención (`D3` — Cumplimiento / Válvula de escape):** Tasa de recomendaciones útiles (`puntaje juez >= 4/5`) en consultas de moda y activación oportuna de la válvula (`"No encuentro esa información en el catálogo"`) ante preguntas fuera de alcance (precios, envíos, tallas numéricas o imágenes ajenas a moda).

---

## 2. Arquitectura del Sistema Integrado (Diagrama)

Los cinco módulos operan como un único pipeline secuencial (`final/pipeline.py`) sin dependencias entre notebooks sueltos:

```text
[Entrada Usuario: Texto + Imagen opcional]
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ M4 · PERCEPCIÓN VISUAL (Qwen/Qwen2-VL-2B-Instruct, 4-bit NF4)     │
│  • Si adjunta imagen:                                             │
│    1. Guardrail visual: verifica si es prenda/moda (`es_moda`)    │
│       └─ NO (gráfico, ruido, fuera de dominio) ──► Válvula escape │
│    2. Extrae género aparente, clima sugerido y descripción breve  │
└───────────────────┬───────────────────────────────────────────────┘
                    │ Consulta enriquecida (Texto + Atributos visuales)
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ M1 · EXTRACTOR DE PERFIL (Qwen2.5-1.5B + LoRA r=16 activo)        │
│  • Extrae JSON estricto bajo ontología de 25 valores:             │
│    {estilo, ocasion, clima, paleta, fit}                          │
└───────────────────┬───────────────────────────────────────────────┘
                    │ Perfil JSON estructurado
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ M3 · AGENTE RAG HÍBRIDO (Qwen2.5-1.5B con LoRA desactivado)       │
│  • Herramienta `filtrar_catalogo`: mapea perfil M1 + visión M4 a  │
│    filtros duros de metadatos (`gender`, `season`, `occasion`)    │
│  • Recuperación Triple + RRF (k=60):                              │
│    1. Búsqueda densa (`intfloat/multilingual-e5-small` + Chroma)  │
│    2. Búsqueda léxica (`BM25Okapi`)                              │
│    3. HyDE estructurado M5 (desde perfil M1 + consulta)           │
│  • Generación fundamentada con citación obligatoria de IDs        │
└───────────────────┬───────────────────────────────────────────────┘
                    │ Respuesta + Traza JSON + Métricas
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│ M2 & M5 · EVALUACIÓN, EFICIENCIA Y OBSERVABILIDAD                 │
│  • Harness M2: D1 (citación) + D2 (juez independiente) + D3       │
│  • M5 Producción: Caché LRU, `/health`, JSONL sin PII + W&B       │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Decisiones Técnicas Justificadas

| Componente | Decisión Elegida | Alternativa Descartada | Justificación Técnica |
| :--- | :--- | :--- | :--- |
| **Modelo Base y Ajuste (M1/M3)** | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (`r=16`, `alpha=32`, `dropout=0.05`, proyecciones `q/k/v/o_proj`, 4.35M parámetros entrenables = 0.2815 %) | *Full fine-tuning* o modelos > 7B | Sobre 1,040 ejemplos de entrenamiento (y 260 de validación), LoRA elevó el *Exact-Match* de `20.0 %` a `66.5 %` (y `F1` promedio de `0.688` a `0.752`), permitiendo alternar dinámicamente entre modo extractor (LoRA activo) y generador RAG (`disable_adapter()`) sin duplicar pesos en VRAM. |
| **Recuperación y Reranking (M3)** | Recuperación Híbrida (`e5-small` + `BM25`) fusionada con RRF (`k=60`) + filtrado por metadatos + HyDE | Cross-encoder reranker adicional o búsqueda puramente densa | En la ablación de M3, añadir un cross-encoder degradó el cumplimiento (`D3` bajó de `0.833` a `0.667`) y sumó latencia; la fusión híbrida con HyDE equilibró coincidencias exactas de prendas (`BM25`) con intención semántica (`e5-small`). |
| **Patrón Visual (M4)** | VLM generativo (`Qwen/Qwen2-VL-2B-Instruct`) como extractor semántico y guardrail de dominio hacia el puente textual | Embeddings imagen-texto directos (`CLIP ViT-B/32`) | El catálogo (`800` outfits) contiene descripciones detalladas en español pero no fotografías propias; CLIP tiene alineación débil en español y carece de razonamiento para rechazar imágenes fuera de dominio con la válvula de escape. |
| **Técnica de Eficiencia (M5)** | Cuantización `int4` NF4 (`bitsandbytes`) + atención `SDPA` + HyDE estructurado determinista + caché LRU | Destilación a `Qwen2.5-0.5B` o mantener FP16 sin optimizar | `int4` redujo la VRAM máxima en **74.14 %** (`12.45 GB` $\rightarrow$ `3.22 GB`), permitiendo desplegar en hardware restringido. Se descartó bajar a `0.5B` porque degrada el seguimiento estricto de citación de IDs y reglas negativas. |

---

## 4. Evolución del Sistema (M1 $\rightarrow$ M5)

Todas las etapas integradas (`M2` a `M5`) se evaluaron bajo la misma vara del harness (`D1` citación, `D2` juez independiente en escala 1–5 y `D3` cumplimiento de reglas/válvula):

| Etapa del Sistema | Dataset / Eval Set | Calidad (`D1` / `D2` / `D3`) | Casos con Imagen | Latencia Mediana (`p50`) |
| :--- | :--- | :--- | :---: | :---: |
| **M1 · Extractor LoRA (solo JSON)** | 260 val (`1,300` sintéticos) | `F1 = 0.752` (`EM = 66.5 %` vs `20.0 %` zero-shot) | No aplica | — |
| **M2 · Línea Base sin RAG** | 12 casos texto (`eval_set_rag`) | `D1 = 0.167` · `D2 = 4.750` · `D3 = 0.750` | No aplica | `4.712 s` |
| **M3 · RAG Híbrido + HyDE + Tool** | 12 casos texto (`eval_set_rag`) | `D1 = 0.833` · `D2 = 4.667` · `D3 = 0.833` *(sin tool)* / `0.750 · 4.000 · 0.667` *(con tool)* | No aplica | `13.539 s` / `12.134 s` |
| **M4 · Multimodal Integrado (FP16)** | 18 casos (`12` texto + `6` imagen) | `D1 = 0.833` · `D2 = 3.889` · `D3 = 0.722` *(juez externo)* | Sí (`D1=0.833`, `D2=3.833`, `D3=0.667`) | `10.476 s` (`9.875 s` texto / `15.932 s` img) |
| **M5 · Final Optimizado (`int4` NF4)** | 18 casos (`12` texto + `6` imagen) | `D1 = 0.750` · `D2 = 3.611` · `D3 = 0.611` *(juez externo)* | Sí (incluidos en los 18 casos) | `11.375 s` (`< 0.01 s` en *cache hit*) |

*(Nota metodológica: A partir de M4 se reemplazó el juez interno Qwen por un juez externo de familia distinta para eliminar el sesgo de auto-preferencia).*

---

## 5. Evaluación de la Técnica de Eficiencia (Antes vs. Después)

### Criterio previo a la medición
Antes de ejecutar la optimización definimos el umbral de aceptación: **"aceptamos perder hasta $X = 0.30$ puntos en el juez `D2` (y hasta $0.15$ en `D1`/`D3`) si la memoria VRAM baja por debajo de $4.0\text{ GB}$ y la latencia `p95` baja de $16.5\text{ s}$ manteniendo `p50` cerca de $Y = 11.5\text{ s}$"**.

### Tabla comparativa en la misma máquina (NVIDIA T4, 18 casos)

| Versión | Calidad (`D1` / `D2` / `D3`) | Latencia `p50` | Latencia `p95` | Memoria VRAM Máx. |
| :--- | :---: | :---: | :---: | :---: |
| **Antes — M4 Completo (FP16 sin optimizar)** | `0.833` / `3.889` / `0.722` | `10.476 s` | `16.837 s` | `12.45 GB` |
| **Después — M5 Optimizado (`int4` NF4 + SDPA + HyDE estructurado)** | `0.750` / `3.611` / `0.611` | `11.375 s` | `16.319 s` | `3.22 GB` |
| **Variación relativa (`M4` $\rightarrow$ `M5`)** | `-9.96 %` / `-7.15 %` / `-15.37 %` | `+8.58 %` | `-3.08 %` | **`-74.14 %` (`-9.23 GB`)** |

**Lectura honesta del resultado:**
* **Se cumplió** el objetivo de memoria (`3.22 GB < 4.0 GB`, ahorro de `9.23 GB`), la tolerancia en calidad (`-0.278` en `D2` $\le 0.30$) y la reducción de cola de latencia (`p95 = 16.319 s < 16.837 s`).
* **Trade-off de hardware en `p50`:** En una GPU NVIDIA T4 (arquitectura Turing sin Tensor Cores nativos para `NF4`), descuantizar pesos de 4 bits a FP16 en cada capa lineal añade sobrecarga de cómputo en modelos compactos (1.5B–2B). En nuestra primera prueba 4-bit (manteniendo HyDE generativo), `p50` subió a `14.986 s`; al sustituir HyDE generativo por HyDE estructurado y acotar tokens de salida, logramos recuperar la latencia hasta `11.375 s` manteniendo el ahorro de `74.14 %` en VRAM.

### Pruebas de No-Exactitud sobre la Versión Eficiente (Válvula, Adversariales e Invarianza)

| Variante Visual (`ropa_01.png`) | ¿Moda Detectada? | ¿Válvula Activada? | Solapamiento IDs vs. Original | Top-3 Outfits Recuperados |
| :--- | :---: | :---: | :---: | :--- |
| `original` | `True` | `False` | `1.00` | `B_b32, B_b425, B_b118` |
| `flip_horizontal` | `True` | `False` | `0.00` | `C_b122, C_b195, C_b185` |
| `brillo_mas_20` | `True` | `False` | `0.00` | `C_b122, C_b157, C_b155` |
| `recorte_85pct` | `True` | `False` | `0.00` | `C_b195, C_b266, C_b196` |

* **Invarianza:** La clasificación de dominio (`es_moda = True`) y no-activación falsa de la válvula es **100 % invariante** ante espejado, brillo (+20 %) y recorte central (15 %). Sin embargo, la cuantización `int4` altera ligeramente la redacción descriptiva del VLM, desplazando el ranking exacto de IDs hacia otros outfits del mismo estilo (`solapamiento = 0.0` frente al `0.6–0.8` de FP16).
* **Válvula y casos adversariales:** En texto fuera de catálogo (precios/envíos), la válvula se activa correctamente (`valvula_activada = True` en el registro `abe15eb6` de `logs_observabilidad.jsonl`). No obstante, en los **2 casos adversariales con imagen** (`adv_grafico.png` y `adv_ruido.png`), la cuantización `int4` + prompt visual compacto redujo la capacidad de rechazo del VLM (`es_moda = True`), explicando la caída de `D3` de `0.722` a `0.611`.

---

## 6. Estudios de Ablación

Realizamos ablaciones sistemáticas en M1 y M3 para validar cada componente antes de integrarlo:

| Configuración Evaluada | Métrica Principal | Latencia Mediana | Conclusión |
| :--- | :---: | :---: | :--- |
| **M3 · Completo (Híbrida BM25+e5 + HyDE)** | `D1 = 0.833` · `D2 = 4.667` · `D3 = 0.833` | `13.591 s` | Mejor equilibrio entre citación, calidad y cumplimiento. |
| **M3 · Ablación sin BM25 (`solo dense`)** | `D1 = 0.917` · `D2 = 4.333` · `D3 = 0.750` | `8.183 s` | Pierde coincidencias léxicas de prendas específicas (`-0.083` en `D3`). |
| **M3 · Ablación sin HyDE (`hibrida sin HyDE`)** | `D1 = 0.833` · `D2 = 4.667` · `D3 = 0.750` | `8.111 s` | Consultas coloquiales pierden expansión semántica (`-0.083` en `D3`). |
| **M3 · Ablación con Reranker (descartado)** | `D1 = 0.833` · `D2 = 4.583` · `D3 = 0.667` | `12.968 s` | Penaliza respuestas válidas de válvula de escape (`-0.166` en `D3`). |
| **M1 · Ablación sin LoRA (`zero-shot`)** | `EM = 20.0 %` · `F1 = 0.688` (vs `66.5 %` / `0.752` LoRA) | — | Sin LoRA el extractor confunde valores de `estilo` y `ocasion`. |

---

## 7. Análisis de Errores (5 Casos Reales)

| # | Tipo de Fallo | Caso Concreto | Pieza que Falló | Causa Raíz y Propuesta de Corrección |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Falso negativo en guardrail visual (`int4`)** | `m4_adv_01` y `m4_adv_02` (gráfico de barras y ruido aleatorio clasificados como `es_moda: true`) | **M4 + M5** (`Qwen2-VL-2B` cuantizado a 4-bit) | La cuantización NF4 junto con un prompt visual acortado redujo la estrictez del filtro OOD. **Corrección:** Añadir un clasificador binario ligero (CLIP zero-shot o MobileNet) antes del VLM como primera barrera determinista de dominio. |
| **2** | **Alucinación de atributo en consulta OOD de texto** | `eval_07` (M2): *"¿Cuánto vale el envío a Medellín y me sirve mi talla 32?"* $\rightarrow$ `{'fit': 'regular'}` | **M1** (Extractor LoRA) | El extractor asocia la palabra `"talla"` con el campo `fit`. **Corrección:** Incluir 10 % de ejemplos negativos (`output: {}`) sobre envíos, precios y tallas numéricas en `train.jsonl`. |
| **3** | **Sobre-filtrado de metadatos** | `eval_06` (M3 con tool): consulta ambigua sin género explícito pierde candidatos relevantes | **M3** (`filtrar_catalogo`) | Aplicar filtros duros (`AND`) demasiado restrictivos reduce el *Context Recall* (`0.75` $\rightarrow$ `0.50` en RAGAS). **Corrección:** Convertir el filtro de metadatos en *soft-boost* dentro de RRF en lugar de exclusión estricta. |
| **4** | **Desplazamiento de ranking bajo perturbación visual** | `invarianza_m5.csv` (`flip_horizontal` / `brillo_mas_20` cambian el top-3 de `B_b32` a `C_b122`) | **M4 + M5** (VLM `int4` $\rightarrow$ puente textual) | Pequeñas variaciones en los adjetivos generados por el VLM cuantizado alteran el orden en RRF, aunque mantienen el estilo. **Corrección:** Normalizar los atributos visuales al vocabulario cerrado de M1 antes de consultar Chroma. |
| **5** | **Citación incompleta cuando hay 1 solo outfit claro** | Casos en M5 con presupuesto estricto de `115 tokens` que omiten el segundo ID recuperado | **M3 + M5** (Generador) | El límite corto de tokens prioriza la explicación en prosa sobre la enumeración de IDs. **Corrección:** Formatear la salida con plantilla fija (`IDs recomendados: ... | Justificación: ...`). |

---

## 8. Despliegue, Health Check y Observabilidad

* **Interfaz y URL pública (`final/app.py`):** Desplegada en Hugging Face Spaces con Gradio en dos pestañas: *Asistente Personal Shopper* (texto + imagen opcional + traza JSON de M1/M4/M3) y *Health Check & Observabilidad*.
* **Health Check (`PersonalShopperOptimizado.health_check()`):** Verifica sin ejecutar inferencia que el índice vectorial Chroma tenga los 800 outfits cargados, el dispositivo activo (`cuda`/`cpu`), el estado de cuantización (`quantization_4bit`), la carga del adaptador LoRA (`lora_loaded`) y la memoria VRAM pico.
* **Logging estructurado sin PII (`final/observability.py`):** Cada petición escribe una línea JSON en `final/resultados/logs_observabilidad.jsonl` y sincroniza métricas en tiempo real con **Weights & Biases (`wandb`)** (`inferencia/latencia_s`, `inferencia/uso_herramienta`, `inferencia/valvula_activada`, `inferencia/cache_hit`, `inferencia/error`). Ningún texto crudo ni imagen se persiste: las consultas se anonimizan mediante `SHA-256` truncado a 12 caracteres (`query_hash`).

---

## 9. Limitaciones del Sistema

* **Dónde NO creerle al sistema:**
  * **Disponibilidad de stock en tiempo real, precios, tallas numéricas (36, 38, M, L) y envíos:** El catálogo indexado contiene únicamente descripciones estilísticas de outfits (`corpus_look_es.csv`); cualquier consulta comercial debe ser rechazada por la válvula de escape.
  * **Imágenes ambiguas o no relacionadas con ropa en modo 4-bit:** Como evidenció la evaluación de M5, el VLM cuantizado puede intentar interpretar gráficos abstractos como estampados textiles si no se acompaña de un filtro previo de categoría.
  * **Hardware CPU gratuito en Hugging Face Spaces:** En tier gratuito sin GPU (`CPU Basic`), la inferencia de modelos de 1.5B–2B tarda decenas de segundos por petición fría (mitigado parcialmente mediante el caché LRU para consultas repetidas).
* **Qué hace cuando no sabe:** Ante consultas administrativas/comerciales o imágenes detectadas como `es_moda: false`, el sistema corta la ejecución antes del RAG y responde explícitamente con la válvula de escape: *"No encuentro esa información en el catálogo"*.

---

## 10. Consideraciones Éticas

1. **¿A quién podría fallarle más?**
   El catálogo (`corpus_look_es.csv`) está estructurado en categorías binarias tradicionales (`De mujer`, `De hombre`, `Unisex`) y en español estándar, por lo que el sistema funciona peor para personas que buscan moda no binaria/fluida, tallas especiales (*plus size*, *petite*) o que usan regionalismos muy locales de Latinoamérica no presentes en el vocabulario del catálogo. Esto se observó cualitativamente en consultas ambiguas sin género explícito donde el filtro tiende a sesgarse hacia una de las dos categorías mayoritarias.
2. **¿Qué pasa con los datos?**
   El sistema recibe texto libre y fotografías subidas por el usuario durante la sesión. Ninguna imagen ni texto crudo se almacena en disco ni se envía a terceros: el módulo `final/observability.py` convierte cada consulta en un hash criptográfico irreversible (`SHA-256` truncado a 12 caracteres) y registra únicamente metadatos técnicos (latencia, uso de herramienta, activación de válvula) en `logs_observabilidad.jsonl` y en Weights & Biases, informándolo en la pestaña *Health Check & Observabilidad*.
3. **¿Qué daño haría un error?**
   El peor error operativo sería recomendar prendas inexistentes (alucinación de inventario) o afirmar falsamente compatibilidad de tallas/políticas de devolución, causando frustración al comprador, compras erróneas y costos logísticos de devolución para la tienda. Para mitigarlo, el sistema obliga a citar IDs verificados del catálogo (`D1`), aplica la válvula de escape ante consultas fuera de alcance y mantiene la decisión final de compra siempre en manos del usuario.

---

## 11. Quién Hizo Qué

* **Isabela Mendoza:** Diseño y generación del dataset sintético de dominio, entrenamiento y evaluación del extractor LoRA (M1) y redacción de casos de prueba.
* **Juan Manuel Lopera Soto:** Implementación del componente visual multimodal (M4), integración del pipeline unificado, cuantización `int4` y observabilidad en Weights & Biases (M5).
* **César Montoya:** Construcción del harness de evaluación multi-dimensión (M2), índice híbrido RAG con filtrado por metadatos y estudios de ablación (M3).

---

## 12. Declaración de Uso de IA

Para la construcción de esta entrega se utilizó **Gemini** como asistente de programación y depuración técnica para: (1) estructurar el pipeline modular en `final/` (`pipeline.py`, `observability.py`, `app.py`), (2) depurar conflictos de dependencias en Google Colab (`Pillow`, `bitsandbytes`, `HF_TOKEN`) y (3) organizar la síntesis de tablas de este reporte. Todo el código fue ejecutado, verificado y ajustado por el equipo sobre nuestro catálogo y conjunto de evaluación propio, detectando y corrigiendo problemas reales como el sesgo del juez propio en M4 y la sensibilidad de resolución en las pruebas de invarianza de M5.