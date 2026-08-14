# Fine-Tuning Qwen2.5-1.5B para Recomendación de Ropa

**¿De qué trata este proyecto?**
Este sistema funciona como el cerebro de un asistente de moda: lee lo que una persona pide con sus propias palabras (como "quiero algo bien ancho y relajado para el frío") y extrae automáticamente las características exactas de la ropa (clima, estilo, ajuste) para que una tienda o aplicación pueda recomendarle las mejores prendas sin que el usuario tenga que pelear con filtros manuales (Se integrará en el futuro con los siguientes módulos de nuestro proyecto, los cuales recibirán estas categorías y realizaran el proceso de selección en la base de datos de ropa que se este usando).

## Modelo Base y Familia
**Modelo elegido:** `Qwen/Qwen2.5-1.5B-Instruct` (Familia Qwen).  
**Justificación:** Se seleccionó esta familia por su excelente capacidad multilingüe (especialmente para procesar español natural con abreviaciones comunes) y su fuerte rendimiento en el seguimiento de instrucciones estructuradas, lo que permite extraer atributos de forma precisa manteniendo un costo computacional eficiente (Algo que se adapta a nuestro contexto de enfoque a la experiencia de usuario).

## Baseline
**Baseline evaluado:** `Qwen2.5-1.5B-Instruct` en modalidad *Zero-shot*.  
**Justificación:** Se compara contra el modelo sin afinar para aislar y medir el impacto real de LoRA, comprobando si el modelo base era capaz de respetar la ontología estricta (categorías y valores permitidos) y omitir atributos no mencionados basándose únicamente en el prompt inicial.

## Tabla de Resultados
Evaluación realizada sobre el mismo conjunto de validación:

| Modelo | Exact-match | F1 macro |
|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | 20.0% | 0.688 |
| Qwen2.5-1.5B-Instruct + LoRA (fine-tuned) | **66.2%** | **0.752** |

*Nota técnica: Entrenamiento LoRA configurado con `r=16`, `alpha=32`, `dropout=0.05` afectando las proyecciones de atención (`q_proj`, `k_proj`, `v_proj`, `o_proj`).*

## Lectura Honesta de Resultados
El afinamiento mejoró sustancialmente el desempeño, logrando triplicar la métrica más estricta (Exact-match saltó de 20.0% a 66.2%) y elevando el F1 macro a 0.752. El principal salto de calidad se debe a que el modelo base "alucinaba" respuestas: tendía a rellenar forzosamente categorías que el usuario no había pedido (como asumir `ocasion: viaje` o `paleta: monocromatico` por defecto). El fine-tuning le enseñó a ser preciso y extraer solo la información presente en el texto. Sin embargo, sigue fallando en casi un 34% de los casos exactos; esto ocurre cuando el modelo enfrenta descripciones muy ambiguas, expresiones con jerga muy marcada, o cuando el usuario pide características que se cruzan entre estilos (haciendo que el modelo asigne un estilo distinto al de la etiqueta original).
