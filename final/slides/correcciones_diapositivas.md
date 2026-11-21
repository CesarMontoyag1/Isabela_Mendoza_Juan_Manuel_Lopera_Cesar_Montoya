# Correcciones de las diapositivas — Personal Shopper de Moda (SI4006)

Todas las cifras están verificadas contra los datos reales del repositorio
(`comparacion_m4_vs_m5.csv`, `scorecard_m3.csv`, `scorecard_m4.csv`, `eval_set_m4.json`, `pipeline.py`).
Se indica qué **dejar igual**, qué **cambiar** y qué **añadir**.

---

## SLIDE 1 — Portada

✅ **Déjala igual.** Está perfecta y coherente.

*(Opcional)* Si quieres, añade bajo los nombres una línea pequeña con el enlace:

> `huggingface.co/spaces/CesarMontoya/personal-shopper-moda`

---

## SLIDE 2 — El problema y el usuario

✅ **Déjala igual.** Coherente con el reporte §1. Nada que cambiar.

---

## SLIDE 3 — El sistema en un dibujo

Un solo cambio menor de precisión, en la tarjeta **"3 · BUSCA"**:

- **Dice:** "Filtra y compara con los 800 conjuntos reales."
- **Cambiar por:** "Busca y compara entre los 800 conjuntos reales."

*(Motivo: el sistema recupera con búsqueda híbrida; "filtrar" es solo una parte. "Busca" es más fiel.)*

El resto de la slide: ✅ déjala igual.

---

## SLIDE 4 — Demo en vivo

Reemplaza los textos genéricos por los **ejemplos reales** que vas a usar en vivo.
Ejemplos que **funcionan** (verificados contra el eval set y el comportamiento del sistema):

**Tarjeta FÁCIL (Solo texto):**

> "algo elegante y minimalista para la oficina en invierno, en tonos oscuros"

**Tarjeta DIFÍCIL (Texto y foto):**

> "Quiero un outfit que combine con esta prenda para salir el fin de semana" + subir una de las
> fotos de `final/imagenes_test/` (p. ej. `ropa_01.png`)

**Tarjeta ADVERSARIAL (Pregunta sin respuesta):** — déjala como está, funciona seguro:

> "¿Cuánto cuesta el envío a Medellín?" → debe responder «No encuentro esa información en el catálogo»

⚠️ **Importante para la demo en vivo:** usa el adversarial **de texto** (precio/envío), NO el
adversarial de imagen. Según tu propio análisis de errores (§7 #1), la versión del sistema a veces
**no detecta** imágenes que no son ropa. Si muestras ese caso en vivo y falla, queda mal. El
adversarial de texto activa la válvula de forma fiable.

Puedes quitar la nota "Ejemplos sugeridos: reemplazar..." una vez pongas los reales.

---

## SLIDE 5 — Los números

✅ **La tabla está correcta** (todos los valores verificados). Solo cambia la **nota al pie** para
que sea más clara:

- **Dice:** "Pocos casos de prueba (12 a 18): una diferencia pequeña equivale a un solo caso. El
  punto de partida responde rápido, pero inventa."
- **Cambiar por:** "Pocos casos de prueba (12 de solo texto, 18 al sumar imágenes): una diferencia
  pequeña equivale a un solo caso. El punto de partida responde rápido, pero inventa."

---

## SLIDE 6 — Hacerlo más liviano

✅ Las 3 tarjetas (−74% / −0,28 / =) están correctas. **El único cambio obligatorio** es la nota al
pie, porque la actual da a entender que el Space corre en la versión cuantizada, y no es así (corre
en bfloat16 por ZeroGPU). Esto debe ser coherente con lo que ya dejamos escrito en el reporte §5.

- **Dice:** "Costo escondido: la versión liviana dejó pasar imágenes que no eran ropa, y eso el
  promedio no lo mostraba."
- **Cambiar por (texto completo nuevo del pie):**

> "La reducción de memoria se midió en una GPU T4. El Space público corre en ZeroGPU, que no admite
> esta técnica, así que allí el modelo va en su versión normal; la medición demuestra que la técnica
> permite reducir el modelo a un tercio. Costo escondido: la versión liviana dejó pasar imágenes que
> no eran ropa, y eso el promedio no lo mostraba."

*(Si no cabe todo, parte en dos líneas o reduce un poco la fuente del pie. Lo esencial es la primera
frase: aclarar T4 vs ZeroGPU.)*

---

## SLIDE 7 — Lo que falla

✅ **Déjala igual.** Las 3 tarjetas y el pie ético/privacidad son coherentes con reporte §7, §9 y
§10. Nada que cambiar.

---

## SLIDE 8 — NUEVA (añádela al final): Enlaces y cierre

Crea una slide de cierre (mismo estilo que las demás) con este contenido:

- **Título pequeño (eyebrow):** `GRACIAS`
- **Título grande:** `Pruébalo y revisa los números`

**Tres bloques o una lista:**

- **Demo en vivo (Hugging Face Spaces):**
  `huggingface.co/spaces/CesarMontoya/personal-shopper-moda`
- **Observabilidad (Weights & Biases):**
  Proyecto `personal-shopper-moda` (privado). Evidencia del dashboard en el reporte.
- **Repositorio (GitHub):**
  `github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya`

**Pie:** `Equipo: Isabela Mendoza · Juan Manuel Lopera · César Montoya`

---

## Resumen de acciones

| Slide | Acción | Prioridad |
|---|---|---|
| 1 | (Opcional) añadir enlace del Space | 🟢 |
| 2 | Nada | — |
| 3 | "Busca y compara" en vez de "Filtra y compara" | 🟢 |
| 4 | Poner los 3 ejemplos reales (usar adversarial de **texto**) | 🔴 |
| 5 | Nota al pie más clara ("12 solo texto, 18 con imágenes") | 🟡 |
| 6 | **Nota al pie nueva** (T4 vs ZeroGPU) | 🔴 obligatorio |
| 7 | Nada | — |
| 8 | **Crear slide de cierre** con enlaces | 🔴 |

---

## Antes de la sustentación — 2 recordatorios del enunciado

1. **Tengan a mano un video corto de la demo funcionando** (el enunciado lo sugiere por si el Space
   o la conexión fallan en vivo). Graba un screencast de 1-2 min con los 3 casos.
2. **Máximo 10 minutos.** Con 8 slides, ~1 min por slide + la demo en vivo, vas bien. No te pases del
   máximo (eso cuenta en contra).

Las dos correcciones críticas (🔴) son la **slide 6** (coherencia) y la **slide 4 + slide 8**
(contenido). El resto son mejoras menores.
