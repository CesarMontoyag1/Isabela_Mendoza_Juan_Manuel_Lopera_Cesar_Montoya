---
title: Personal Shopper de Moda
emoji: 👗
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
pinned: false
license: cc-by-nc-4.0
---

<!-- Hardware: ZeroGPU (GPU gratis por petición). El generador corre en GPU (fp16) dentro de
     funciones @spaces.GPU. No se usa GGUF aquí (GGUF es para el experimento de eficiencia en CPU). -->


# Personal Shopper de Moda — Demo (SI4006)

Sistema RAG que recomienda outfits reales de un catálogo de moda a partir de una petición en
lenguaje natural o de una **foto de una prenda**. Si la pregunta no está en el catálogo (precios,
envíos, tallas), lo dice en vez de inventar (**válvula de escape**).

- **Generador:** Qwen2.5-1.5B-Instruct cuantizado a **GGUF int4** (corre en CPU).
- **Recuperación:** híbrida (BM25 + embeddings e5, fusión RRF). HyDE desactivado por defecto en el
  Space (latencia); se puede activar con `RAG_USAR_HYDE=1`.
- **Herramienta:** filtro de metadatos (género/estación/ocasión).
- **Visión:** BLIP (image captioning) para convertir una foto en una consulta.

## Health check

Botón "Comprobar estado" en la UI, o por código:

```python
from gradio_client import Client
print(Client("https://CesarMontoya00-personal-shopper-moda.hf.space").predict(api_name="/health"))
```

## Variables de entorno (configurar como SECRETOS del Space, nunca en el repo)

| Variable | Para qué | Por defecto |
|---|---|---|
| `RAG_LOAD_IN` | backend del generador (`gguf` / `fp16`) | `gguf` |
| `RAG_USAR_HYDE` | activar HyDE (`1`/`0`) | `0` en el Space |
| `WANDB_API_KEY` | observabilidad en W&B (opcional) | — |
| `WANDB_PROJECT` | nombre del proyecto W&B | `personal-shopper-moda` |

## Fuente del corpus y licencia

Corpus derivado de **FashionStylist** (subsets Female + Male), traducido al español.
Licencia: **CC BY-NC 4.0** — https://github.com/recsys-benchmark/FashionStylist

> El sistema **sugiere**; la decisión es de la persona. Es un proyecto académico (SI4006).
