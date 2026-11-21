---
title: Personal Shopper Multimodal IA
emoji: 🧥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# Personal Shopper Multimodal (Módulo 5 — Sistema en Producción)

Sistema integrado de recomendación de outfits que combina:
- **M1:** Extractor de perfil de estilo con LoRA (`Qwen/Qwen2.5-1.5B-Instruct`).
- **M3:** Recuperación híbrida (`BM25` + `intfloat/multilingual-e5-small` + `RRF`) con filtrado duro de metadatos y HyDE estructurado.
- **M4:** Percepción visual y guardrail fuera de dominio (`Qwen/Qwen2-VL-2B-Instruct`).
- **M5:** Técnica de eficiencia (cuantización 4-bit NF4 con `bitsandbytes`) **medida** sobre el generador (VRAM 12.45 → 3.22 GB en NVIDIA T4; ver `05_evaluacion_m5.ipynb` y el reporte). Este Space corre en **ZeroGPU**, que no admite `bitsandbytes`, así que el generador se sirve en **bfloat16**; complementan el despliegue el HyDE estructurado, el caché LRU, el health check y el logging estructurado JSON sin PII sincronizado con Weights & Biases (`wandb`).