---
title: Personal Shopper Multimodal IA
emoji: 🧥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Personal Shopper Multimodal (Módulo 5 — Sistema en Producción)

Sistema integrado de recomendación de outfits que combina:
- **M1:** Extractor de perfil de estilo con LoRA (`Qwen/Qwen2.5-1.5B-Instruct`).
- **M3:** Recuperación híbrida (`BM25` + `intfloat/multilingual-e5-small` + `RRF`) con filtrado duro de metadatos y HyDE estructurado.
- **M4:** Percepción visual y guardrail fuera de dominio (`Qwen/Qwen2-VL-2B-Instruct`).
- **M5:** Cuantización 4-bit NF4 (en GPU) / bfloat16 (en CPU), caché LRU, health check y logging estructurado JSON sin PII sincronizado con Weights & Biases (`wandb`).