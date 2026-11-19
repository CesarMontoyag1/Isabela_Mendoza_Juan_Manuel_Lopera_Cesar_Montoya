# -*- coding: utf-8 -*-
"""Genera final/eficiencia.ipynb: mide la técnica de eficiencia (GGUF int4) antes/después."""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): CELLS.append({"cell_type": "code", "metadata": {}, "source": t,
                           "execution_count": None, "outputs": []})

def _split(t):
    t = t.rstrip("\n")
    if t == "": return [""]
    ls = t.split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]

def build():
    for c in CELLS: c["source"] = _split(c["source"])
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eficiencia.ipynb")
    with open(out, "w", encoding="utf-8") as f: json.dump(nb, f, ensure_ascii=False, indent=1)
    with open(out, encoding="utf-8") as f: json.load(f)
    print("OK:", out, len(CELLS), "celdas")


md(r'''# Módulo 5 — Técnica de eficiencia: cuantización GGUF int4 del generador

**Componente elegido:** el **generador Qwen2.5-1.5B**, por ser el que más tarda y más memoria ocupa
en el pipeline (HyDE + respuesta son dos generaciones por consulta). Los embeddings y el reranking
ya son ligeros; el generador es el cuello de botella.

**Técnica:** cuantización a **GGUF int4 (Q4_K_M)** con `llama.cpp`, que corre **rápido en CPU** —
necesario para el despliegue en Hugging Face Spaces gratuito (sin GPU). Alternativa descartada:
bitsandbytes int4/int8, que **requiere GPU** y no sirve en el Space gratuito.

**Presupuesto de calidad (decidido ANTES de medir):**
> *Aceptamos perder hasta 1 punto del juez (D2) y hasta 0.10 en D1/D3 del harness si la latencia
> mediana en CPU baja de ~30 s a menos de ~8 s por consulta y el tamaño del modelo baja a < 1.5 GB.*

Al final de este notebook decimos si se cumplió.''')

code(r'''# Dependencias (Colab). Descomenta la primera vez.
# !pip -q install llama-cpp-python huggingface_hub chromadb sentence-transformers rank-bm25 \
#     transformers accelerate pandas numpy
# !git clone -b cesar https://github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya.git
# %cd Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, 'final')
CORPUS_DIR = 'notebooks/M3'
EVAL = 'notebooks/M3/eval/eval_set_rag.json'
print('listo')''')

md(r'''## 1. Convertir Qwen2.5-1.5B a GGUF int4

Descargamos un GGUF ya cuantizado del modelo (Q4_K_M) desde Hugging Face. Si prefieres convertirlo
tú, se hace con `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`); aquí usamos el publicado
para reproducibilidad. Medimos su tamaño en disco.''')

code(r'''from huggingface_hub import hf_hub_download
# GGUF oficial de Qwen2.5-1.5B-Instruct (cuantización Q4_K_M)
GGUF_REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
GGUF_FILE = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
ruta_gguf = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
tam_gguf_mb = os.path.getsize(ruta_gguf) / 1e6
print('GGUF descargado:', ruta_gguf)
print('Tamaño GGUF int4: %.0f MB' % tam_gguf_mb)''')

md(r'''## 2. Cargar las dos versiones del sistema (misma vara, mismo eval set)

- **Antes:** generador fp16 (Transformers). En Colab GPU.
- **Después:** generador GGUF int4 (llama.cpp). En CPU.

Ambas comparten corpus, índice, recuperación (híbrida+HyDE) y herramienta: lo único que cambia es
el backend del generador, así el delta es atribuible a la cuantización.''')

code(r'''from rag_core import PersonalShopper
from harness import harness, cargar_eval_set, hacer_juez_gemini, hacer_juez_mismo_generador

EVAL_SET = cargar_eval_set(EVAL)
print('Casos:', len(EVAL_SET))

# ANTES: fp16 (usa GPU si hay; si no, fp32 en CPU)
shopper_antes = PersonalShopper(corpus_dir=CORPUS_DIR, load_in='fp16')
ids_validos = set(shopper_antes.doc_ids)

# Juez: preferimos Gemini (distinto del generador) para que D2 discrimine; si no hay clave,
# caemos al juez local (mismo generador) y lo reportamos como limitación.
juez = hacer_juez_gemini() or hacer_juez_mismo_generador(shopper_antes)
print('Juez:', 'Gemini' if hacer_juez_gemini() else 'local (mismo generador)')''')

code(r'''# Medir ANTES (fp16) con el sistema avanzado
import psutil, os as _os
def mem_mb():
    return psutil.Process(_os.getpid()).memory_info().rss / 1e6

m0 = mem_mb()
sc_antes, det_antes = harness(EVAL_SET, lambda q: shopper_antes.sistema(q, modo='avanzado'),
                              juez, ids_validos, nombre='Antes (fp16)')
sc_antes['memoria_proceso_mb'] = round(mem_mb() - m0 + m0, 0)  # RSS actual
print(sc_antes)''')

code(r'''# Liberar memoria del modelo fp16 antes de cargar el GGUF
import gc, torch
del shopper_antes
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# DESPUÉS: GGUF int4 en CPU
shopper_despues = PersonalShopper(corpus_dir=CORPUS_DIR, load_in='gguf',
                                  gguf_repo=GGUF_REPO, gguf_file=GGUF_FILE)
juez2 = hacer_juez_gemini() or hacer_juez_mismo_generador(shopper_despues)
sc_despues, det_despues = harness(EVAL_SET, lambda q: shopper_despues.sistema(q, modo='avanzado'),
                                  juez2, ids_validos, nombre='Después (GGUF int4)')
sc_despues['memoria_proceso_mb'] = round(tam_gguf_mb, 0)
print(sc_despues)''')

md(r'''## 3. Tabla antes / después''')

code(r'''tabla = pd.DataFrame([sc_antes, sc_despues])
os.makedirs('final/resultados', exist_ok=True)
tabla.to_csv('final/resultados/eficiencia_antes_despues.csv', index=False, encoding='utf-8')
print('Guardado: final/resultados/eficiencia_antes_despues.csv')
tabla''')

md(r'''## 4. Pruebas que NO miden exactitud, repetidas sobre la versión eficiente

El enunciado: "un modelo más chico puede acertar lo mismo y perder la capacidad de decir *no sé*".
Repetimos sobre la versión GGUF int4: (a) la **válvula** en los 2 casos fuera de corpus, y
(b) una **prueba de invarianza** (parafraseos de la misma consulta deben dar outfits similares).''')

code(r'''# (a) Válvula sobre la versión eficiente
fuera = [e for e in EVAL_SET if e['tipo'] == 'fuera_de_corpus']
print('=== Válvula (GGUF int4) ===')
for e in fuera:
    s = shopper_despues.sistema(e['input'], modo='avanzado')
    print('-', e['input'][:50], '->', 'VÁLVULA OK' if shopper_despues.activo_valvula(s['respuesta'])
          else 'NO activó (inventó)')''')

code(r'''# (b) Invarianza: parafraseos de la misma intención deben recuperar outfits parecidos
paras = ['abrigo largo de lana para un evento formal en invierno',
         'un abrigo de lana largo para una ocasión formal de invierno',
         'quiero un abrigo de lana largo, elegante, para invierno formal']
sets = []
for p in paras:
    s = shopper_despues.sistema(p, modo='avanzado')
    sets.append(set(s['outfits']))
    print('-', p[:50], '->', s['outfits'][:5])
inter = set.intersection(*sets) if sets else set()
union = set.union(*sets) if sets else set()
print('Jaccard de solapamiento entre parafraseos:', round(len(inter)/max(1,len(union)), 2))''')

md(r'''## 5. ¿Se cumplió el presupuesto?

Comparar `sc_antes` vs `sc_despues`: ¿la pérdida de calidad (D1/D2/D3) quedó dentro del presupuesto
(≤1 punto en D2, ≤0.10 en D1/D3) y la latencia p50 bajó del objetivo? Escribir el veredicto en el
reporte final (`final/reporte.md`). Si NO se cumplió, también es un resultado válido: se documenta y
se decide qué versión se despliega.''')

code(r'''d_d2 = sc_antes['D2_juez_medio'] - sc_despues['D2_juez_medio']
d_d1 = sc_antes['D1_citacion_outfits'] - sc_despues['D1_citacion_outfits']
d_d3 = sc_antes['D3_cumplimiento'] - sc_despues['D3_cumplimiento']
print('Pérdida D2: %.2f (presupuesto <= 1.0)' % d_d2)
print('Pérdida D1: %.3f (presupuesto <= 0.10)' % d_d1)
print('Pérdida D3: %.3f (presupuesto <= 0.10)' % d_d3)
print('Latencia p50: antes %.1fs -> después %.1fs' % (sc_antes['latencia_p50_s'], sc_despues['latencia_p50_s']))
cumple = (d_d2 <= 1.0) and (d_d1 <= 0.10) and (d_d3 <= 0.10)
print('\\nVEREDICTO:', 'SE CUMPLIÓ el presupuesto' if cumple else 'NO se cumplió (documentar decisión)')''')


if __name__ == "__main__":
    build()
