# -*- coding: utf-8 -*-
"""Genera final/probar_demo.ipynb: prueba la app Gradio de punta a punta en Colab."""
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probar_demo.ipynb")
    with open(out, "w", encoding="utf-8") as f: json.dump(nb, f, ensure_ascii=False, indent=1)
    with open(out, encoding="utf-8") as f: json.load(f)
    print("OK:", out, len(CELLS), "celdas")


md(r'''# Probar la demo del Personal Shopper en Colab

Este notebook levanta la app Gradio (`final/app.py`) de punta a punta y te da un **enlace público
temporal** para probarla. Es el paso "desplegar pronto" antes de ir a Hugging Face Spaces.

**Antes de empezar:** `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU` (acelera el
generador y el componente visual). Funciona también en CPU, pero más lento.

Ejecuta las celdas **en orden**.''')

md(r'''## Celda 1 — Instalar dependencias (tarda 5-10 min la primera vez)

`llama-cpp-python` se **compila desde cero** en Colab (no hay binario listo), por eso tarda. Verás
mensajes como `Preparing metadata (pyproject.toml) ... done`: es normal, está compilando, NO está
colgado. Espera a que aparezca el `✅ dependencias listas` del final.''')

code(r'''%pip -q install gradio llama-cpp-python huggingface_hub chromadb sentence-transformers \
    rank-bm25 transformers accelerate pillow pandas numpy
print("✅ dependencias listas")''')

md(r'''## Celda 2 — Clonar el repositorio''')

code(r'''import os
if not os.path.isdir('Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya'):
    !git clone -b cesar https://github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya.git
%cd Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya
print("Contenido de final/:", os.listdir('final'))''')

md(r'''## Celda 3 — (Opcional) Observabilidad en W&B

Si tienes cuenta de W&B, pega aquí tu API key (de https://wandb.ai/authorize). Si la dejas vacía,
el sistema igual funciona y solo escribe el log JSON local. **No subas tu key al repositorio.**''')

code(r'''import os
# os.environ['WANDB_API_KEY'] = 'pega-tu-key-aqui'   # opcional
os.environ['WANDB_PROJECT'] = 'personal-shopper-moda'
# Backend del generador: 'gguf' (rápido en CPU) o 'fp16' (necesita GPU).
os.environ['RAG_LOAD_IN'] = 'gguf'
print('config lista. Juez/generador:', os.environ['RAG_LOAD_IN'])''')

md(r'''## Celda 4 — Lanzar la demo

Arranca la app. La **primera vez** descarga los modelos (embeddings, BLIP y el GGUF del generador);
puede tardar 2-4 min. Al final verás una línea:

> `Running on public URL: https://xxxx.gradio.live`

Ese es tu enlace público temporal (dura ~72 h). Ábrelo para usar la demo: escribe una petición o
sube una foto de una prenda.''')

code(r'''import os, sys
# Añadir final/ al path con RUTA ABSOLUTA (sin cambiar de directorio, para que las rutas al
# corpus 'notebooks/M3' sigan siendo válidas desde la raíz del repo).
assert os.path.isdir('final'), 'Ejecuta primero la celda del git clone + %cd.'
sys.path.insert(0, os.path.abspath('final'))
# La app por defecto busca el corpus en 'notebooks/M3' relativo al cwd (raíz del repo).
import app as _app  # carga modelos al importar (tarda la primera vez)
_app.demo.launch(share=True)''')

md(r'''## Celda 5 — (Opcional) Probar el health check desde código

Sin pasar por la UI, confirma que los modelos cargaron y el índice responde.''')

code(r'''print(_app.health())''')


if __name__ == "__main__":
    build()
