# -*- coding: utf-8 -*-
"""
empaquetar_space.py — Ensambla la carpeta lista para subir a Hugging Face Spaces.

Copia a final/space/build/ todo lo que el Space necesita en su RAÍZ:
  app.py, rag_core.py, vision.py, observabilidad.py, harness.py, requirements.txt, README.md
  y corpus/ (los CSV traducidos).

Uso (desde la raíz del repo):
    python final/space/empaquetar_space.py

Luego subes el contenido de final/space/build/ al repositorio del Space (ver instrucciones
que imprime el script al final).
"""
import os
import shutil

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
FINAL = os.path.join(RAIZ, "final")
SPACE = os.path.join(FINAL, "space")
BUILD = os.path.join(SPACE, "build")
CORPUS_SRC = os.path.join(RAIZ, "notebooks", "M3", "corpus")

MODULOS = ["app.py", "rag_core.py", "vision.py", "observabilidad.py", "harness.py"]


def main():
    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)

    # módulos .py del sistema
    for m in MODULOS:
        shutil.copy2(os.path.join(FINAL, m), os.path.join(BUILD, m))
    # requirements y README específicos del Space
    shutil.copy2(os.path.join(SPACE, "requirements.txt"), os.path.join(BUILD, "requirements.txt"))
    shutil.copy2(os.path.join(SPACE, "README.md"), os.path.join(BUILD, "README.md"))
    # corpus (solo los dos CSV que usa el sistema; NO la fuente en inglés ni la caché)
    os.makedirs(os.path.join(BUILD, "corpus"), exist_ok=True)
    for f in ("corpus_look_es.csv", "corpus_label_es.csv"):
        shutil.copy2(os.path.join(CORPUS_SRC, f), os.path.join(BUILD, "corpus", f))

    print("Carpeta del Space lista en:", BUILD)
    print("Contenido:")
    for root, _, files in os.walk(BUILD):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), BUILD)
            print("  ", rel)
    print("""
--- Cómo subirlo al Space ---
1) En huggingface.co -> New Space -> SDK: Gradio -> nombre: personal-shopper-moda.
2) Clona el Space:  git clone https://huggingface.co/spaces/<usuario>/personal-shopper-moda
3) Copia TODO el contenido de esta carpeta build/ dentro del repo del Space.
4) git add . && git commit -m "deploy" && git push
5) En el Space -> Settings -> Variables and secrets: añade WANDB_API_KEY (y GEMINI_API_KEY si
   usas Gemini). RAG_LOAD_IN=gguf y RAG_USAR_HYDE=0 ya son los valores por defecto.
El Space construirá el entorno y arrancará app.py automáticamente.
""")


if __name__ == "__main__":
    main()
