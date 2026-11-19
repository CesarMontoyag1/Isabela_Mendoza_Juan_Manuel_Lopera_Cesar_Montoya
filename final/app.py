# -*- coding: utf-8 -*-
"""
app.py — Interfaz web (Gradio) del personal shopper de moda. Punto de entrada del despliegue.

Une las piezas de los módulos en un solo sistema usable desde una URL pública:
  - M1/M3: `PersonalShopper` (RAG híbrida+HyDE + herramienta + generador Qwen, opcionalmente
    cuantizado para caber en Hugging Face Spaces gratuito).
  - M4: componente visual (BLIP) — el usuario sube una foto de una prenda y el sistema busca algo
    parecido.
  - M5: health check, logging estructurado en JSON y observabilidad en W&B.

Cómo se ejecuta:
  - Local / Colab:   python final/app.py
  - Hugging Face Spaces: este archivo se llama app.py y Spaces lo ejecuta automáticamente.

Variables de entorno (en Spaces se guardan como SECRETOS, nunca en el repo):
  - RAG_LOAD_IN   : 'fp16' (def.) | 'int8' | 'int4'  -> cuantización del generador.
  - WANDB_API_KEY : (opcional) activa la observabilidad en W&B.
  - WANDB_PROJECT : (opcional) nombre del proyecto W&B (def. 'personal-shopper-moda').
  - RAG_CORPUS_DIR: carpeta que contiene corpus/ y eval/ (def. 'notebooks/M3').
"""

import os
import gradio as gr

from rag_core import PersonalShopper
from vision import VisionCaptioner
from observabilidad import Observador, construir_evento


# --- Configuración desde el entorno (reproducible, sin secretos en el código) ---
CORPUS_DIR = os.environ.get("RAG_CORPUS_DIR", "notebooks/M3")
# En Spaces (CPU) usamos GGUF int4 por defecto: rápido en CPU y cabe en el hardware gratuito.
LOAD_IN    = os.environ.get("RAG_LOAD_IN", "gguf")
GGUF_REPO  = os.environ.get("RAG_GGUF_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
GGUF_FILE  = os.environ.get("RAG_GGUF_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "personal-shopper-moda")

# --- Carga única de los modelos (al arrancar el Space) ---
print("Cargando sistema (puede tardar en el primer arranque)...")
if LOAD_IN == "gguf":
    shopper = PersonalShopper(corpus_dir=CORPUS_DIR, load_in="gguf",
                              gguf_repo=GGUF_REPO, gguf_file=GGUF_FILE)
else:
    shopper = PersonalShopper(corpus_dir=CORPUS_DIR, load_in=LOAD_IN)
vision = VisionCaptioner()
obs = Observador(wandb_project=WANDB_PROJECT)


def responder(pregunta, imagen, modo):
    """Maneja una petición de la UI: imagen opcional -> consulta -> RAG -> respuesta + log."""
    con_imagen = imagen is not None
    nota_visual = ""
    if con_imagen:
        caption, consulta_img = vision.imagen_a_consulta(imagen)
        if consulta_img:
            nota_visual = "Interpreté tu imagen como: *" + caption + "*\n\n"
            # si el usuario no escribió nada, usamos la consulta derivada de la imagen
            pregunta = (pregunta.strip() + ". " + consulta_img) if pregunta.strip() else consulta_img
        else:
            nota_visual = "_(No pude procesar la imagen; respondo solo con tu texto.)_\n\n"

    if not (pregunta or "").strip():
        return "Escribe una petición o sube una imagen de una prenda.", "", "{}"

    salida = shopper.sistema(pregunta, modo=modo)
    salida["valvula"] = shopper.activo_valvula(salida["respuesta"])

    # Detalle de outfits recomendados (prendas reales desde los datos)
    detalle = ""
    for oid in salida["outfits"][:3]:
        exp = shopper.expandir_outfit(oid)
        if not exp:
            continue
        prendas = "\n".join("   • " + p["titulo"] + " (" + p["color"] + ")" for p in exp["prendas"])
        detalle += ("**" + oid + "** — " + exp["look"][:90] + "\n" + prendas + "\n\n")

    # Logging estructurado + W&B (sin datos personales)
    evento = construir_evento(pregunta, salida, modo, con_imagen=con_imagen)
    obs.registrar(evento)

    respuesta_md = nota_visual + salida["respuesta"]
    meta = ("Modo: " + modo + " | herramienta: " +
            ("sí " + str(salida["filtro"]) if salida["uso_herramienta"] else "no") +
            " | latencia: " + str(round(salida["latencia_s"], 2)) + " s" +
            (" | válvula activada" if salida["valvula"] else ""))
    return respuesta_md, (detalle or "_(sin outfits para mostrar)_"), meta


def health():
    """Health check: estado del sistema sin hacer una pregunta real."""
    import json
    return json.dumps(shopper.health_check(), ensure_ascii=False, indent=2)


# --- UI ---
with gr.Blocks(title="Personal Shopper de Moda") as demo:
    gr.Markdown(
        "# 👗 Personal Shopper de Moda\n"
        "Pídeme un outfit con tus palabras (o **sube una foto** de una prenda) y te recomiendo "
        "conjuntos **reales del catálogo**. Si preguntas algo que no está en el catálogo "
        "(precios, envíos, tallas), te lo digo en vez de inventar."
    )
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="¿Qué buscas?",
                             placeholder="Ej: algo elegante y abrigado para una boda en invierno")
            img = gr.Image(label="O sube una foto de una prenda (opcional)", type="pil")
            modo = gr.Radio(["final", "avanzado", "ingenuo", "baseline"], value="avanzado",
                            label="Versión del sistema",
                            info="avanzado = híbrida+HyDE; final = + herramienta de filtro")
            btn = gr.Button("Recomiéndame", variant="primary")
        with gr.Column(scale=4):
            out_resp = gr.Markdown(label="Recomendación")
            out_det = gr.Markdown(label="Prendas de los outfits recomendados")
            out_meta = gr.Textbox(label="Trazabilidad de la petición", interactive=False)

    btn.click(responder, inputs=[txt, img, modo], outputs=[out_resp, out_det, out_meta])

    gr.Markdown("---\n### Estado del sistema (health check)")
    hc = gr.Textbox(label="health", interactive=False)
    gr.Button("Comprobar estado").click(health, outputs=hc)

    gr.Examples(
        examples=[
            ["algo elegante y minimalista para la oficina en invierno, tonos oscuros", None, "final"],
            ["ropa de hombre estilo outdoor con pantalones cargo para el frío", None, "final"],
            ["¿cuánto cuesta el envío a Medellín?", None, "avanzado"],
        ],
        inputs=[txt, img, modo],
    )


if __name__ == "__main__":
    demo.launch()
