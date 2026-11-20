# -*- coding: utf-8 -*-
"""
app.py — Interfaz web (Gradio) del personal shopper de moda. Punto de entrada del despliegue.

Une las piezas de los módulos en un solo sistema usable desde una URL pública:
  - M1/M3: `PersonalShopper` (RAG híbrida+HyDE + herramienta + generador Qwen, opcionalmente
    cuantizado para caber en Hugging Face Spaces gratuito).
  - M4: componente visual (BLIP) — el usuario sube una foto de una prenda y el sistema busca algo
    parecido.
  - M5: health check, logging estructurado en JSON y observabilidad en MLflow.

Cómo se ejecuta:
  - Local / Colab:   python final/app.py
  - Hugging Face Spaces: este archivo se llama app.py y Spaces lo ejecuta automáticamente.

Variables de entorno (en Spaces se guardan como SECRETOS, nunca en el repo):
  - RAG_LOAD_IN   : 'fp16' (def.) | 'int8' | 'int4'  -> cuantización del generador.
  - MLFLOW_TRACKING_URI    : (opcional) URI del servidor MLflow; sin ella solo hay log JSON local.
  - MLFLOW_EXPERIMENT_NAME : (opcional) experimento MLflow (def. 'personal-shopper-moda').
  - MLFLOW_TRACKING_TOKEN  : (opcional) token del servidor MLflow (lo lee la librería).
  - RAG_CORPUS_DIR: carpeta que contiene corpus/ y eval/ (def. 'notebooks/M3').
"""

import atexit
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

# --- Carga única de los modelos (al arrancar el Space) ---
print("Cargando sistema (puede tardar en el primer arranque)...")
if LOAD_IN == "gguf":
    shopper = PersonalShopper(corpus_dir=CORPUS_DIR, load_in="gguf",
                              gguf_repo=GGUF_REPO, gguf_file=GGUF_FILE)
else:
    shopper = PersonalShopper(corpus_dir=CORPUS_DIR, load_in=LOAD_IN)
vision = VisionCaptioner()
obs = Observador()
atexit.register(obs.cerrar)


def responder(pregunta, imagen, modo):
    """Maneja una petición de la UI: imagen opcional -> consulta -> RAG -> respuesta + log.
    Mide latencia extremo a extremo (visión + RAG) y captura errores sin romper la UI."""
    import time, traceback
    t_ini = time.perf_counter()
    pregunta = (pregunta or "").strip()
    con_imagen = imagen is not None
    vision_usada = False
    nota_visual = ""

    try:
        if con_imagen:
            t_v = time.perf_counter()
            caption, consulta_img = vision.imagen_a_consulta(imagen)
            lat_vision = time.perf_counter() - t_v
            if consulta_img:                       # se reconoció una prenda
                vision_usada = True
                nota_visual = "Interpreté tu imagen como: *" + caption + "*\n\n"
                pregunta = (pregunta + ". " + consulta_img) if pregunta else consulta_img
            elif caption is not None:              # hay modelo pero la imagen no parece ropa
                nota_visual = ("_No reconocí una prenda en la imagen (la vi como: " + caption +
                               "). Respondo con tu texto si lo hay._\n\n")
            else:                                  # no hay modelo de visión
                nota_visual = "_(Componente visual no disponible; respondo solo con tu texto.)_\n\n"

        if not pregunta:
            return ("Escribe una petición o sube una **foto de una prenda**.", "",
                    "sin consulta")

        salida = shopper.sistema(pregunta, modo=modo)
        salida["valvula"] = shopper.activo_valvula(salida["respuesta"]) or salida.get("valvula_pre", False)
        lat_total = time.perf_counter() - t_ini

        # Detalle de outfits (prendas reales desde los datos, no de la prosa del modelo)
        detalle = ""
        for oid in salida["outfits"][:3]:
            exp = shopper.expandir_outfit(oid)
            if not exp:
                continue
            prendas = "\n".join("   • " + p["titulo"] + " (" + p["color"] + ")"
                                for p in exp["prendas"])
            detalle += ("**" + oid + "** — " + exp["look"][:90] + "\n" + prendas + "\n\n")

        # Logging estructurado + MLflow (sin datos personales). 'vision' solo si se usó de verdad.
        evento = construir_evento(pregunta, salida, modo, con_imagen=vision_usada)
        evento["latencia_total_s"] = round(lat_total, 3)
        evento["latencias_etapa"] = salida.get("latencias", {})
        evento["vision_ok"] = vision_usada
        obs.registrar(evento)

        respuesta_md = nota_visual + salida["respuesta"]
        meta = ("Modo: " + modo + " | herramienta: " +
                ("sí " + str(salida["filtro"]) if salida["uso_herramienta"] else "no") +
                " | latencia total: " + str(round(lat_total, 2)) + " s" +
                (" | VÁLVULA activada" if salida["valvula"] else ""))
        return respuesta_md, (detalle or "_(sin outfits para mostrar)_"), meta

    except Exception as e:
        traceback.print_exc()
        try:
            obs.registrar({"ts": "", "modo": modo, "error": str(e)[:200],
                           "latencia_s": round(time.perf_counter() - t_ini, 3)})
        except Exception:
            pass
        return ("Ocurrió un error procesando tu petición. Intenta de nuevo con otra consulta.",
                "", "error: " + str(e)[:120])


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
        "(precios, envíos, tallas), te lo digo en vez de inventar.\n\n"
        "> ⚠️ **Es una sugerencia, no una garantía.** El catálogo no tiene precios, tallas "
        "numéricas ni envíos. 🔒 **Privacidad:** si subes una foto, se procesa solo para esta "
        "recomendación y **no se guarda**; los registros solo almacenan métricas (latencia, si se "
        "usó la herramienta), nunca tu texto ni tu imagen."
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

    gr.Markdown("---\n### Estado del sistema (health check)\n"
                "Se expone como API con nombre `health`. Desde código:\n"
                "`from gradio_client import Client; "
                "print(Client('<url-del-space>').predict(api_name='/health'))`")
    hc = gr.Textbox(label="health", interactive=False)
    gr.Button("Comprobar estado").click(health, outputs=hc, api_name="health")

    gr.Examples(
        examples=[
            ["algo elegante y minimalista para la oficina en invierno, tonos oscuros", None, "final"],
            ["ropa de hombre estilo outdoor con pantalones cargo para el frío", None, "final"],
            ["¿cuánto cuesta el envío a Medellín?", None, "avanzado"],
        ],
        inputs=[txt, img, modo],
    )


if __name__ == "__main__":
    # Cola con concurrencia 1: llama.cpp no es seguro con llamadas concurrentes al mismo contexto.
    demo.queue(max_size=16, default_concurrency_limit=1)
    demo.launch()
