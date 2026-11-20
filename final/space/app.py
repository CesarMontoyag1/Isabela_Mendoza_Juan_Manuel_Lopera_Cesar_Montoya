# -*- coding: utf-8 -*-
"""
app.py — Interfaz Gradio del personal shopper, adaptada a Hugging Face Spaces ZeroGPU.

ZeroGPU asigna GPU por petición solo dentro de funciones decoradas con @spaces.GPU. Por eso:
  - El generador usa el backend Transformers en GPU (fp16), NO GGUF (que es para CPU).
  - La función que genera la recomendación va decorada con @spaces.GPU.
  - Los modelos se cargan de forma perezosa en la primera llamada (dentro del contexto GPU).

Une M1/M3 (RAG), M4 (visión BLIP) y M5 (health, logging, W&B). Punto de entrada del Space.

Variables de entorno (SECRETOS del Space, nunca en el repo):
  WANDB_API_KEY, WANDB_PROJECT, GEMINI_API_KEY (opcional), RAG_USAR_HYDE (def. '0').
"""

import os
import gradio as gr

try:
    import spaces  # disponible en Spaces ZeroGPU
    EN_ZEROGPU = True
except Exception:
    # Permite correr este mismo app localmente/Colab sin el paquete 'spaces'.
    EN_ZEROGPU = False
    class _Dummy:
        def GPU(self, *a, **k):
            def deco(f):
                return f
            return deco
    spaces = _Dummy()

from rag_core import PersonalShopper
from vision import VisionCaptioner
from observabilidad import Observador, construir_evento

CORPUS_DIR = os.environ.get("RAG_CORPUS_DIR", ".")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "personal-shopper-moda")
os.environ.setdefault("RAG_USAR_HYDE", "0")  # en el Space, HyDE desactivado por latencia

# Carga perezosa: en ZeroGPU los modelos deben cargarse con la GPU asignada (dentro de @spaces.GPU).
_sistema = {"shopper": None, "vision": None}
obs = Observador(wandb_project=WANDB_PROJECT)


def _asegurar_cargado():
    if _sistema["shopper"] is None:
        print("Cargando sistema (primera petición)...")
        # fp16 en GPU (ZeroGPU). Si no hay GPU (local), cae a fp32/CPU automáticamente.
        _sistema["shopper"] = PersonalShopper(corpus_dir=CORPUS_DIR, load_in="fp16")
        _sistema["vision"] = VisionCaptioner()
    return _sistema["shopper"], _sistema["vision"]


@spaces.GPU(duration=120)
def responder(pregunta, imagen, modo):
    """Petición extremo a extremo (visión + RAG). Decorada para usar la GPU de ZeroGPU."""
    import time, traceback
    t_ini = time.perf_counter()
    pregunta = (pregunta or "").strip()
    con_imagen = imagen is not None
    vision_usada = False
    nota_visual = ""
    try:
        shopper, vision = _asegurar_cargado()

        if con_imagen:
            caption, consulta_img = vision.imagen_a_consulta(imagen)
            if consulta_img:
                vision_usada = True
                nota_visual = "Interpreté tu imagen como: *" + caption + "*\n\n"
                pregunta = (pregunta + ". " + consulta_img) if pregunta else consulta_img
            elif caption is not None:
                nota_visual = ("_No reconocí una prenda en la imagen (la vi como: " + caption +
                               "). Respondo con tu texto si lo hay._\n\n")
            else:
                nota_visual = "_(Componente visual no disponible; respondo solo con tu texto.)_\n\n"

        if not pregunta:
            return ("Escribe una petición o sube una **foto de una prenda**.", "", "sin consulta")

        salida = shopper.sistema(pregunta, modo=modo)
        salida["valvula"] = shopper.activo_valvula(salida["respuesta"]) or salida.get("valvula_pre", False)
        lat_total = time.perf_counter() - t_ini

        detalle = ""
        for oid in salida["outfits"][:3]:
            exp = shopper.expandir_outfit(oid)
            if not exp:
                continue
            prendas = "\n".join("   • " + p["titulo"] + " (" + p["color"] + ")"
                                for p in exp["prendas"])
            detalle += ("**" + oid + "** — " + exp["look"][:90] + "\n" + prendas + "\n\n")

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
        return ("Ocurrió un error procesando tu petición. Intenta de nuevo con otra consulta.",
                "", "error: " + str(e)[:120])


@spaces.GPU(duration=60)
def health():
    import json
    shopper, _ = _asegurar_cargado()
    return json.dumps(shopper.health_check(), ensure_ascii=False, indent=2)


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
            modo = gr.Radio(["final", "avanzado", "ingenuo", "baseline"], value="final",
                            label="Versión del sistema",
                            info="final = híbrida + herramienta; avanzado = + HyDE")
            btn = gr.Button("Recomiéndame", variant="primary")
        with gr.Column(scale=4):
            out_resp = gr.Markdown(label="Recomendación")
            out_det = gr.Markdown(label="Prendas de los outfits recomendados")
            out_meta = gr.Textbox(label="Trazabilidad de la petición", interactive=False)

    btn.click(responder, inputs=[txt, img, modo], outputs=[out_resp, out_det, out_meta])

    gr.Markdown("---\n### Estado del sistema (health check)\n"
                "Botón que comprueba que el índice responde y el generador está cargado.")
    hc = gr.Textbox(label="health", interactive=False)
    gr.Button("Comprobar estado").click(health, outputs=hc)

    # Ejemplos solo de texto (sin columna de imagen, que rompía gr.Examples con None).
    gr.Examples(
        examples=[
            ["algo elegante y minimalista para la oficina en invierno, tonos oscuros"],
            ["ropa de hombre estilo outdoor con pantalones cargo para el frío"],
            ["un look de chica dulce y atrevida con minifalda y botas altas"],
            ["¿cuánto cuesta el envío a Medellín?"],
        ],
        inputs=[txt],
    )


if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=1)
    # show_api=False evita el bug de get_api_info ('bool is not iterable') en gradio 4.44.x.
    # El health check sigue disponible por el botón de la UI.
    demo.launch(show_api=False)
