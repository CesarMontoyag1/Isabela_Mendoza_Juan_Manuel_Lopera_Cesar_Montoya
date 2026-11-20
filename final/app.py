import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import spaces
import json
import gradio as gr
import gradio_client.utils as _gc_utils
from PIL import Image

_orig_schema_to_py = _gc_utils._json_schema_to_python_type
def _safe_schema_to_py(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_schema_to_py(schema, defs)

_gc_utils._json_schema_to_python_type = _safe_schema_to_py

from observability import obtener_resumen_metricas
from pipeline import PersonalShopperOptimizado

_shopper = None

def get_shopper():
    global _shopper
    if _shopper is None:
        _shopper = PersonalShopperOptimizado()
    return _shopper

@spaces.GPU(duration=120)
def procesar_consulta(texto, imagen=None):
    shopper = get_shopper()
    res = shopper.consultar(texto, imagen=imagen, registrar_log=True)

    traza = {
        "latencia_s": round(res["latencia_s"], 3),
        "cache_hit": res["cache_hit"],
        "valvula_activada": res["valvula_activada"],
        "perfil_extraido_M1": res["perfil_m1"],
        "analisis_visual_M4": res["info_visual"],
        "filtros_catalogo_M3": res["filtro"],
        "outfits_citables": res["outfits"],
    }
    return res["respuesta"], json.dumps(traza, ensure_ascii=False, indent=2)


def ver_salud_y_observabilidad():
    if _shopper is None:
        salud = {
            "status": "ready (lazy-init on first GPU request)",
            "device": "zerogpu (cuda on demand)",
            "catalog_outfits": 800,
        }
    else:
        salud = _shopper.health_check()
    metricas = obtener_resumen_metricas(ultimos_n=100)
    return json.dumps(salud, ensure_ascii=False, indent=2), json.dumps(metricas, ensure_ascii=False, indent=2)


with gr.Blocks(title="Personal Shopper Multimodal IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Personal Shopper Multimodal (M1 + M3 + M4 + M5)\n"
        "Asistente de moda con extracción de perfil LoRA (`Qwen2.5-1.5B`), recuperación híbrida "
        "(`BM25 + e5-small + HyDE`) y percepción visual (`Qwen2-VL-2B`)."
    )

    with gr.Tab("Asistente Personal Shopper"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_txt = gr.Textbox(
                    label="¿Qué estás buscando hoy?",
                    placeholder="Ej: Busco un outfit casual y cómodo para clima cálido de fin de semana...",
                    lines=3,
                )
                inp_img = gr.Image(label="Imagen de referencia opcional (prenda o estilo)", type="pil")
                btn_enviar = gr.Button("Recomendar Outfits", variant="primary")

                gr.Examples(
                    examples=[
                        ["Necesito un outfit formal de hombre para una reunión de trabajo en clima frío.", None],
                        ["Busco un look casual y holgado de mujer para viajar en verano.", None],
                        ["¿Cuánto cuesta el envío a Medellín y qué tallas tienen?", None],
                    ],
                    inputs=[inp_txt, inp_img],
                )

            with gr.Column(scale=1):
                out_resp = gr.Textbox(label="Recomendación del Personal Shopper", lines=8)
                out_traza = gr.Code(label="Traza del Sistema (M1 Perfil + M4 Visión + M3 Filtros)", language="json")

        btn_enviar.click(fn=procesar_consulta, inputs=[inp_txt, inp_img], outputs=[out_resp, out_traza])

    with gr.Tab("Health Check & Observabilidad"):
        gr.Markdown("Estado del servicio y métricas estructuradas en tiempo real (sin almacenamiento de PII).")
        btn_refresh = gr.Button("Actualizar Estado y Métricas")
        with gr.Row():
            out_health = gr.Code(label="Endpoint /health", language="json")
            out_obs = gr.Code(label="Logs Estructurados (p50 / p95 / tasas)", language="json")

        btn_refresh.click(fn=ver_salud_y_observabilidad, inputs=[], outputs=[out_health, out_obs])
        demo.load(fn=ver_salud_y_observabilidad, inputs=[], outputs=[out_health, out_obs])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)