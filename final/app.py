import json
import gradio as gr
from PIL import Image

from observability import obtener_resumen_metricas
from pipeline import PersonalShopperOptimizado

shopper = PersonalShopperOptimizado(usar_4bit=True, usar_cache=True)


def procesar_consulta(texto: str, imagen: Image.Image | None):
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
    salud = shopper.health_check()
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
    demo.launch(server_name="0.0.0.0", server_port=7860)