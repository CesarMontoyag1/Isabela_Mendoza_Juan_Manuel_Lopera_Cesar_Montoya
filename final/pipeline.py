import collections
import csv
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import torch
from peft import PeftModel
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)

from observability import registrar_evento

BASE_DIR = Path(__file__).parent
LOOK_CSV = BASE_DIR / "corpus" / "corpus_look_es.csv"
LABEL_CSV = BASE_DIR / "corpus" / "corpus_label_es.csv"
ADAPTER_PATH = BASE_DIR / "adapter_final"
CHROMA_DIR = BASE_DIR / "chroma_db_final"

GEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VIS_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5
VALVULA = "No encuentro esa información en el catálogo"

ONTOLOGIA_M1 = {
    "estilo": ["Casual", "Formal", "Minimalista", "Urbano", "Bohemio", "Deportivo", "Clasico"],
    "ocasion": ["boda", "trabajo", "fin_de_semana", "viaje", "deporte", "evento_formal"],
    "clima": ["calido", "frio", "templado"],
    "paleta": ["neutros", "pasteles", "oscuros", "colores_vivos", "monocromatico"],
    "fit": ["holgado", "regular", "ajustado", "oversized"],
}

SYSTEM_PROMPT_M1 = (
    "Eres un asistente que extrae el perfil de estilo de un cliente a partir de su mensaje "
    "de chat en una tienda de moda online.\n"
    "Debes devolver UNICAMENTE un objeto JSON con los campos que el cliente haya mencionado "
    "o se puedan inferir claramente, usando SOLO estos campos y valores posibles:\n\n"
    "- estilo: Casual, Formal, Minimalista, Urbano, Bohemio, Deportivo, Clasico\n"
    "- ocasion: boda, trabajo, fin_de_semana, viaje, deporte, evento_formal\n"
    "- clima: calido, frio, templado\n"
    "- paleta: neutros, pasteles, oscuros, colores_vivos, monocromatico\n"
    "- fit: holgado, regular, ajustado, oversized\n\n"
    "No incluyas campos que no se puedan inferir del mensaje. No agregues texto fuera del "
    "JSON. No inventes valores fuera de esta lista."
)

VISUAL_SYSTEM_PROMPT = (
    "Eres el módulo de visión de un personal shopper de moda. Analiza la imagen y responde "
    "ÚNICAMENTE con un JSON válido:\n"
    '{"es_moda": true/false, "genero_aparente": "De mujer"|"De hombre"|"Unisex"|null, '
    '"clima_sugerido": "frio"|"calido"|"templado"|null, '
    '"descripcion": "máximo 12 palabras con prenda, color, corte y estilo"}\n'
    "Si NO muestra ropa, calzado o accesorios de moda (ej. autos, comida, gráficos, ruido), "
    'devuelve {"es_moda": false, "genero_aparente": null, "clima_sugerido": null, "descripcion": "OUT_OF_DOMAIN"}.'
)

SYSTEM_PROMPT_M3 = (
    "Eres un personal shopper de un e-commerce de moda. Respondes en español, breve y concreto.\n"
    "REGLA 1 (recomendar): si el CONTEXTO contiene outfits de ropa, SIEMPRE recomienda el o los "
    "que mejor encajen con la petición y cítalos por su identificador (por ejemplo, B_b12). "
    "El contexto son outfits de ropa; úsalos aunque no sean una coincidencia perfecta.\n"
    'REGLA 2 (válvula de escape): usa la frase "No encuentro esa información en el catálogo" '
    "ÚNICAMENTE cuando la pregunta NO es sobre recomendar ropa, sino sobre precios, envíos, "
    "tallas numéricas (38, 40...) o políticas de la tienda (devoluciones). En esos casos no hay "
    "outfit que recomendar.\n"
    "NUNCA uses la válvula si la pregunta pide ropa y el contexto trae outfits. No inventes prendas "
    "que no estén en el contexto."
)

SINONIMOS_GENERO = {
    "De hombre": ["hombre", "masculino", "caballero", "hombres", "para él", "para el"],
    "De mujer": ["mujer", "femenino", "dama", "mujeres", "para ella", "chica", "vestido", "falda"],
}
MAPA_CLIMA_M1_A_M3 = {"frio": "Invierno", "calido": "Verano", "templado": "Primavera"}
MAPA_OCASION_M1_A_M3 = {
    "boda": "Eventos sociales",
    "evento_formal": "Eventos sociales",
    "trabajo": "Trabajo",
    "viaje": "Viaje",
    "deporte": "Deporte",
}


def parse_json_seguro(texto: str) -> Dict[str, Any]:
    if not texto:
        return {}
    ini, fin = texto.find("{"), texto.rfind("}")
    if ini == -1 or fin == -1 or fin < ini:
        return {}
    try:
        obj = json.loads(texto[ini : fin + 1])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class PersonalShopperOptimizado:
    """Pipeline integrado M1 + M3 + M4 compatible con ZeroGPU (CPU init -> CUDA inference)."""

    def __init__(self, usar_4bit: bool = False, usar_cache: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.usar_4bit = False
        self.usar_cache = usar_cache
        self._cache: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()
        self._max_cache = 64

        self._cargar_corpus_e_indice()
        self._cargar_modelos()

    def _cargar_corpus_e_indice(self):
        def _norm(v):
            mapa = {"Business": "Trabajo"}
            return "/".join(mapa.get(p.strip(), p.strip()) for p in str(v).split("/"))

        with open(LOOK_CSV, encoding="utf-8-sig", newline="") as f:
            looks = list(csv.DictReader(f))
        with open(LABEL_CSV, encoding="utf-8-sig", newline="") as f:
            labels = list(csv.DictReader(f))

        labels_by_id = {row["itemID"]: row for row in labels}

        def genero_outfit(items):
            gens = [labels_by_id[i]["gender"] for i in items if i in labels_by_id]
            for g in ("De mujer", "De hombre"):
                if g in gens:
                    return g
            return "Unisex"

        self.documentos, self.metadatos, self.doc_ids = [], [], []
        for o in looks:
            items = [i.strip() for i in o["items"].split(",") if i.strip()]
            partes = [
                f"Outfit {o['outfitID']}: {o['look']}",
                f"Estación: {_norm(o['season'])}. Ocasión: {_norm(o['occasion'])}.",
            ]
            detalles = []
            for i in items:
                p = labels_by_id.get(i)
                if p:
                    d = f"- {p['title']} ({p['category']}): {p['color']}, {p['materials']}, {p['outline']}"
                    if p.get("detail"):
                        d += f". {p['detail']}"
                    detalles.append(d)
            if detalles:
                partes.append("Prendas:")
                partes.extend(detalles)

            self.documentos.append("\n".join(partes))
            self.doc_ids.append(o["outfitID"])
            self.metadatos.append({
                "outfitID": o["outfitID"],
                "season": _norm(o["season"]),
                "occasion": _norm(o["occasion"]),
                "gender": genero_outfit(items),
            })

        self.embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        client = chromadb.EphemeralClient()
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        client = chromadb.EphemeralClient()
        self.col = client.create_collection("outfits_m5", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(self.documentos), 128):
            sl = slice(i, i + 128)
            embs = self.embedder.encode(
                ["passage: " + t for t in self.documentos[sl]],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            self.col.add(
                ids=self.doc_ids[sl],
                documents=self.documentos[sl],
                embeddings=embs,
                metadatas=self.metadatos[sl],
            )
        self.docs_by_id = dict(zip(self.doc_ids, self.documentos))
        self.bm25 = BM25Okapi([re.findall(r"\w+", d.lower()) for d in self.documentos])

    def _cargar_modelos(self):
        # En ZeroGPU al iniciar no hay GPU: cargamos en CPU en bfloat16 y movemos a CUDA dentro de @spaces.GPU
        self.gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        if self.gen_tok.pad_token is None:
            self.gen_tok.pad_token = self.gen_tok.eos_token

        torch.set_num_threads(max(1, os.cpu_count() or 2))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_gen = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        if ADAPTER_PATH.exists():
            self.gen_model = PeftModel.from_pretrained(base_gen, str(ADAPTER_PATH)).to(self.device)
            self.tiene_lora = True
        else:
            self.gen_model = base_gen
            self.tiene_lora = False

        self.gen_model.eval()
        gc.collect()

        self.vis_processor = None
        self.vis_model = None

    @torch.inference_mode()
    def _analizar_imagen(self, imagen: Image.Image) -> Dict[str, Any]:
        if imagen.mode != "RGB":
            imagen = imagen.convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VISUAL_SYSTEM_PROMPT}]}]
        prompt_txt = self.vis_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.vis_processor(text=[prompt_txt], images=[imagen], padding=True, return_tensors="pt").to(self.device)

        out = self.vis_model.generate(
            **inputs,
            max_new_tokens=75,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.vis_processor.tokenizer.eos_token_id,
        )
        raw = self.vis_processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        parsed = parse_json_seguro(raw)
        if not parsed:
            es_ood = "out_of_domain" in raw.lower() or "no muestra ropa" in raw.lower()
            return {
                "es_moda": not es_ood,
                "genero_aparente": None,
                "clima_sugerido": None,
                "descripcion": "OUT_OF_DOMAIN" if es_ood else raw[:140],
            }
        return {
            "es_moda": bool(parsed.get("es_moda", False)),
            "genero_aparente": parsed.get("genero_aparente"),
            "clima_sugerido": parsed.get("clima_sugerido"),
            "descripcion": str(parsed.get("descripcion", "")).strip(),
        }

    @torch.inference_mode()
    def _extraer_perfil_m1(self, texto: str) -> Dict[str, str]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_M1}, {"role": "user", "content": texto}]
        prompt = self.gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.gen_tok(prompt, return_tensors="pt").to(self.device)
        out = self.gen_model.generate(
            **inputs, max_new_tokens=48, do_sample=False, use_cache=True, pad_token_id=self.gen_tok.eos_token_id
        )
        raw = self.gen_tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        obj = parse_json_seguro(raw)
        return {k: v for k, v in obj.items() if k in ONTOLOGIA_M1 and v in ONTOLOGIA_M1[k]}

    @torch.inference_mode()
    def _generar_base(self, sys_p: str, usr_p: str, max_tokens: int = 240) -> str:
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        prompt = self.gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.gen_tok(prompt, return_tensors="pt").to(self.device)
        if self.tiene_lora:
            with self.gen_model.disable_adapter():
                out = self.gen_model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True, pad_token_id=self.gen_tok.eos_token_id
                )
        else:
            out = self.gen_model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True, pad_token_id=self.gen_tok.eos_token_id
            )
        return self.gen_tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def _buscar_denso(self, q_txt: str, k: int = TOP_K):
        q_emb = self.embedder.encode(["query: " + q_txt], normalize_embeddings=True, show_progress_bar=False).tolist()
        res = self.col.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas"])
        return list(zip([m["outfitID"] for m in res["metadatas"][0]], res["documents"][0]))

    def _buscar_bm25(self, q_txt: str, k: int = 20, candidatos_idx=None):
        scores = self.bm25.get_scores(re.findall(r"\w+", q_txt.lower()))
        idx = range(len(self.documentos)) if candidatos_idx is None else candidatos_idx
        ranked = sorted(idx, key=lambda i: scores[i], reverse=True)[:k]
        return [(self.doc_ids[i], self.documentos[i]) for i in ranked]

    def _rrf(self, listas, k_rrf: int = 60, top: int = TOP_K):
        puntajes = collections.defaultdict(float)
        docs = {}
        for lista in listas:
            for rank, (i, d) in enumerate(lista):
                puntajes[i] += 1.0 / (k_rrf + rank)
                docs[i] = d
        return [(i, docs[i]) for i in sorted(puntajes, key=puntajes.get, reverse=True)[:top]]

    def _clave_cache(self, pregunta: str, imagen: Optional[Image.Image]) -> str:
        h = hashlib.sha256(pregunta.strip().lower().encode("utf-8"))
        if imagen is not None:
            thumb = imagen.convert("RGB").resize((32, 32))
            h.update(thumb.tobytes())
        return h.hexdigest()

    def health_check(self) -> Dict[str, Any]:
        """Endpoint lógico de verificación de salud para monitoreo y HF Spaces."""
        vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
        return {
            "status": "healthy" if self.col.count() > 0 else "degraded",
            "device": "zerogpu (cuda on demand)",
            "precision": "bfloat16",
            "lora_loaded": self.tiene_lora,
            "catalog_outfits": self.col.count(),
            "vram_max_gb": vram_gb,
        }

    def consultar(self, pregunta: str, imagen: Optional[Image.Image] = None, registrar_log: bool = True) -> Dict[str, Any]:
        # Dentro de @spaces.GPU ya tenemos GPU asignada: actualizamos self.device y movemos los modelos a CUDA si aún no lo están
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != target_device or next(self.gen_model.parameters()).device.type != target_device:
            self.device = target_device
            self.gen_model.to(self.device)
            self.embedder.to(self.device)

        if imagen is not None:
            if self.vis_model is None:
                self.vis_processor = AutoProcessor.from_pretrained(
                    VIS_MODEL_ID,
                    min_pixels=224 * 28 * 28,
                    max_pixels=448 * 28 * 28,
                )
                self.vis_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    VIS_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                self.vis_model.eval()
                gc.collect()
            if next(self.vis_model.parameters()).device.type != self.device:
                self.vis_model.to(self.device)

        t0 = time.time()
        if not pregunta or not pregunta.strip():
            pregunta = "Recomiéndame un outfit del catálogo que combine con esta imagen."

        clave = self._clave_cache(pregunta, imagen)
        if self.usar_cache and clave in self._cache:
            res_cached = dict(self._cache[clave])
            res_cached["latencia_s"] = time.time() - t0
            res_cached["cache_hit"] = True
            if registrar_log:
                registrar_evento(
                    consulta=pregunta,
                    tiene_imagen=imagen is not None,
                    es_moda=True,
                    uso_herramienta=res_cached["uso_herramienta"],
                    valvula_activada=res_cached["valvula_activada"],
                    outfits_recuperados=res_cached["outfits"],
                    latencia_s=res_cached["latencia_s"],
                    cache_hit=True,
                    estado="ok",
                )
            return res_cached

        try:
            info_visual = None
            consulta_enr = pregunta.strip()

            # 1. Módulo Visual M4
            if imagen is not None:
                info_visual = self._analizar_imagen(imagen)
                if not info_visual["es_moda"]:
                    lat = time.time() - t0
                    salida = {
                        "respuesta": f"{VALVULA}: la imagen adjunta no corresponde a prendas de vestir o moda.",
                        "contextos": [],
                        "outfits": [],
                        "latencia_s": lat,
                        "uso_vision": True,
                        "uso_herramienta": False,
                        "valvula_activada": True,
                        "perfil_m1": {},
                        "info_visual": info_visual,
                        "filtro": {},
                        "cache_hit": False,
                    }
                    if registrar_log:
                        registrar_evento(
                            consulta=pregunta,
                            tiene_imagen=True,
                            es_moda=False,
                            uso_herramienta=False,
                            valvula_activada=True,
                            outfits_recuperados=[],
                            latencia_s=lat,
                            estado="valvula_visual",
                        )
                    return salida
                consulta_enr = f"{pregunta}. Prenda o referencia visual adjunta: {info_visual['descripcion']}"

            # 1.5 Guardrail determinista pre-RAG para consultas administrativas / fuera de catálogo
            q_check = consulta_enr.lower()
            patrones_ood = [
                r"\benv[ií]os?\b",
                r"\bdomicilios?\b",
                r"\bcu[aá]nto (cuesta|vale|salen?)\b",
                r"\bprecios?\b",
                r"\bdevoluci[oó]n(es)?\b",
                r"\bcambios?\b",
                r"\bgarant[ií]a\b",
                r"\bmedios? de pago\b",
                r"\bcontraentrega\b",
                r"\btallas?\b",
                r"\bmedidas?\b",
            ]
            palabras_prenda = [
                "outfit", "look", "ropa", "chaqueta", "camisa", "pantal",
                "vestido", "falda", "abrigo", "zapatos", "tenis", "botines",
                "combinar", "recomienda", "recomiéndame", "busco un", "busco una",
            ]
            es_consulta_admin = any(re.search(p, q_check) for p in patrones_ood)
            pide_outfit = any(w in q_check for w in palabras_prenda)

            if es_consulta_admin and not pide_outfit and imagen is None:
                lat = time.time() - t0
                respuesta_ood = (
                    f"{VALVULA}: este asistente únicamente recomienda combinaciones de prendas "
                    "del catálogo y no gestiona precios, disponibilidad de tallas, envíos ni devoluciones."
                )
                salida = {
                    "respuesta": respuesta_ood,
                    "contextos": [],
                    "outfits": [],
                    "latencia_s": lat,
                    "uso_vision": False,
                    "uso_herramienta": False,
                    "valvula_activada": True,
                    "perfil_m1": {},
                    "info_visual": None,
                    "filtro": {},
                    "cache_hit": False,
                }
                if registrar_log:
                    registrar_evento(
                        consulta=pregunta,
                        tiene_imagen=False,
                        es_moda=False,
                        uso_herramienta=False,
                        valvula_activada=True,
                        outfits_recuperados=[],
                        latencia_s=lat,
                        estado="valvula_pre_rag",
                    )
                return salida

            # 2. Extractor LoRA M1
            perfil_m1 = self._extraer_perfil_m1(consulta_enr)

            # 3. Filtro estructurado + Recuperación Híbrida + HyDE condicional
            args: Dict[str, str] = {}
            q_low = consulta_enr.lower()
            for canon, claves in SINONIMOS_GENERO.items():
                if any(k in q_low for k in claves):
                    args["gender"] = canon
                    break
            if "gender" not in args and info_visual and info_visual.get("genero_aparente") in ("De mujer", "De hombre"):
                args["gender"] = info_visual["genero_aparente"]

            if perfil_m1.get("clima") in MAPA_CLIMA_M1_A_M3:
                args["season"] = MAPA_CLIMA_M1_A_M3[perfil_m1["clima"]]
            elif info_visual and info_visual.get("clima_sugerido") in MAPA_CLIMA_M1_A_M3:
                args["season"] = MAPA_CLIMA_M1_A_M3[info_visual["clima_sugerido"]]

            if perfil_m1.get("ocasion") in MAPA_OCASION_M1_A_M3:
                args["occasion"] = MAPA_OCASION_M1_A_M3[perfil_m1["ocasion"]]

            idx_validos = (
                [k for k, m in enumerate(self.metadatos) if all(v in m.get(c, "") for c, v in args.items())]
                if args
                else list(range(len(self.documentos)))
            )
            if not idx_validos:
                idx_validos = list(range(len(self.documentos)))

            pool = 20
            sub_ids = set(self.doc_ids[i] for i in idx_validos)
            densos = [(i, d) for i, d in self._buscar_denso(consulta_enr, k=pool * 3) if i in sub_ids][:pool]
            lexicos = self._buscar_bm25(consulta_enr, k=pool, candidatos_idx=idx_validos)

            if perfil_m1:
                hip = (
                    f"Outfit {perfil_m1.get('estilo', 'casual')} para {perfil_m1.get('ocasion', 'uso diario')}, "
                    f"clima {perfil_m1.get('clima', 'templado')}, paleta {perfil_m1.get('paleta', 'neutra')}, "
                    f"corte {perfil_m1.get('fit', 'regular')}. {consulta_enr}"
                )
            else:
                hip = f"Conjunto de ropa y prendas del catálogo para: {consulta_enr}"

            hyde_densos = [(i, d) for i, d in self._buscar_denso(hip, k=pool * 3) if i in sub_ids][:pool]
            finalistas = self._rrf([densos, lexicos, hyde_densos], top=TOP_K) or self._buscar_denso(consulta_enr, k=TOP_K)
            contextos = [d for _, d in finalistas]

            # 4. Generación Final con descripción explícita + ficha exacta del catálogo
            ctx_txt = "\n\n".join(contextos) if contextos else "(sin resultados)"
            prompt_final = (
                f"PERFIL: {json.dumps(perfil_m1, ensure_ascii=False)}\n\n"
                f"CONTEXTO (outfits):\n{ctx_txt}\n\n"
                f"PREGUNTA: {consulta_enr}\n\n"
                "Responde en español de forma clara y concreta: recomienda los 2 outfits del contexto que mejor "
                "encajen citando su ID y el nombre del look (ej. C_b102 - Nombre del look), menciona las prendas "
                "principales que incluyen y explica brevemente por qué combinan con lo pedido. "
                "Usa la válvula de escape SOLO si preguntan por precios, envíos, tallas numéricas o devoluciones."
            )
            respuesta = self._generar_base(SYSTEM_PROMPT_M3, prompt_final, max_tokens=220)
            valvula_act = VALVULA.lower() in respuesta.lower()

            if not valvula_act and finalistas:
                citados = [oid for oid, _ in finalistas if oid in respuesta]
                if not citados:
                    citados = [oid for oid, _ in finalistas[:2]]
                fichas = "\n\n".join(self.docs_by_id[oid] for oid in citados if oid in self.docs_by_id)
                if fichas:
                    respuesta = f"{respuesta}\n\n--- Detalle de los outfits recomendados ---\n{fichas}"

            lat = time.time() - t0

            salida = {
                "respuesta": respuesta,
                "contextos": contextos,
                "outfits": [i for i, _ in finalistas],
                "latencia_s": lat,
                "uso_vision": imagen is not None,
                "uso_herramienta": bool(args),
                "valvula_activada": valvula_act,
                "perfil_m1": perfil_m1,
                "info_visual": info_visual,
                "filtro": args,
                "cache_hit": False,
            }

            if self.usar_cache:
                self._cache[clave] = dict(salida)
                if len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)

            if registrar_log:
                registrar_evento(
                    consulta=pregunta,
                    tiene_imagen=imagen is not None,
                    es_moda=True,
                    uso_herramienta=bool(args),
                    valvula_activada=valvula_act,
                    outfits_recuperados=salida["outfits"],
                    latencia_s=lat,
                    estado="ok",
                )
            return salida

        except Exception as exc:
            lat = time.time() - t0
            if registrar_log:
                registrar_evento(
                    consulta=pregunta,
                    tiene_imagen=imagen is not None,
                    es_moda=False,
                    uso_herramienta=False,
                    valvula_activada=False,
                    outfits_recuperados=[],
                    latencia_s=lat,
                    estado="error",
                    detalle_error=type(exc).__name__,
                )
            raise