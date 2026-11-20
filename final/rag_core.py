# -*- coding: utf-8 -*-
"""
rag_core.py — Núcleo del sistema RAG del personal shopper (Módulos 1-3 integrados).

Empaqueta, fuera del notebook, todo el pipeline del Módulo 3 ya corregido:
  - Corpus (FashionStylist, subsets Female+Male, traducido al español).
  - Índice: embeddings e5 + Chroma (en memoria) y BM25 (léxico).
  - Técnicas avanzadas elegidas por diagnóstico: búsqueda híbrida (BM25+dense, RRF) + HyDE.
    (El reranking cross-encoder se evaluó y se DESCARTÓ: la ablación mostró que bajaba D3.)
  - Capa agéntica: herramienta `filtrar_catalogo` (router determinista por regex + respaldo LLM).
  - Generador local Qwen2.5-1.5B-Instruct con prompt base + válvula de escape.

Expone la clase `PersonalShopper` con un método único:
    sistema(pregunta, modo='avanzado'|'ingenuo'|'final'|'baseline') -> dict

Correcciones aplicadas tras la retroalimentación del docente (Módulo 3):
  - BUG del router corregido: 'para el' era substring de 'para ella' (mujer->hombre). Ahora regex \\b.
  - `sistema_final` = `sistema_avanzado` restringido a un subconjunto de IDs (mismo pipeline + filtro),
    para que el delta sea atribuible SOLO a la herramienta.
  - Licencia real del corpus documentada (CC BY-NC 4.0, FashionStylist).

El generador puede cargarse en fp16 o cuantizado (int8/int4) vía `load_in` (Módulo 5).
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import collections

import numpy as np


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

SEED = 42

EMBED_MODEL = "intfloat/multilingual-e5-small"
GEN_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K = 5
POOL  = 20

# Prompt base COMPARTIDO por todas las versiones (delta limpio).
SYSTEM_PROMPT = (
    "Eres un personal shopper de un e-commerce de moda. Respondes en español, breve y concreto.\n"
    "REGLA 1 (recomendar): si el CONTEXTO contiene outfits de ropa, SIEMPRE recomienda el o los "
    "que mejor encajen con la petición y cítalos por su identificador (por ejemplo, B_b12). "
    "El contexto son outfits de ropa; úsalos aunque no sean una coincidencia perfecta.\n"
    "REGLA 2 (válvula de escape): usa la frase \"No encuentro esa información en el catálogo\" "
    "ÚNICAMENTE cuando la pregunta NO es sobre recomendar ropa, sino sobre precios, envíos, "
    "tallas numéricas (38, 40...) o políticas de la tienda (devoluciones). En esos casos no hay "
    "outfit que recomendar.\n"
    "NUNCA uses la válvula si la pregunta pide ropa y el contexto trae outfits. No inventes prendas "
    "que no estén en el contexto."
)
VALVULA = "No encuentro esa información en el catálogo"


# Router determinista (herramienta). Regex con \b para evitar el bug 'para el'/'para ella'.
SINONIMOS = {
    "gender": {
        # Nota: NO usamos 'para él' como señal: en español "para el día/trabajo/frío" usa el
        # artículo "el" y daría falsos positivos (bug señalado por el docente). "para ella" sí es
        # inequívoco (no existe el artículo "ella"), por eso solo esa queda en 'De mujer'.
        "De hombre": [r"\bhombres?\b", r"\bmasculino\b", r"\bcaballero\b"],
        "De mujer":  [r"\bmujeres?\b", r"\bfemenin[ao]\b", r"\bdama\b", r"\bpara ella\b",
                      r"\bchica\b", r"\bvestido\b"],
    },
    "season": {
        "Invierno":  [r"\binvierno\b", r"\bfr[ií]o\b", r"\babrigad[ao]\b", r"\bnieve\b"],
        "Verano":    [r"\bverano\b", r"\bcalor\b", r"\bcaluroso\b", r"\bplaya\b"],
        "Otoño":     [r"\boto[ñn]o\b"],
        "Primavera": [r"\bprimavera\b"],
    },
    "occasion": {
        "Trabajo":          [r"\btrabajo\b", r"\boficina\b", r"\bejecutiv", r"\blaboral\b",
                             r"\breuni[óo]n\b"],
        "Eventos sociales": [r"\bevento\b", r"\bsocial\b", r"\bformal\b", r"\bboda\b", r"\bgala\b",
                             r"\bfiesta\b", r"\bceremonia\b", r"\bcita\b"],
        "Viaje":            [r"\bviaj", r"\bturismo\b", r"\bvacacion"],
        "Escuela":          [r"\bescuela\b", r"\bcolegio\b", r"\buniversidad\b", r"\bclase\b",
                             r"\bestudi"],
        "Hogar":            [r"\bcasa\b", r"\bhogar\b"],
        "Deporte":          [r"\bdeporte\b", r"\bgimnasio\b", r"\bgym\b", r"\bentrenar\b",
                             r"\bcorrer\b", r"\brunning\b"],
    },
}
VALID = {k: set(v.keys()) for k, v in SINONIMOS.items()}


def _norm(v):
    """Normaliza categóricos: 'Business' quedó sin traducir -> 'Trabajo'."""
    mapa = {"Business": "Trabajo"}
    return "/".join(mapa.get(p.strip(), p.strip()) for p in str(v).split("/"))


# Pre-filtro determinista de la VÁLVULA DE ESCAPE. El catálogo solo describe estilos, prendas,
# estaciones y ocasiones; NO tiene precios, envíos, tallas numéricas, stock ni políticas. Si la
# consulta pide eso, activamos la válvula ANTES de llamar al generador (un modelo de 1.5B cuantizado
# no es fiable escribiendo la frase exacta). Esto corrige el fallo 2/2 de la válvula en GGUF.
PATRONES_FUERA_CATALOGO = [
    r"\bprecio", r"\bcu[aá]nto\s+(cuesta|vale|sale)", r"\bvale\b", r"\bcost[oó]\b",
    r"\benv[ií]o", r"\bdespacho", r"\bentrega\b", r"\bdomicilio", r"\benv[ií]an\b",
    r"\bdevoluci[oó]n", r"\bdevolver", r"\bgarant[ií]a", r"\bcambio\b", r"\breembolso",
    r"\btalla\s*\d", r"\btallas?\s+(num[eé]ric|\d)", r"\bn[uú]mero de talla",
    r"\bstock\b", r"\bdisponib", r"\binventario", r"\bhorario", r"\bhora de atenci",
    r"\bm[eé]todo de pago", r"\btarjeta\b", r"\bfactura", r"\bdescuento", r"\bcup[oó]n",
]
_RE_FUERA = re.compile("|".join(PATRONES_FUERA_CATALOGO), re.IGNORECASE)


def es_fuera_de_catalogo(pregunta):
    """True si la consulta pide algo que el catálogo no contiene (precio/envío/talla/etc.)."""
    return bool(_RE_FUERA.search(pregunta or ""))


# ===========================================================================
# SISTEMA
# ===========================================================================

class PersonalShopper:
    """Sistema RAG del personal shopper. Carga corpus, índice y generador una vez."""

    def __init__(self, corpus_dir, device=None, load_in="fp16", gen_model=GEN_MODEL,
                 gguf_repo=None, gguf_file=None, verbose=True):
        """load_in: 'fp16' | 'fp32' | 'int8' | 'int4' (bitsandbytes, requiere GPU) | 'gguf' (llama.cpp, CPU).
        En CPU, 'fp16' y 'fp32' cargan fp32 (fp16 en CPU es inviable).
        Para GGUF: pasar gguf_repo (repo HF con el .gguf) y gguf_file (nombre del archivo), o una
        ruta local en gguf_file. GGUF es la técnica de eficiencia del Módulo 5 para correr rápido
        en CPU (Hugging Face Spaces gratuito)."""
        import random
        random.seed(SEED); np.random.seed(SEED)
        self.verbose = verbose
        self.corpus_dir = corpus_dir
        self.load_in = load_in
        self.gen_model_name = gen_model
        self.gguf_repo = gguf_repo
        self.gguf_file = gguf_file
        self._device = device
        self._backend = "gguf" if load_in == "gguf" else "hf"
        # HyDE duplica las generaciones (dispara la latencia). Se puede desactivar vía env var
        # RAG_USAR_HYDE=0 (recomendado en el Space CPU) o con el parámetro en sistema().
        self.usar_hyde = os.environ.get("RAG_USAR_HYDE", "1") not in ("0", "false", "False")
        self._log("Inicializando PersonalShopper...")
        self.corpus_dir = self._resolver_corpus_dir()
        self._log("corpus_dir resuelto: " + os.path.abspath(self.corpus_dir))
        self._cargar_corpus()
        self._cargar_embeddings()
        self._cargar_bm25()
        self._indexar_chroma()
        self._cargar_generador()
        self._log("Sistema listo.")

    # ---------- utilidades ----------
    def _log(self, *a):
        if self.verbose:
            print("[rag_core]", *a)

    # ---------- carga de datos ----------
    def _resolver_corpus_dir(self):
        """Encuentra la carpeta que contiene corpus/corpus_look_es.csv, probando varias rutas.
        Hace el sistema robusto tanto en el repo (notebooks/M3) como en el Space (raíz)."""
        candidatos = [
            self.corpus_dir,
            os.path.join(self.corpus_dir, "..", "..", "notebooks", "M3"),
            "notebooks/M3", ".", "corpus", "..",
            os.path.dirname(os.path.abspath(__file__)),  # junto a este .py (caso Space)
        ]
        for base in candidatos:
            if base and os.path.exists(os.path.join(base, "corpus", "corpus_look_es.csv")):
                return base
        # búsqueda recursiva como último recurso
        import glob
        hits = glob.glob("**/corpus/corpus_look_es.csv", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
        raise FileNotFoundError(
            "No encuentro corpus/corpus_look_es.csv. Revisa corpus_dir=" + str(self.corpus_dir))

    def _leer_csv(self, nombre):
        ruta = os.path.join(self.corpus_dir, nombre)
        with open(ruta, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _cargar_corpus(self):
        looks = self._leer_csv(os.path.join("corpus", "corpus_look_es.csv"))
        labels = self._leer_csv(os.path.join("corpus", "corpus_label_es.csv"))
        self.labels_by_id = {r["itemID"]: r for r in labels}
        self.looks = looks

        self.documentos, self.metadatos, self.doc_ids = [], [], []
        for o in looks:
            items = [i.strip() for i in o["items"].split(",") if i.strip()]
            self.documentos.append(self._construir_documento(o, items))
            self.doc_ids.append(o["outfitID"])
            self.metadatos.append({
                "outfitID": o["outfitID"],
                "season":   _norm(o["season"]),
                "occasion": _norm(o["occasion"]),
                "gender":   self._genero_outfit(items),
            })
        self.id2idx = {i: k for k, i in enumerate(self.doc_ids)}
        self._log(f"Corpus: {len(self.documentos)} outfits, {len(labels)} prendas.")

    def _genero_outfit(self, items):
        gens = [self.labels_by_id[i]["gender"] for i in items if i in self.labels_by_id]
        for g in ("De mujer", "De hombre"):
            if g in gens:
                return g
        return "Unisex"

    def _construir_documento(self, outfit, items):
        partes = ["Outfit " + outfit["outfitID"] + ": " + outfit["look"]]
        partes.append("Estación: " + _norm(outfit["season"]) +
                      ". Ocasión: " + _norm(outfit["occasion"]) + ".")
        detalles = []
        for i in items:
            p = self.labels_by_id.get(i)
            if not p:
                continue
            d = ("- " + p["title"] + " (" + p["category"] + "): " + p["color"] + ", " +
                 p["materials"] + ", " + p["outline"])
            if p.get("detail"):
                d += ". " + p["detail"]
            detalles.append(d)
        if detalles:
            partes.append("Prendas:")
            partes.extend(detalles)
        return "\n".join(partes)

    def expandir_outfit(self, outfit_id):
        """Devuelve las prendas reales (desde los datos) de un outfit. Para la UI/M4."""
        fila = next((o for o in self.looks if o["outfitID"] == outfit_id), None)
        if not fila:
            return None
        items = [i.strip() for i in fila["items"].split(",") if i.strip()]
        prendas = []
        for i in items:
            p = self.labels_by_id.get(i)
            if p:
                prendas.append({"itemID": i, "categoria": p["category"], "titulo": p["title"],
                                "color": p["color"], "genero": p["gender"]})
        return {"outfitID": outfit_id, "look": fila["look"],
                "season": _norm(fila["season"]), "occasion": _norm(fila["occasion"]),
                "prendas": prendas}

    # ---------- modelos ----------
    def _cargar_embeddings(self):
        from sentence_transformers import SentenceTransformer
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self._device)
        self._log(f"Embeddings {EMBED_MODEL} en {self._device}.")

    def _embed(self, textos, tipo):
        pref = "query: " if tipo == "query" else "passage: "
        return self.embedder.encode([pref + t for t in textos], normalize_embeddings=True)

    def _cargar_bm25(self):
        from rank_bm25 import BM25Okapi
        self._tok = lambda t: re.findall(r"\w+", t.lower())
        self.bm25 = BM25Okapi([self._tok(d) for d in self.documentos])

    def _indexar_chroma(self):
        import chromadb
        self.client = chromadb.EphemeralClient()
        try:
            self.client.delete_collection("outfits")
        except Exception:
            pass
        self.col = self.client.create_collection("outfits", metadata={"hnsw:space": "cosine"})
        B = 128
        for i in range(0, len(self.documentos), B):
            sl = slice(i, i + B)
            embs = self._embed(self.documentos[sl], "passage").tolist()
            self.col.add(ids=self.doc_ids[sl], documents=self.documentos[sl],
                         embeddings=embs, metadatas=self.metadatos[sl])
        self._log(f"Índice Chroma: {self.col.count()} chunks.")

    def _cargar_generador(self):
        if self._backend == "gguf":
            self._cargar_generador_gguf()
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.gen_tok = AutoTokenizer.from_pretrained(self.gen_model_name)
        kwargs = {}
        if self.load_in in ("int8", "int4") and self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                if self.load_in == "int8":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                self._log(f"Generador cuantizado {self.load_in} (bitsandbytes).")
            except Exception as e:
                self._log(f"No se pudo cuantizar ({e}); usando fp16/fp32.")
                kwargs = {}
        if "quantization_config" not in kwargs:
            if self._device == "cuda":
                # GPU: fp16 (o el dtype nativo del modelo) es lo rápido.
                kwargs["torch_dtype"] = torch.float16 if self.load_in == "fp16" else "auto"
                kwargs["device_map"] = "cuda"
            else:
                # CPU: fp16/bf16 NO tienen kernels rápidos y la generación puede tardar una
                # eternidad (el notebook de eficiencia parecía "colgado"). En CPU siempre fp32.
                if self.load_in == "fp16":
                    self._log("Aviso: fp16 en CPU es muy lento; se carga en fp32.")
                kwargs["torch_dtype"] = torch.float32
                try:
                    torch.set_num_threads(os.cpu_count() or 4)
                except Exception:
                    pass
        self.gen_model = AutoModelForCausalLM.from_pretrained(self.gen_model_name, **kwargs)
        self.gen_model.eval()
        self._log(f"Generador {self.gen_model_name} cargado (load_in={self.load_in}).")

    def _cargar_generador_gguf(self):
        """Carga el generador cuantizado en GGUF con llama-cpp-python (rápido en CPU).
        Técnica de eficiencia del Módulo 5. Descarga el .gguf desde HF si se da gguf_repo."""
        from llama_cpp import Llama
        ruta = self.gguf_file
        if self.gguf_repo:
            from huggingface_hub import hf_hub_download
            ruta = hf_hub_download(repo_id=self.gguf_repo, filename=self.gguf_file)
        # n_ctx=2048 basta (el prompt trae top-5 outfits); seed fijo para reproducibilidad.
        self.llm = Llama(model_path=ruta, n_ctx=2048, n_threads=os.cpu_count() or 4,
                         n_batch=512, seed=SEED, verbose=False, chat_format="qwen")
        self.gen_tok = None
        self._log("Generador GGUF cargado: " + str(ruta))

    def generar(self, system_prompt, user_prompt, max_new_tokens=200):
        if self._backend == "gguf":
            out = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                max_tokens=max_new_tokens, temperature=0.0)
            return out["choices"][0]["message"]["content"].strip()
        import torch
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
        texto = self.gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.gen_tok(texto, return_tensors="pt").to(self.gen_model.device)
        with torch.no_grad():
            out = self.gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.gen_tok.eos_token_id)
        return self.gen_tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()

    # ---------- recuperación ----------
    def _buscar_denso(self, pregunta, k=TOP_K, ids_permitidos=None):
        # Si hay subconjunto permitido, lo pasamos como filtro 'where' de Chroma con $in sobre el
        # outfitID. Así Chroma devuelve k resultados YA filtrados (no pedimos k*4 y recortamos).
        where = None
        if ids_permitidos is not None:
            ids = list(ids_permitidos)
            if not ids:
                return []
            where = {"outfitID": {"$in": ids}}
        res = self.col.query(query_embeddings=self._embed([pregunta], "query").tolist(),
                             n_results=k, where=where, include=["documents", "metadatas"])
        return list(zip([m["outfitID"] for m in res["metadatas"][0]], res["documents"][0]))

    def _buscar_bm25(self, pregunta, k=POOL, ids_permitidos=None):
        scores = self.bm25.get_scores(self._tok(pregunta))
        idx = range(len(self.documentos)) if ids_permitidos is None else \
            [self.id2idx[i] for i in ids_permitidos if i in self.id2idx]
        ranked = sorted(idx, key=lambda i: scores[i], reverse=True)[:k]
        return [(self.doc_ids[i], self.documentos[i]) for i in ranked]

    @staticmethod
    def _rrf(listas, k_rrf=60, top=10):
        puntajes = collections.defaultdict(float); docs = {}
        for lista in listas:
            for rank, (i, d) in enumerate(lista):
                puntajes[i] += 1.0 / (k_rrf + rank); docs[i] = d
        return [(i, docs[i]) for i in sorted(puntajes, key=puntajes.get, reverse=True)[:top]]

    def _hyde(self, pregunta):
        sys = ("Eres un redactor de catálogo de moda. Dada una petición de un cliente, redacta en "
               "2-3 frases un outfit hipotético que la satisfaría (estilo, prendas, colores, "
               "estación, ocasión). Solo la descripción, en español.")
        return self.generar(sys, "Petición: " + pregunta + "\nDescripción del outfit ideal:",
                            max_new_tokens=60)

    def _recuperar_avanzado(self, pregunta, ids_permitidos=None, usar_hibrido=True,
                            usar_hyde=True):
        """Recuperación avanzada: híbrida (BM25+dense, RRF) + HyDE (opcional). Puede restringirse
        a un subconjunto de IDs (lo usa el modo 'final' con la herramienta)."""
        listas = []
        if usar_hibrido:
            listas.append(self._buscar_denso(pregunta, k=POOL, ids_permitidos=ids_permitidos))
            listas.append(self._buscar_bm25(pregunta, k=POOL, ids_permitidos=ids_permitidos))
        if usar_hyde:
            # HyDE: genera un documento hipotético (1 generación extra = el mayor costo de latencia).
            hip = self._hyde(pregunta)
            listas.append(self._buscar_denso(hip, k=POOL, ids_permitidos=ids_permitidos))
            listas.append(self._buscar_bm25(hip, k=POOL, ids_permitidos=ids_permitidos))
        if not listas:
            return self._buscar_denso(pregunta, k=TOP_K, ids_permitidos=ids_permitidos)
        return self._rrf(listas, top=TOP_K)

    # ---------- herramienta ----------
    def router_determinista(self, pregunta):
        q = pregunta.lower(); args = {}
        for campo, mapa in SINONIMOS.items():
            for canonico, patrones in mapa.items():
                if any(re.search(p, q) for p in patrones):
                    args[campo] = canonico
                    break
        return args

    def _ids_filtrados(self, args):
        if not args:
            return None
        def ok(m):
            return all(v in m.get(c, "") for c, v in args.items())
        ids = [self.doc_ids[k] for k, m in enumerate(self.metadatos) if ok(m)]
        if not ids:
            self._log("Aviso: el filtro " + str(args) + " no dejó outfits; se ignora (sin filtrar).")
            return None
        return ids

    # ---------- prompt ----------
    def _armar_prompt(self, pregunta, contextos):
        ctx = "\n\n".join(contextos) if contextos else "(sin resultados)"
        guia = ("El contexto SÍ trae outfits: recomiéndale al cliente el o los que mejor encajen y "
                "cita sus identificadores." if contextos else "No hay outfits en el contexto.")
        return ("CONTEXTO (outfits del catálogo):\n" + ctx +
                "\n\nPREGUNTA DEL CLIENTE: " + pregunta + "\n\n" + guia +
                " Explica en 1-2 frases por qué encaja. Solo usa la válvula de escape si la "
                "pregunta es de precio, envío, talla numérica o políticas de la tienda.")

    # ---------- API pública ----------
    def sistema(self, pregunta, modo="avanzado", usar_hyde=None):
        """Punto de entrada único. modo:
          'baseline' = sin RAG (generador solo);
          'ingenuo'  = top-k denso;
          'avanzado' = híbrida (+ HyDE según self.usar_hyde);
          'final'    = avanzado + herramienta de filtro de metadatos.
        Devuelve dict: respuesta, contextos, outfits, latencia_s, latencias (por etapa),
        uso_herramienta, filtro, valvula_pre (si la válvula se activó por el pre-filtro).
        """
        t0 = time.perf_counter()
        uso_herramienta, filtro = False, {}
        etapas = {}
        hyde = self.usar_hyde if usar_hyde is None else usar_hyde

        # --- Válvula determinista ANTES del generador (precio/envío/talla/etc.) ---
        if modo != "baseline" and es_fuera_de_catalogo(pregunta):
            return {"respuesta": VALVULA + " (la consulta pide datos de precio, envío, talla "
                    "numérica, stock o políticas, que no están en el catálogo de outfits).",
                    "contextos": [], "outfits": [], "latencia_s": time.perf_counter() - t0,
                    "latencias": {"valvula_pre": time.perf_counter() - t0},
                    "uso_herramienta": False, "filtro": {}, "valvula_pre": True}

        if modo == "baseline":
            prompt = ("PREGUNTA DEL CLIENTE: " + pregunta +
                      "\n\nResponde como personal shopper. (No tienes un catálogo disponible.)")
            resp = self.generar(SYSTEM_PROMPT, prompt)
            return {"respuesta": resp, "contextos": [], "outfits": [],
                    "latencia_s": time.perf_counter() - t0, "latencias": {}, "uso_herramienta": False,
                    "filtro": {}, "valvula_pre": False}

        tr = time.perf_counter()
        if modo == "ingenuo":
            recup = self._buscar_denso(pregunta, k=TOP_K)
        elif modo == "final":
            filtro = self.router_determinista(pregunta)
            uso_herramienta = bool(filtro)
            ids = self._ids_filtrados(filtro)
            recup = self._recuperar_avanzado(pregunta, ids_permitidos=ids, usar_hyde=hyde)
        else:  # avanzado
            recup = self._recuperar_avanzado(pregunta, usar_hyde=hyde)
        etapas["recuperacion_s"] = round(time.perf_counter() - tr, 3)

        contextos = [d for _, d in recup]
        outfits = [i for i, _ in recup]
        tg = time.perf_counter()
        resp = self.generar(SYSTEM_PROMPT, self._armar_prompt(pregunta, contextos))
        etapas["generacion_s"] = round(time.perf_counter() - tg, 3)
        return {"respuesta": resp, "contextos": contextos, "outfits": outfits,
                "latencia_s": time.perf_counter() - t0, "latencias": etapas,
                "uso_herramienta": uso_herramienta, "filtro": filtro, "valvula_pre": False}

    def activo_valvula(self, respuesta):
        return VALVULA.lower() in (respuesta or "").lower()

    def health_check(self):
        """Comprueba de verdad que el sistema está vivo: índice responde, generador cargado.
        Devuelve status 'ok' o 'degraded' con el detalle de cada componente. Para el despliegue."""
        checks = {}
        # 1) índice responde a una consulta de 1 resultado
        try:
            r = self.col.query(query_embeddings=self._embed(["abrigo"], "query").tolist(),
                               n_results=1, include=["metadatas"])
            checks["indice"] = "ok" if r["metadatas"][0] else "vacio"
        except Exception as e:
            checks["indice"] = "error: " + str(e)[:80]
        # 2) generador cargado
        if self._backend == "gguf":
            checks["generador"] = "ok" if getattr(self, "llm", None) is not None else "no cargado"
        else:
            checks["generador"] = "ok" if getattr(self, "gen_model", None) is not None else "no cargado"
        # 3) corpus
        checks["corpus"] = "ok" if len(self.documentos) > 0 else "vacio"

        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {
            "status": status,
            "checks": checks,
            "corpus_outfits": len(self.documentos),
            "indice_chunks": self.col.count(),
            "generador_modelo": self.gen_model_name,
            "load_in": self.load_in,
            "usar_hyde": self.usar_hyde,
            "device": self._device,
            "embeddings": EMBED_MODEL,
        }
