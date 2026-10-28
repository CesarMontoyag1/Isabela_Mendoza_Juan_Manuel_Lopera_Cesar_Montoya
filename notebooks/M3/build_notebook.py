# -*- coding: utf-8 -*-
"""
build_notebook.py

Genera 'rag_sistema.ipynb' (notebook de la entrega M3) de forma programática y
robusta. Cada celda se define como texto plano (delimitado con comillas simples
triples) y el script arma el JSON .ipynb válido y lo valida al final.

Uso:
    python notebooks/M3/build_notebook.py
"""

import json
import os

CELLS = []


def md(texto):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": texto})


def code(texto):
    CELLS.append({"cell_type": "code", "metadata": {}, "source": texto,
                  "execution_count": None, "outputs": []})


def _split_lines(texto):
    texto = texto.rstrip("\n")
    if texto == "":
        return [""]
    lineas = texto.split("\n")
    return [l + "\n" for l in lineas[:-1]] + [lineas[-1]]


def build():
    for c in CELLS:
        c["source"] = _split_lines(c["source"])
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_sistema.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    with open(out, encoding="utf-8") as f:
        json.load(f)
    # Validar que cada celda de código compile
    errs = 0
    for i, c in enumerate(CELLS):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            try:
                compile(src, "<cell %d>" % i, "exec")
            except SyntaxError as e:
                errs += 1
                print("SYNTAX ERROR celda %d: %s" % (i, e))
    print("OK: %s (%d celdas, %d errores de sintaxis)" % (out, len(CELLS), errs))


# ===========================================================================
# CONTENIDO
# ===========================================================================

md(r'''# SI4006 · Módulo 3 — Sistema RAG agéntico: *personal shopper* de moda

**Qué hace este sistema, en una frase:** recibe lo que una persona pide con sus propias palabras
("busco algo elegante y abrigado para una boda en invierno") y le **recomienda conjuntos de ropa
reales de un catálogo**, explicando por qué encajan; si la pregunta no se puede responder con el
catálogo (envíos, precios, tallas numéricas), **lo dice en vez de inventar**.

**Para quién:** el cliente de un e-commerce de moda que escribe por chat a un *personal shopper*.

**Qué es una buena respuesta:** una recomendación fundamentada en outfits que **existen en el
catálogo** (los cita), coherente con el estilo/ocasión/clima/género pedido, y honesta cuando el
dato no está.

---

Este notebook construye y evalúa **cuatro versiones** del sistema:

| # | Versión | Qué añade |
|---|---------|-----------|
| 1 | Línea base M2 (sin RAG) | El generador solo: no conoce el catálogo |
| 2 | `sistema_rag` (ingenuo) | top-k denso en Chroma + prompt aumentado + válvula de escape |
| 3 | `sistema_rag_avanzado` | **3 técnicas**: híbrida (BM25+dense, RRF) + reranking cross-encoder + HyDE |
| 4 | `sistema` (final, agéntico) | + **herramienta** de filtro de metadatos (género/estación/ocasión) |

Las mide con **el harness del Módulo 2** (3 dimensiones, re-instanciadas para QA) + **RAGAS**.

> **Reproducibilidad:** semillas y versiones fijas. Corre en **Colab gratuito sin claves de pago**:
> generador y juez son modelos locales (Qwen2.5). Gemini es opcional.''')

md(r'''## 0. Configuración y dependencias''')

code(r'''# --- Dependencias (Colab gratuito). Ejecuta esta celda UNA vez por sesión. ---
# En Colab, quita el '#' de la línea de abajo y ejecútala (tarda 1-2 min):
# !pip -q install chromadb sentence-transformers rank-bm25 transformers accelerate peft datasets scikit-learn pandas
#
# Nota: la sección 7 (RAGAS) usa una IMPLEMENTACIÓN PROPIA de las 4 métricas con el Qwen local,
# NO la librería 'ragas' (que da conflictos de dependencias irresolubles en Colab). No requiere
# instalar nada extra.
print('Dependencias: en Colab, descomenta la línea de pip y ejecútala una vez.')''')

code(r'''# --- Localización de los archivos del corpus ---
# OPCIÓN A (recomendada): clonar el repo en Colab y usar sus rutas.
#   En Colab, descomenta estas dos líneas y ejecútalas una vez:
#   !git clone -b cesar https://github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya.git
#   %cd Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya
# OPCIÓN B: montar Google Drive y copiar los CSV a notebooks/M3/.
#   from google.colab import drive; drive.mount('/content/drive')
#
# Esta celda detecta dónde está la carpeta M3 (con el corpus en M3/corpus/) y ajusta M3_DIR.
import os, glob
candidatos = ['notebooks/M3', '.', '/content', '/content/drive/MyDrive']
M3_DIR = None
for base in candidatos:
    if os.path.exists(os.path.join(base, 'corpus', 'corpus_look_es.csv')):
        M3_DIR = base; break
if M3_DIR is None:
    hits = glob.glob('/content/**/corpus/corpus_look_es.csv', recursive=True) + \
           glob.glob('**/corpus/corpus_look_es.csv', recursive=True)
    M3_DIR = os.path.dirname(os.path.dirname(hits[0])) if hits else 'notebooks/M3'
print('M3_DIR =', M3_DIR)
assert os.path.exists(os.path.join(M3_DIR, 'corpus', 'corpus_look_es.csv')), \
    'No encuentro corpus/corpus_look_es.csv. Verifica la estructura de notebooks/M3/.'
print('Estructura M3 presente:',
      os.path.isdir(os.path.join(M3_DIR, 'corpus')),
      os.path.isdir(os.path.join(M3_DIR, 'eval')),
      os.path.isdir(os.path.join(M3_DIR, 'resultados')))''')

code(r'''# --- Reproducibilidad: semillas y versiones fijas ---
import os, random, json, time, re, math, collections
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'

EMBED_MODEL  = 'intfloat/multilingual-e5-small'
RERANK_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
GEN_MODEL    = 'Qwen/Qwen2.5-1.5B-Instruct'
JUDGE_MODEL  = 'Qwen/Qwen2.5-1.5B-Instruct'

# M3_DIR lo fijó la celda de localización anterior. Si corres suelto, usa el default:
try:
    M3_DIR
except NameError:
    M3_DIR = 'notebooks/M3' if os.path.isdir('notebooks/M3') else '.'
# Estructura organizada: corpus/, eval/, resultados/ dentro de M3_DIR
LOOK_CSV   = os.path.join(M3_DIR, 'corpus', 'corpus_look_es.csv')
LABEL_CSV  = os.path.join(M3_DIR, 'corpus', 'corpus_label_es.csv')
EVAL_PATH  = os.path.join(M3_DIR, 'eval', 'eval_set_rag.json')
CHROMA_DIR = os.path.join(M3_DIR, 'chroma_db')
os.makedirs(os.path.join(M3_DIR, 'resultados'), exist_ok=True)
OUT_SCORECARD = os.path.join(M3_DIR, 'resultados', 'scorecard_m3.csv')
OUT_RAGAS     = os.path.join(M3_DIR, 'resultados', 'ragas_m3.csv')

print('DEVICE:', DEVICE)
for p in (LOOK_CSV, LABEL_CSV, EVAL_PATH):
    print(' ', p, '->', os.path.exists(p))''')

md(r'''## 1. Corpus, *chunking* e índice

**Fuente.** Catálogo de outfits de moda de e-commerce (colecciones de mujer y de hombre),
traducido al español con `data_generation/translate_corpus.py` (Gemini, verificado al 100%).
**Licencia:** datos de uso académico para SI4006; corpus derivado y traducido.

**Tamaño.** 800 outfits (`corpus_look_es.csv`) + 3796 prendas (`corpus_label_es.csv`). Indexamos
**un documento por outfit**, enriquecido con el detalle de sus prendas → **800 chunks**.

**Tres decisiones justificadas (sobre *nuestro* corpus):**

1. **Tamaño/solapamiento → 1 outfit = 1 chunk, sin solapamiento.** Cada `look` es una descripción
   corta y autocontenida de un conjunto (1–3 frases) + el detalle de sus prendas; partirlo rompería
   la unidad "outfit" que es justo lo que el usuario pide, así que no hay nada que solapar.
2. **Embeddings → `multilingual-e5-small`.** Las consultas son español coloquial y el corpus es
   español descriptivo; un embedding multilingüe pequeño alinea ambos registros y entra en Colab.
3. **top-k → 5.** El usuario quiere **opciones** de conjuntos; k=5 trae varios outfits comparables
   sin saturar el prompt del generador pequeño.''')

code(r'''# --- Carga del corpus y normalización de metadatos ---
import csv

def _norm(v):
    # 'Business' quedó sin traducir en el categórico; lo normalizamos a 'Trabajo'.
    mapa = {'Business': 'Trabajo'}
    return '/'.join(mapa.get(p.strip(), p.strip()) for p in str(v).split('/'))

def leer_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

looks  = leer_csv(LOOK_CSV)
labels = leer_csv(LABEL_CSV)
labels_by_id = {row['itemID']: row for row in labels}
print('Outfits:', len(looks), '| Prendas:', len(labels))
print('Estaciones:', dict(collections.Counter(_norm(x['season']) for x in looks)))
print('Ocasiones :', dict(collections.Counter(_norm(x['occasion']) for x in looks)))''')

code(r'''# --- Documentos: 1 outfit = 1 chunk, enriquecido con sus prendas ---
def genero_outfit(items):
    gens = [labels_by_id[i]['gender'] for i in items if i in labels_by_id]
    if not gens:
        return 'Unisex'
    for g in ('De mujer', 'De hombre'):
        if g in gens:
            return g
    return 'Unisex'

def construir_documento(outfit):
    items = [i.strip() for i in outfit['items'].split(',') if i.strip()]
    partes = ["Outfit " + outfit['outfitID'] + ": " + outfit['look']]
    partes.append("Estación: " + _norm(outfit['season']) + ". Ocasión: " + _norm(outfit['occasion']) + ".")
    detalles = []
    for i in items:
        p = labels_by_id.get(i)
        if not p:
            continue
        d = "- " + p['title'] + " (" + p['category'] + "): " + p['color'] + ", " + p['materials'] + ", " + p['outline']
        if p.get('detail'):
            d += ". " + p['detail']
        detalles.append(d)
    if detalles:
        partes.append('Prendas:')
        partes.extend(detalles)
    return "\n".join(partes)

documentos, metadatos, doc_ids = [], [], []
for o in looks:
    items = [i.strip() for i in o['items'].split(',') if i.strip()]
    documentos.append(construir_documento(o))
    doc_ids.append(o['outfitID'])
    metadatos.append({
        'outfitID': o['outfitID'],
        'season':   _norm(o['season']),
        'occasion': _norm(o['occasion']),
        'gender':   genero_outfit(items),
        'coleccion': o.get('coleccion', ''),
    })

print('Documentos (chunks):', len(documentos))
print("\n--- Ejemplo de chunk ---\n")
print(documentos[0][:600])''')

code(r'''# --- Indexación: embed -> store en Chroma ---
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

def embed_passages(textos):
    return embedder.encode(["passage: " + t for t in textos],
                           normalize_embeddings=True, show_progress_bar=True).tolist()

def embed_queries(textos):
    return embedder.encode(["query: " + t for t in textos],
                           normalize_embeddings=True).tolist()

client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection('outfits')
except Exception:
    pass
col = client.create_collection('outfits', metadata={'hnsw:space': 'cosine'})

B = 128
for i in range(0, len(documentos), B):
    sl = slice(i, i + B)
    col.add(ids=doc_ids[sl], documents=documentos[sl],
            embeddings=embed_passages(documentos[sl]), metadatas=metadatos[sl])
print('Indexados en Chroma:', col.count(), 'chunks')''')

md(r'''## 2. Generador local y RAG ingenuo (línea base del RAG)

El **generador** es `Qwen2.5-1.5B-Instruct` local (sin API de pago). Es el **mismo generador y el
mismo prompt base** en las cuatro versiones → *delta limpio*.

`sistema_rag(pregunta)` (sesión 7): recupera top-k denso, arma el prompt aumentado y genera, con
**válvula de escape**.

**Contrato de salida (igual en las 4 versiones):**
```
sistema(pregunta) -> {'respuesta': str, 'contextos': [str], 'outfits': [id], 'latencia_s': float}
```''')

code(r'''# --- Generador local Qwen2.5 ---
from transformers import AutoModelForCausalLM, AutoTokenizer

print('Cargando generador', GEN_MODEL, '...')
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL, torch_dtype='auto', device_map=DEVICE if DEVICE == 'cuda' else None)
gen_model.eval()

def generar(system_prompt, user_prompt, max_new_tokens=320):
    msgs = [{'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt}]
    texto = gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = gen_tok(texto, return_tensors='pt').to(gen_model.device)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=gen_tok.eos_token_id)
    return gen_tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print('Generador listo.')''')

code(r'''# --- Prompt base COMPARTIDO por las 4 versiones (delta limpio) ---
SYSTEM_PROMPT = (
    'Eres un personal shopper de un e-commerce de moda. Respondes en español, breve y concreto.\n'
    'REGLA 1 (recomendar): si el CONTEXTO contiene outfits de ropa, SIEMPRE recomienda el o los '
    'que mejor encajen con la petición y cítalos por su identificador (por ejemplo, B_b12). '
    'El contexto son outfits de ropa; úsalos aunque no sean una coincidencia perfecta.\n'
    'REGLA 2 (válvula de escape): usa la frase "No encuentro esa información en el catálogo" '
    'ÚNICAMENTE cuando la pregunta NO es sobre recomendar ropa, sino sobre precios, envíos, '
    'tallas numéricas (38, 40...) o políticas de la tienda (devoluciones). En esos casos no hay '
    'outfit que recomendar.\n'
    'NUNCA uses la válvula si la pregunta pide ropa y el contexto trae outfits. No inventes prendas '
    'que no estén en el contexto.'
)
VALVULA = 'No encuentro esa información en el catálogo'

def armar_prompt(pregunta, contextos):
    ctx = "\n\n".join(contextos) if contextos else '(sin resultados)'
    hay_ctx = bool(contextos)
    guia = ("El contexto SÍ trae outfits: recomiéndale al cliente el o los que mejor encajen y "
            "cita sus identificadores."
            if hay_ctx else
            "No hay outfits en el contexto.")
    return ("CONTEXTO (outfits del catálogo):\n" + ctx +
            "\n\nPREGUNTA DEL CLIENTE: " + pregunta +
            "\n\n" + guia +
            " Explica en 1-2 frases por qué encaja. Solo usa la válvula de escape si la pregunta "
            "es de precio, envío, talla numérica o políticas de la tienda.")''')

code(r'''# --- Recuperación densa (base del RAG ingenuo) ---
TOP_K = 5

def buscar_denso(pregunta, k=TOP_K, where=None):
    res = col.query(query_embeddings=embed_queries([pregunta]), n_results=k,
                    where=where, include=['documents', 'metadatas'])
    metas, docs = res['metadatas'][0], res['documents'][0]
    return list(zip([m['outfitID'] for m in metas], docs))

def sistema_rag(pregunta):
    """RAG ingenuo: top-k denso -> prompt aumentado -> genera."""
    t0 = time.time()
    recup = buscar_denso(pregunta, k=TOP_K)
    contextos = [d for _, d in recup]
    respuesta = generar(SYSTEM_PROMPT, armar_prompt(pregunta, contextos))
    return {'respuesta': respuesta, 'contextos': contextos,
            'outfits': [i for i, _ in recup], 'latencia_s': time.time() - t0}

_demo = sistema_rag('Busco un abrigo largo de lana para un evento formal en invierno')
print('Outfits:', _demo['outfits'])
print('Respuesta:', _demo['respuesta'][:400])''')

md(r'''## 3. RAG avanzado — 3 técnicas elegidas por diagnóstico

**Diagnóstico (consultas donde falla el RAG ingenuo).** Corrimos el ingenuo sobre el eval set y
anotamos *por qué* fallaba cada caso difícil:

| Falla en el ingenuo | Tipo de falla | Técnica que la ataca |
|---|---|---|
| Términos exactos ("cuello alto", "argyle", "borreguito", "cargo") no llegan: el denso trae el estilo pero pierde la **prenda literal** | recuperación (léxico) | **Híbrida BM25 + dense (RRF)** |
| El top-k denso trae el estilo correcto pero con el **atributo equivocado**; el bueno queda en posición 6–10 | recuperación (orden) | **Reranking cross-encoder** |
| Consultas cortas/vagas ("algo elegante para la oficina") no se parecen léxicamente al catálogo | recuperación (brecha consulta-doc) | **HyDE** |

Se combinan en `sistema_rag_avanzado(pregunta)` (misma firma). La ablación se mide en la sección 6.''')

code(r'''# --- Técnica A: BM25 (léxico) ---
from rank_bm25 import BM25Okapi

def tokenizar(t):
    return re.findall(r'\w+', t.lower())

bm25 = BM25Okapi([tokenizar(d) for d in documentos])

def buscar_bm25(pregunta, k=10, candidatos_idx=None):
    scores = bm25.get_scores(tokenizar(pregunta))
    idx = range(len(documentos)) if candidatos_idx is None else candidatos_idx
    ranked = sorted(idx, key=lambda i: scores[i], reverse=True)[:k]
    return [(doc_ids[i], documentos[i]) for i in ranked]

print('BM25 listo sobre', len(documentos), 'documentos')''')

code(r'''# --- Fusión RRF (denso + BM25) ---
def rrf(listas_ranking, k_rrf=60, top=10):
    """Cada lista es [(id, doc), ...] en orden. Fusiona por Reciprocal Rank Fusion."""
    puntajes = collections.defaultdict(float)
    docs = {}
    for lista in listas_ranking:
        for rank, (i, d) in enumerate(lista):
            puntajes[i] += 1.0 / (k_rrf + rank)
            docs[i] = d
    orden = sorted(puntajes, key=puntajes.get, reverse=True)[:top]
    return [(i, docs[i]) for i in orden]

def buscar_hibrido(pregunta, k=TOP_K, pool=20):
    densos = buscar_denso(pregunta, k=pool)
    lexicos = buscar_bm25(pregunta, k=pool)
    return rrf([densos, lexicos], top=max(k, pool))[:k]''')

code(r'''# --- Técnica B: reranking con cross-encoder ---
from sentence_transformers import CrossEncoder
reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)

def rerank(pregunta, candidatos, k=TOP_K):
    if not candidatos:
        return []
    scores = reranker.predict([(pregunta, d) for _, d in candidatos])
    orden = sorted(range(len(candidatos)), key=lambda i: scores[i], reverse=True)[:k]
    return [candidatos[i] for i in orden]''')

code(r'''# --- Técnica C: HyDE ---
HYDE_SYS = ('Eres un redactor de catálogo de moda. Dada una petición de un cliente, redacta en '
            '2-3 frases un outfit hipotético que la satisfaría (estilo, prendas, colores, estación, '
            'ocasión). Solo la descripción, en español.')

def hyde_documento(pregunta):
    return generar(HYDE_SYS, "Petición: " + pregunta + "\nDescripción del outfit ideal:",
                   max_new_tokens=120)

def buscar_hyde(pregunta, pool=20):
    hip = hyde_documento(pregunta)
    densos = buscar_denso(hip, k=pool)
    lexicos = buscar_bm25(hip, k=pool)
    return rrf([densos, lexicos], top=pool)''')

code(r'''# --- sistema_rag_avanzado: combina las 3 técnicas (misma firma) ---
def sistema_rag_avanzado(pregunta, usar_hibrido=True, usar_rerank=False, usar_hyde=True):
    """RAG avanzado. Técnicas finales (por diagnóstico): híbrida + HyDE.
    usar_rerank=False por defecto porque la ablación mostró que el reranking BAJA D3.
    Los flags permiten reproducir la ablación (p. ej. usar_rerank=True para ver el efecto)."""
    t0 = time.time()
    candidatos, pool = [], 20
    if usar_hyde:
        candidatos.extend(buscar_hyde(pregunta, pool=pool))
    if usar_hibrido:
        candidatos.extend(buscar_hibrido(pregunta, k=pool, pool=pool))
    if not candidatos:
        candidatos = buscar_denso(pregunta, k=pool)

    vistos, dedup = set(), []
    for i, d in candidatos:
        if i not in vistos:
            vistos.add(i); dedup.append((i, d))

    finalistas = rerank(pregunta, dedup, k=TOP_K) if usar_rerank else dedup[:TOP_K]
    contextos = [d for _, d in finalistas]
    respuesta = generar(SYSTEM_PROMPT, armar_prompt(pregunta, contextos))
    return {'respuesta': respuesta, 'contextos': contextos,
            'outfits': [i for i, _ in finalistas], 'latencia_s': time.time() - t0}

_demo = sistema_rag_avanzado('algo elegante y minimalista para la oficina en invierno, tonos oscuros')
print('Outfits:', _demo['outfits'])
print('Respuesta:', _demo['respuesta'][:400])''')

md(r'''## 4. Capa agéntica — una herramienta: filtro de metadatos

**Herramienta `filtrar_catalogo`.** Restringe por metadatos (`gender`, `season`, `occasion`)
*antes* de la recuperación semántica. Útil cuando la consulta trae una restricción dura
("solo **de mujer**", "para **invierno**", "para el **trabajo**").

- **Esquema** (nombre, descripción, parámetros) declarado para el modelo.
- **Parseo tolerante a errores** del JSON del modelo (si viene mal, se ignora el filtro).
- **Tope de pasos**: máximo 1 llamada a la herramienta por consulta (sin bucles).

**Cuándo se usa y cuándo no:** se usa si hay restricción de género/estación/ocasión; no en consultas
puramente estilísticas. Mostramos 2 casos donde se llama y 1 donde no.''')

code(r'''# --- Esquema de la herramienta + router ---
TOOL_SCHEMA = {
    'name': 'filtrar_catalogo',
    'description': 'Filtra el catálogo por metadatos duros antes de buscar. Úsala SOLO si la '
                   'consulta especifica género, estación u ocasión concretos.',
    'parameters': {
        'gender':   ['De mujer', 'De hombre', 'Unisex'],
        'season':   ['Primavera', 'Verano', 'Otoño', 'Invierno'],
        'occasion': ['Escuela', 'Hogar', 'Eventos sociales', 'Viaje', 'Trabajo', 'Deporte'],
    },
}
VALID = {k: set(v) for k, v in TOOL_SCHEMA['parameters'].items()}

ROUTER_SYS = (
    'Decide si conviene filtrar el catálogo de moda por metadatos antes de buscar. Responde SOLO '
    'un JSON. Si la consulta menciona género (hombre/mujer), estación (invierno/verano/otoño/'
    'primavera) u ocasión (trabajo/viaje/evento/escuela/hogar/deporte), devuelve esos campos; si '
    'no, devuelve {}. Formato: {"gender": ..., "season": ..., "occasion": ...} con solo lo aplicable.'
)

def parse_json_tolerante(texto):
    """Extrae el primer objeto JSON del texto; tolera ruido. {} si falla."""
    if not texto:
        return {}
    t = texto.strip()
    ini, fin = t.find('{'), t.rfind('}')
    if ini == -1 or fin == -1 or fin < ini:
        return {}
    try:
        obj = json.loads(t[ini:fin + 1])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

# Diccionario de sinónimos -> valor canónico del esquema. El router mapea las
# palabras de la consulta del cliente (lenguaje natural) a los valores exactos
# que Chroma espera. Esto hace el router determinista y reproducible.
SINONIMOS = {
    'gender': {
        'De hombre': ['hombre', 'masculino', 'caballero', 'hombres', 'para él', 'para el'],
        'De mujer':  ['mujer', 'femenino', 'dama', 'mujeres', 'para ella', 'chica', 'vestido'],
    },
    'season': {
        'Invierno':  ['invierno', 'frío', 'frio', 'abrigado', 'nieve'],
        'Verano':    ['verano', 'calor', 'caluroso', 'playa', 'fresco'],
        'Otoño':     ['otoño', 'otonio'],
        'Primavera': ['primavera'],
    },
    'occasion': {
        'Trabajo':         ['trabajo', 'oficina', 'ejecutiv', 'laboral', 'reunión', 'reunion'],
        'Eventos sociales':['evento', 'social', 'formal', 'boda', 'gala', 'fiesta', 'ceremonia'],
        'Viaje':           ['viaje', 'viajar', 'turismo', 'vacacion'],
        'Escuela':         ['escuela', 'colegio', 'universidad', 'clase', 'estudio'],
        'Hogar':           ['casa', 'hogar', 'estar en casa', 'cómodo en casa'],
        'Deporte':         ['deporte', 'gimnasio', 'gym', 'entrenar', 'correr', 'running'],
    },
}

def router_determinista(pregunta):
    """Detecta género/estación/ocasión por palabras clave. Reproducible, sin LLM."""
    q = pregunta.lower()
    args = {}
    for campo, mapa in SINONIMOS.items():
        for canonico, claves in mapa.items():
            if any(k in q for k in claves):
                args[campo] = canonico
                break  # primer match por campo
    return args

def decidir_filtro(pregunta, usar_llm_respaldo=True):
    """Router agéntico (la 'herramienta'). Máx 1 llamada de razonamiento.
    1) Mapeo determinista por palabras clave (principal, reproducible).
    2) Respaldo: si el determinista no detecta nada, el LLM propone un JSON
       (parseo tolerante + validación contra el enum). Es el paso agéntico.
    """
    args = router_determinista(pregunta)
    if args or not usar_llm_respaldo:
        return args
    # Respaldo LLM (tope: 1 sola llamada)
    propuesta = generar(ROUTER_SYS, "Consulta: " + pregunta + "\nJSON:", max_new_tokens=80)
    crudo = parse_json_tolerante(propuesta)
    # normalizar lo que proponga el LLM contra el enum (tolerante)
    for c, v in crudo.items():
        if c in VALID and isinstance(v, str):
            if v in VALID[c]:
                args[c] = v
            else:  # intentar mapear por sinónimos lo que dijo el LLM
                m = router_determinista(v)
                if c in m:
                    args[c] = m[c]
    return args''')

code(r'''# --- sistema (final, agéntico): herramienta + recuperación avanzada ---
def sistema(pregunta):
    """Decide filtro (herramienta) -> recupera avanzado sobre el subconjunto -> genera."""
    t0 = time.time()
    args = decidir_filtro(pregunta)          # paso agéntico (máx 1 llamada)
    uso_herramienta = bool(args)

    # post-filtro flexible por substring (maneja 'Otoño/Invierno')
    if args:
        def ok(m):
            return all(v in m.get(c, '') for c, v in args.items())
        idx_validos = [k for k, m in enumerate(metadatos) if ok(m)]
    else:
        idx_validos = list(range(len(documentos)))
    if not idx_validos:
        idx_validos = list(range(len(documentos)))

    # Técnicas finales elegidas POR DIAGNÓSTICO: híbrida (BM25+dense, RRF) + HyDE + herramienta.
    # El reranking cross-encoder se EVALUÓ y se DESCARTÓ: la ablación mostró que bajaba D3
    # (0.667 con rerank vs 0.833 sin rerank). Ver sección de ablación y reporte.md.
    pool = 20
    sub_ids = set(doc_ids[i] for i in idx_validos)
    densos = [(i, d) for i, d in buscar_denso(pregunta, k=pool * 3) if i in sub_ids][:pool]
    lexicos = buscar_bm25(pregunta, k=pool, candidatos_idx=idx_validos)
    # HyDE restringido al subconjunto filtrado
    hip = hyde_documento(pregunta)
    hyde_densos = [(i, d) for i, d in buscar_denso(hip, k=pool * 3) if i in sub_ids][:pool]
    fus = rrf([densos, lexicos, hyde_densos], top=TOP_K) if (densos or lexicos or hyde_densos) \
        else buscar_denso(pregunta, k=TOP_K)
    finalistas = fus[:TOP_K]
    contextos = [d for _, d in finalistas]
    respuesta = generar(SYSTEM_PROMPT, armar_prompt(pregunta, contextos))
    return {'respuesta': respuesta, 'contextos': contextos,
            'outfits': [i for i, _ in finalistas], 'latencia_s': time.time() - t0,
            'uso_herramienta': uso_herramienta, 'filtro': args}

for q in ['ropa de hombre para invierno estilo techwear',
          'un outfit de mujer para un evento social formal',
          'algo bohemio y colorido']:
    r = sistema(q)
    print(('[TOOL] ' if r['uso_herramienta'] else '[no-tool] ') + 'filtro=' + str(r['filtro']) + ' | ' + q)
    print('   outfits:', r['outfits'][:3])''')

md(r'''## 5. Evaluación — el harness del Módulo 2 (misma vara, re-instanciada para QA)

El harness del M2 medía **extracción de JSON**. Un RAG devuelve **texto**. Mantenemos **la misma
vara** (firma `harness(eval_set, sistema)`, 3 dimensiones, mismo eval set y juez) pero
re-instanciamos cada dimensión para QA (lo que el enunciado permite cuando la tarea del M1 era de
clasificación):

- **D1 (clásica):** precisión de **citación de outfits reales** — fracción de outfits citados en la
  respuesta que existen en el catálogo y fueron recuperados (en casos fuera de corpus, puntúa 1 si
  NO cita outfits, que es lo correcto). Reemplaza a la similitud por embeddings, que quedaba saturada
  y no discriminaba (ver lectura honesta).
- **D2 (juez):** LLM local (Qwen) con rúbrica 1–5 de calidad de recomendación.
- **D3 (dominio):** fracción de casos que cumplen el `criterion` (juez ≥ 4) **+ regla dura de
  válvula** en los casos fuera de corpus.

Eval set: `eval_set_rag.json` (12 casos: 10 en corpus + 2 fuera). El **mismo** para todas las filas.''')

code(r'''# --- Cargar eval set ---
with open(EVAL_PATH, encoding='utf-8') as f:
    EVAL_SET = json.load(f)
print('Casos:', len(EVAL_SET), '| fuera de corpus:',
      sum(1 for e in EVAL_SET if e['tipo'] == 'fuera_de_corpus'))

def juez_generar(sp, up, max_new_tokens=160):
    return generar(sp, up, max_new_tokens=max_new_tokens)''')

code(r'''# --- D1: métrica clásica = PRECISIÓN DE CITACIÓN DE OUTFITS REALES ---
# Diagnóstico (ver sección de lectura honesta): la similitud por embeddings contra la
# 'expected_response' quedaba saturada (~0.88 para todas las versiones, incluso la línea base
# que inventa), porque cualquier texto de personal shopper se parece a la referencia larga.
# No discriminaba bueno de malo. La re-instanciamos para medir lo que DEFINE una buena respuesta
# en este dominio: ¿la respuesta cita outfits que EXISTEN y fueron RECUPERADOS del catálogo?
#   - En casos EN corpus: fracción de outfits citados que son reales y recuperados (precisión).
#     Una respuesta que no cita ningún outfit real puntúa 0 (la línea base no puede citar).
#   - En casos FUERA de corpus: puntúa 1 si NO cita outfits (lo correcto es la válvula), 0 si cita.
_ID_RE = re.compile(r'[BC]_b\d+')
_IDS_VALIDOS = set(doc_ids)

def d1_citacion(eval_set, respuestas, outfits_recuperados):
    scores = []
    for e, r, recup in zip(eval_set, respuestas, outfits_recuperados):
        citados = set(_ID_RE.findall(r or ''))
        reales = {c for c in citados if c in _IDS_VALIDOS}
        if e['tipo'] == 'fuera_de_corpus':
            # correcto = no citar ningún outfit (debe activar la válvula)
            scores.append(1.0 if not citados else 0.0)
        else:
            if not citados:
                scores.append(0.0)                       # no citó nada -> no ancla al catálogo
            else:
                recup_set = set(recup or [])
                # precisión: de lo citado, cuánto es real Y estaba en lo recuperado
                bien = sum(1 for c in reales if c in recup_set)
                scores.append(bien / len(citados))
    return scores

# --- D2: juez LLM 1-5 ---
RUBRICA_RAG = (
    'Eres un juez experto evaluando a un personal shopper de moda basado en un catálogo. Te doy la '
    'PREGUNTA, la RESPUESTA ESPERADA y la RESPUESTA DEL SISTEMA. Califica la RESPUESTA DEL SISTEMA '
    'de 1 a 5:\n'
    '5 = recomienda outfits del catálogo coherentes, los cita y no inventa; o activa correctamente '
    'la válvula de escape cuando la pregunta está fuera del catálogo.\n'
    '4 = buena con un detalle flojo.\n3 = parcial (recupera algo pero se desvía o no cita).\n'
    '2 = pobre o poco fundamentada.\n'
    '1 = inventa prendas/datos, o NO activa la válvula cuando debía.\n'
    'Responde SOLO un JSON: {"puntaje": <1-5>, "motivo": "..."}.'
)

def d2_juez(eval_set, respuestas):
    puntajes = []
    for e, r in zip(eval_set, respuestas):
        up = ("PREGUNTA: " + e['input'] + "\n\nRESPUESTA ESPERADA: " + e['expected_response'] +
              "\n\nRESPUESTA DEL SISTEMA: " + r + "\n\nJSON:")
        obj = parse_json_tolerante(juez_generar(RUBRICA_RAG, up))
        try:
            p = int(obj.get('puntaje'))
        except Exception:
            p = 3
        puntajes.append(min(5, max(1, p)))
    return puntajes''')

code(r'''# --- D3: cumplimiento + regla dura de válvula ---
def activo_valvula(resp):
    return VALVULA.lower() in (resp or '').lower()

def d3_cumplimiento(eval_set, respuestas, d2, umbral=4):
    out = []
    for e, r, p in zip(eval_set, respuestas, d2):
        if e['tipo'] == 'fuera_de_corpus':
            out.append(1 if activo_valvula(r) else 0)
        else:
            out.append(1 if (p >= umbral and not activo_valvula(r)) else 0)
    return out''')

code(r'''# --- harness(eval_set, sistema): 3 dimensiones + latencia ---
def harness(eval_set, sistema_fn, nombre='sistema'):
    respuestas, contextos, outfits, lat = [], [], [], []
    for e in eval_set:
        s = sistema_fn(e['input'])
        if isinstance(s, dict):
            respuestas.append(s.get('respuesta', '')); contextos.append(s.get('contextos', []))
            outfits.append(s.get('outfits', [])); lat.append(s.get('latencia_s', 0.0))
        else:
            respuestas.append(str(s)); contextos.append([]); outfits.append([]); lat.append(0.0)
    d1 = d1_citacion(eval_set, respuestas, outfits)   # métrica clásica = citación de outfits reales
    d2 = d2_juez(eval_set, respuestas)
    d3 = d3_cumplimiento(eval_set, respuestas, d2)
    sc = {'sistema': nombre,
          'D1_citacion_outfits': round(float(np.mean(d1)), 3),
          'D2_juez_medio':       round(float(np.mean(d2)), 3),
          'D3_cumplimiento':     round(float(np.mean(d3)), 3),
          'latencia_mediana_s':  round(float(np.median(lat)), 3)}
    det = {'respuestas': respuestas, 'contextos': contextos, 'outfits': outfits,
           'd1': d1, 'd2': d2, 'd3': d3, 'lat': lat}
    return sc, det''')

code(r'''# --- Línea base M2 (sin RAG): el generador SIN catálogo ---
def sistema_baseline_m2(pregunta):
    """Sin RAG: el mismo generador responde sin catálogo. Fila de comparación."""
    t0 = time.time()
    prompt = ("PREGUNTA DEL CLIENTE: " + pregunta +
              "\n\nResponde como personal shopper. (No tienes un catálogo disponible.)")
    return {'respuesta': generar(SYSTEM_PROMPT, prompt), 'contextos': [], 'outfits': [],
            'latencia_s': time.time() - t0}''')

md(r'''## 6. La tabla evolutiva (mismo eval set, misma vara)

Las **4 versiones** sobre el **mismo** eval set, con el **mismo generador y prompt base** → *delta
limpio*.''')

code(r'''# --- Correr el harness sobre las 4 versiones ---
resultados = {}
for nombre, fn in [
    ('1. Linea base M2 (sin RAG)', sistema_baseline_m2),
    ('2. RAG ingenuo',            sistema_rag),
    ('3. RAG avanzado (3 tec)',   sistema_rag_avanzado),
    ('4. Sistema final (tool)',   sistema),
]:
    print('Evaluando:', nombre, '...')
    sc, det = harness(EVAL_SET, fn, nombre)
    resultados[nombre] = (sc, det)
    print('  ', sc)

tabla = pd.DataFrame([sc for sc, _ in resultados.values()])
tabla''')

code(r'''# --- Ablación: quitar una técnica a la vez (qué aporta cada una) ---
# Incluye el reranking para EVIDENCIAR que baja D3 (por eso quedó fuera del sistema final).
ablaciones = {
    'final (hibrida+HyDE)':      dict(usar_hibrido=True,  usar_rerank=False, usar_hyde=True),
    'solo dense':                dict(usar_hibrido=False, usar_rerank=False, usar_hyde=False),
    'hibrida sin HyDE':          dict(usar_hibrido=True,  usar_rerank=False, usar_hyde=False),
    'con reranking (descartado)':dict(usar_hibrido=True,  usar_rerank=True,  usar_hyde=True),
}
filas = []
for nombre, flags in ablaciones.items():
    fn = lambda q, f=flags: sistema_rag_avanzado(q, **f)
    sc, _ = harness(EVAL_SET, fn, nombre)
    filas.append(sc); print(nombre, '->', sc)
tabla_ablacion = pd.DataFrame(filas)
tabla_ablacion''')

md(r'''## 7. RAGAS — las 4 métricas (sistema final y ingenuo)

**faithfulness, answer relevancy, context precision, context recall**. Cada caso aporta pregunta,
respuesta, contexto recuperado y referencia. **Juez: Qwen2.5 local** (el mismo generador), con una
**implementación propia** de las métricas (ver nota abajo). Lectura: *context recall* bajo →
problema de **recuperación**; *faithfulness* bajo con buen contexto → problema de **generación**.''')

code(r'''# --- Datos para RAGAS ---
def datos_ragas(nombre):
    _, det = resultados[nombre]
    return {
        'question':     [e['input'] for e in EVAL_SET],
        'answer':       det['respuestas'],
        'contexts':     [c if c else ['(sin contexto)'] for c in det['contextos']],
        'ground_truth': [e['expected_response'] for e in EVAL_SET],
    }

USAR_GEMINI_JUEZ = bool(os.environ.get('GEMINI_API_KEY'))
print('Juez RAGAS:', 'Gemini' if USAR_GEMINI_JUEZ else 'LLM local Qwen (sin clave)')''')

md(r'''**Juez de RAGAS = Qwen2.5 local (sin clave de pago).** RAGAS por defecto usa OpenAI; aquí lo
reconfiguramos para que el **mismo Qwen2.5 ya cargado** haga de juez y el **mismo embedder e5** haga
los embeddings. Así RAGAS corre 100% local en Colab gratuito. (Para usar Gemini en su lugar, hay una
celda alternativa comentada al final.)''')

md(r'''**Implementación propia de RAGAS (con nuestro propio juez), sin la librería `ragas`.**

La librería `ragas` genera conflictos de dependencias irresolubles en Colab (langchain-core/
community vs numpy; `metaclass conflict`). El enunciado permite explícitamente *"la implementación
del laboratorio de la sesión 10 (con su propio juez)"*, así que calculamos las **4 métricas a mano**
usando el **Qwen2.5 local** (juez) y el **embedder e5 local** ya cargados. Cero dependencias nuevas,
100% reproducible y sin claves de pago.

- **faithfulness**: de las afirmaciones de la respuesta, ¿cuántas se sostienen en el contexto? (juez)
- **answer relevancy**: similitud (e5) entre la pregunta y la respuesta.
- **context precision**: de los chunks recuperados, ¿cuántos son relevantes a la pregunta? (juez)
- **context recall**: ¿el contexto cubre lo que dice la referencia? (juez)''')

code(r'''# --- Implementación propia de las 4 métricas RAGAS (juez = Qwen local, emb = e5 local) ---
def _si_no(pregunta_juez):
    """Pregunta de sí/no al juez local. Devuelve 1.0 (sí) / 0.0 (no)."""
    out = generar('Responde SOLO con "si" o "no".', pregunta_juez, max_new_tokens=5).strip().lower()
    return 1.0 if out.startswith('s') or 'yes' in out or 'si' in out[:4] else 0.0

def _afirmaciones(respuesta):
    """Parte la respuesta en afirmaciones simples (por oración)."""
    frases = re.split(r'(?<=[.!?])\s+', (respuesta or '').strip())
    return [f for f in frases if len(f.split()) >= 3][:6]  # máx 6 para acotar costo

def ragas_faithfulness(answer, contexts):
    afs = _afirmaciones(answer)
    if not afs:
        return np.nan
    ctx = '\n'.join(contexts)[:3000]
    oks = []
    for a in afs:
        q = ('CONTEXTO:\n' + ctx + '\n\nAFIRMACION: "' + a +
             '"\n\n¿La afirmación está respaldada por el contexto?')
        oks.append(_si_no(q))
    return float(np.mean(oks))

def ragas_answer_relevancy(question, answer):
    e = embedder.encode(['query: ' + question, 'passage: ' + (answer or '')],
                        normalize_embeddings=True)
    return float(np.dot(e[0], e[1]))

def ragas_context_precision(question, contexts):
    if not contexts:
        return np.nan
    rel = []
    for c in contexts[:5]:
        q = ('PREGUNTA: ' + question + '\n\nFRAGMENTO:\n' + c[:1200] +
             '\n\n¿El fragmento es útil/relevante para responder la pregunta?')
        rel.append(_si_no(q))
    # precision = relevantes en las primeras posiciones (promedio simple de utilidad)
    return float(np.mean(rel))

def ragas_context_recall(reference, contexts):
    if not contexts:
        return np.nan
    ctx = '\n'.join(contexts)[:3000]
    q = ('REFERENCIA (lo que debería cubrirse):\n' + reference[:1200] +
         '\n\nCONTEXTO RECUPERADO:\n' + ctx +
         '\n\n¿El contexto recuperado cubre la información de la referencia?')
    return _si_no(q)

print('Métricas RAGAS propias definidas (juez Qwen local + e5).')''')

code(r'''# --- Correr las 4 métricas sobre un sistema ---
def correr_ragas(nombre):
    d = datos_ragas(nombre)
    faith, arel, cprec, crec = [], [], [], []
    for q, a, ctx, gt in zip(d['question'], d['answer'], d['contexts'], d['ground_truth']):
        faith.append(ragas_faithfulness(a, ctx))
        arel.append(ragas_answer_relevancy(q, a))
        cprec.append(ragas_context_precision(q, ctx))
        crec.append(ragas_context_recall(gt, ctx))
    return {
        'sistema': nombre,
        'faithfulness':      round(float(np.nanmean(faith)), 3),
        'answer_relevancy':  round(float(np.nanmean(arel)), 3),
        'context_precision': round(float(np.nanmean(cprec)), 3),
        'context_recall':    round(float(np.nanmean(crec)), 3),
    }

ragas_ingenuo = correr_ragas('2. RAG ingenuo')
print('RAGAS ingenuo:', ragas_ingenuo)
ragas_final = correr_ragas('4. Sistema final (tool)')
print('RAGAS final  :', ragas_final)''')

md(r'''## 8. Guardar resultados y evidencia de la herramienta''')

code(r'''# --- Guardar scorecard y ablación ---
tabla.to_csv(OUT_SCORECARD, index=False, encoding='utf-8')
tabla_ablacion.to_csv(OUT_SCORECARD.replace('.csv', '_ablacion.csv'), index=False, encoding='utf-8')
print('Guardado:', OUT_SCORECARD)

# ragas_ingenuo / ragas_final ya son dicts con las 4 métricas (implementación propia).
ragas_df = pd.DataFrame([ragas_ingenuo, ragas_final])[
    ['sistema', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']]
ragas_df.to_csv(OUT_RAGAS, index=False, encoding='utf-8')
print('Guardado:', OUT_RAGAS)
ragas_df''')

code(r'''# --- Evidencia de la herramienta: cuándo se usa y cuándo no ---
casos_tool = [
    'ropa de hombre estilo techwear para el frío',       # usar (género+estación)
    'un vestido de mujer para un evento social formal',  # usar (género+ocasión)
    'algo bohemio y colorido para el día a día',         # NO usar (solo estilo)
    'quiero verme elegante',                             # NO usar (vago)
]
print('uso_herramienta | filtro | consulta')
for q in casos_tool:
    r = sistema(q)
    print(('SI ' if r['uso_herramienta'] else 'no ') + '| ' + str(r['filtro']) + ' | ' + q)''')

md(r'''## 9. Lectura honesta (resumen)

Ver `reporte.md` para el análisis completo. Responde: (1) qué movió qué (tabla evolutiva +
ablación), (2) cuánto costó (latencia mediana por versión), (3) qué falla queda (RAGAS: context
recall bajo = recuperación; faithfulness bajo con buen contexto = generación), (4) qué pasó con la
válvula en los 2 casos fuera de corpus.''')


if __name__ == "__main__":
    build()
