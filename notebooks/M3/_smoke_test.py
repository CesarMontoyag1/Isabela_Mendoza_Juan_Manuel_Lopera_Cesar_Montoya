"""Prueba en seco de la lógica que NO depende de LLM/embeddings:
   carga de corpus, construcción de documentos, metadatos, BM25, RRF,
   parseo JSON tolerante y validación del filtro. No carga Qwen ni Chroma.
"""
import os, re, json, collections

M3 = os.path.dirname(os.path.abspath(__file__))
LOOK = os.path.join(M3, 'corpus_look_es.csv')
LABEL = os.path.join(M3, 'corpus_label_es.csv')

import csv
def _norm(v):
    mapa = {'Business': 'Trabajo'}
    return '/'.join(mapa.get(p.strip(), p.strip()) for p in str(v).split('/'))
def leer(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

looks = leer(LOOK); labels = leer(LABEL)
labels_by_id = {r['itemID']: r for r in labels}
print('Outfits:', len(looks), '| Prendas:', len(labels))

def genero_outfit(items):
    gens = [labels_by_id[i]['gender'] for i in items if i in labels_by_id]
    if not gens: return 'Unisex'
    for g in ('De mujer', 'De hombre'):
        if g in gens: return g
    return 'Unisex'

def construir_documento(o):
    items = [i.strip() for i in o['items'].split(',') if i.strip()]
    partes = ["Outfit " + o['outfitID'] + ": " + o['look']]
    partes.append("Estación: " + _norm(o['season']) + ". Ocasión: " + _norm(o['occasion']) + ".")
    det = []
    for i in items:
        p = labels_by_id.get(i)
        if not p: continue
        d = "- " + p['title'] + " (" + p['category'] + "): " + p['color'] + ", " + p['materials'] + ", " + p['outline']
        if p.get('detail'): d += ". " + p['detail']
        det.append(d)
    if det:
        partes.append('Prendas:'); partes.extend(det)
    return "\n".join(partes)

documentos, metadatos, doc_ids = [], [], []
faltan_items = 0
for o in looks:
    items = [i.strip() for i in o['items'].split(',') if i.strip()]
    faltan_items += sum(1 for i in items if i not in labels_by_id)
    documentos.append(construir_documento(o)); doc_ids.append(o['outfitID'])
    metadatos.append({'outfitID': o['outfitID'], 'season': _norm(o['season']),
                      'occasion': _norm(o['occasion']), 'gender': genero_outfit(items)})
print('Documentos:', len(documentos), '| items sin label (se ignoran):', faltan_items)
print('\n--- chunk ejemplo ---\n', documentos[0][:400])

# BM25
from rank_bm25 import BM25Okapi
def tok(t): return re.findall(r'\w+', t.lower())
bm25 = BM25Okapi([tok(d) for d in documentos])
def buscar_bm25(q, k=5, cand=None):
    sc = bm25.get_scores(tok(q))
    idx = range(len(documentos)) if cand is None else cand
    return [(doc_ids[i], documentos[i]) for i in sorted(idx, key=lambda i: sc[i], reverse=True)[:k]]

print('\n--- BM25: "techwear cargo hombre invierno" ---')
for i, d in buscar_bm25('techwear cargo hombre invierno', k=3):
    print(' ', i, '::', d[:90].replace(chr(10), ' '))

# RRF
def rrf(listas, k_rrf=60, top=10):
    pu = collections.defaultdict(float); docs = {}
    for lista in listas:
        for rank, (i, d) in enumerate(lista):
            pu[i] += 1.0 / (k_rrf + rank); docs[i] = d
    return [(i, docs[i]) for i in sorted(pu, key=pu.get, reverse=True)[:top]]
fus = rrf([buscar_bm25('abrigo lana evento formal', 10), buscar_bm25('invierno elegante', 10)], top=3)
print('\n--- RRF fusion ok, top ids:', [i for i, _ in fus])

# parse_json_tolerante
def parse_json_tolerante(t):
    if not t: return {}
    ini, fin = t.find('{'), t.rfind('}')
    if ini == -1 or fin == -1 or fin < ini: return {}
    try:
        o = json.loads(t[ini:fin+1]); return o if isinstance(o, dict) else {}
    except Exception: return {}
assert parse_json_tolerante('ruido {"gender": "De hombre"} fin') == {'gender': 'De hombre'}
assert parse_json_tolerante('sin json') == {}
assert parse_json_tolerante('{roto') == {}
print('\nparse_json_tolerante: OK')

# filtro por substring (maneja 'Otoño/Invierno')
VALID = {'gender': {'De mujer','De hombre','Unisex'},
         'season': {'Primavera','Verano','Otoño','Invierno'},
         'occasion': {'Escuela','Hogar','Eventos sociales','Viaje','Trabajo','Deporte'}}
args = {'gender': 'De hombre', 'season': 'Invierno'}
def ok(m): return all(v in m.get(c, '') for c, v in args.items())
n = sum(1 for m in metadatos if ok(m))
print('Filtro {hombre, invierno} -> outfits que pasan:', n)
print('\nSMOKE TEST OK')
