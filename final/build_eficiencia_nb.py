# -*- coding: utf-8 -*-
"""Genera final/eficiencia.ipynb: mide la técnica de eficiencia (GGUF int4) ANTES/DESPUÉS
con un diseño correcto (mismo hardware CPU, mismo juez, memoria homogénea, veredicto con latencia).
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): CELLS.append({"cell_type": "code", "metadata": {}, "source": t,
                           "execution_count": None, "outputs": []})

def _split(t):
    t = t.rstrip("\n")
    if t == "": return [""]
    ls = t.split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]

def build():
    for c in CELLS: c["source"] = _split(c["source"])
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eficiencia.ipynb")
    with open(out, "w", encoding="utf-8") as f: json.dump(nb, f, ensure_ascii=False, indent=1)
    with open(out, encoding="utf-8") as f: json.load(f)
    print("OK:", out, len(CELLS), "celdas")


md(r'''# Módulo 5 — Técnica de eficiencia: cuantización GGUF int4 del generador

**Componente elegido:** el **generador Qwen2.5-1.5B**, el que más tarda y más memoria ocupa del
pipeline (con HyDE son dos generaciones por consulta). Embeddings y BM25 ya son ligeros.

**Técnica:** cuantización a **GGUF int4 (Q4_K_M)** con `llama.cpp`, que corre en **CPU** — necesario
para el Space gratuito (sin GPU). **Alternativa descartada:** bitsandbytes int4/int8, que requiere
GPU y no sirve en el Space gratuito.

### Diseño del experimento (corrige la revisión: MISMA MÁQUINA)
- **Experimento A (aísla la cuantización, es el que vale):** generador **fp32 en CPU** (Transformers)
  vs **GGUF int4 en CPU** (llama.cpp), **mismo proceso, misma máquina**, embeddings en CPU en ambos.
  Esto representa el hardware del Space.
- **Experimento B (referencia, opcional):** fp16 en GPU, reportado aparte y etiquetado como tal.

### Presupuesto de calidad/latencia (decidido ANTES de medir, con números verificables)
> *Partimos de que el generador fp32 en CPU tarda del orden de decenas de segundos por consulta.
> **Aceptamos** perder hasta **1.0 punto** del juez (D2) y hasta **0.10** en D1/D3 **si** la latencia
> mediana (p50) en CPU **baja al menos 3×** respecto a fp32-CPU y el **tamaño en disco** del modelo
> baja a **< 1.3 GB** (desde ~3.1 GB en fp16).*

Al final decimos, criterio por criterio, si se cumplió. **Si no se cumplió, también es un resultado**
y documentamos la decisión (p. ej. desplegar sin HyDE, o un modelo 0.5B).''')

code(r'''# --- Celda 1: dependencias (Colab). Usa la WHEEL PRECOMPILADA de llama.cpp para CPU
# (evita compilar 10 min y da binario optimizado). Descomenta la primera vez. ---
# !pip -q install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# !pip -q install huggingface_hub chromadb sentence-transformers rank-bm25 transformers \
#     accelerate pandas numpy psutil google-genai
# !git clone -b cesar https://github.com/CesarMontoyag1/Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya.git
# %cd Isabela_Mendoza_Juan_Manuel_Lopera_Cesar_Montoya
import os, sys, time, json, platform
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath('final'))
assert os.path.isdir('final'), 'Ejecuta el %cd al repo (celda de git clone).'
CORPUS_DIR = 'notebooks/M3'
EVAL = 'notebooks/M3/eval/eval_set_rag.json'
# Reportar el hardware exacto (requisito del enunciado: misma máquina, decir cuál).
try:
    import psutil
    print('CPU:', os.cpu_count(), 'vCPU |', platform.processor() or platform.machine())
    print('RAM total: %.1f GB' % (psutil.virtual_memory().total / 1e9))
except Exception:
    print('CPU:', os.cpu_count(), 'vCPU')
print('final/ en path OK')''')

md(r'''## 1. Juez FIJO para todas las filas

Para que D2 compare de verdad (crítica de M3: el juez era el mismo generador), usamos **un único
juez** para TODAS las versiones: Gemini si hay `GEMINI_API_KEY`, o un juez local fp32 fijo como
alternativa (declarada como limitación). El mismo juez juzga "antes" y "después".''')

code(r'''# Pega tu clave de Gemini para el juez (NO la subas al repo). Opcional.
# os.environ['GEMINI_API_KEY'] = 'tu-key'
from harness import harness, cargar_eval_set, hacer_juez_gemini, hacer_juez_mismo_generador
EVAL_SET = cargar_eval_set(EVAL)
print('Casos en el eval set:', len(EVAL_SET))

JUEZ = None
try:
    JUEZ = hacer_juez_gemini()          # verifica que el modelo responde; lanza si falla
except Exception as e:
    print('Juez Gemini no disponible:', e)
USAR_GEMINI = JUEZ is not None
print('Juez:', 'Gemini (externo, fijo)' if USAR_GEMINI else 'local fp32 (se fija más abajo)')''')

md(r'''## 2. Descargar el GGUF int4 y medir su tamaño en disco''')

code(r'''from huggingface_hub import hf_hub_download
GGUF_REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
GGUF_FILE = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
ruta_gguf = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
tam_gguf_gb = os.path.getsize(ruta_gguf) / 1e9
print('GGUF int4: %.3f GB' % tam_gguf_gb)
# Tamaño de referencia del modelo fp16 (safetensors) para la columna de tamaño:
TAM_FP16_GB = 3.09  # Qwen2.5-1.5B en fp16 (declarado; no lo descargamos entero si no hace falta)''')

md(r'''## 3. Medir ANTES (fp32 en CPU) — misma máquina que el despliegue

Forzamos el generador en **CPU** para aislar el efecto de la cuantización (no GPU vs CPU).

**Nota de diseño:** fp32 en CPU es MUY lento (cada generación ~1-2 min; con HyDE son 2 por consulta).
Medir los 18 casos tardaría ~1 h. La **latencia por consulta no depende del número de casos**, así
que medimos el "antes" sobre un **subconjunto de N_ANTES casos** (suficiente para una mediana
estable) y lo decimos. La versión eficiente (GGUF, rápida) sí se mide sobre todo el eval set.''')

code(r'''import psutil, gc
proc = psutil.Process(os.getpid())
def rss_mb(): return proc.memory_info().rss / 1e6

from rag_core import PersonalShopper

# ANTES: fp32 en CPU. Subconjunto pequeño para latencia (fp32-CPU es muy lento).
N_ANTES = 6  # casos para medir el "antes"; la latencia por consulta no depende de N.
EVAL_ANTES = EVAL_SET[:N_ANTES]
print('Midiendo "antes" (fp32 CPU) sobre', len(EVAL_ANTES), 'casos. SIN HyDE para que sea viable.')

shopper_antes = PersonalShopper(corpus_dir=CORPUS_DIR, load_in='fp16', device='cpu')
ids_validos = set(shopper_antes.doc_ids)
if not USAR_GEMINI:
    JUEZ = hacer_juez_mismo_generador(shopper_antes)

rss0 = rss_mb()
# Medimos el "antes" SIN HyDE (comparable con la versión desplegable, que también va sin HyDE).
sc_antes, det_antes = harness(EVAL_ANTES, lambda q: shopper_antes.sistema(q, modo='avanzado', usar_hyde=False),
                              JUEZ, ids_validos, nombre='Antes (fp32 CPU, sin HyDE)')
sc_antes['rss_pico_mb'] = round(rss_mb(), 0)
sc_antes['tam_disco_gb'] = TAM_FP16_GB
sc_antes['n_casos'] = len(EVAL_ANTES)
print(sc_antes)''')

md(r'''## 4. Medir DESPUÉS (GGUF int4 en CPU) — con y sin HyDE

Medimos la versión eficiente en dos configuraciones: con HyDE (equivalente al "antes") y **sin HyDE**
(la que proponemos desplegar, porque HyDE duplica las generaciones y la ablación de M3 mostró que
aporta poco). Esto también cumple la **ablación obligatoria** del enunciado.''')

code(r'''del shopper_antes; gc.collect()

shopper_gguf = PersonalShopper(corpus_dir=CORPUS_DIR, load_in='gguf',
                               gguf_repo=GGUF_REPO, gguf_file=GGUF_FILE)
# Juez: el MISMO que en 'antes' (si es Gemini, ya está fijo; si es local, re-apuntar al mismo backend
# no es posible tras borrar fp32, así que con juez local se usa el GGUF y SE DECLARA la limitación).
if not USAR_GEMINI:
    JUEZ_D = hacer_juez_mismo_generador(shopper_gguf)
    print('AVISO: sin Gemini, el juez de "después" es el GGUF (distinto del de "antes"). '
          'D2 entre filas NO es comparable; se reporta como limitación. Usa Gemini para comparar.')
else:
    JUEZ_D = JUEZ

# GGUF con HyDE: sobre el mismo subconjunto que el "antes" (comparable, y HyDE es costoso).
sc_hyde, det_hyde = harness(EVAL_ANTES, lambda q: shopper_gguf.sistema(q, modo='avanzado', usar_hyde=True),
                            JUEZ_D, ids_validos, nombre='Después (GGUF int4 CPU, con HyDE) [n=6]')
sc_hyde['tam_disco_gb'] = round(tam_gguf_gb, 3); sc_hyde['n_casos'] = len(EVAL_ANTES)

# GGUF SIN HyDE: la versión DESPLEGABLE. Se mide sobre TODO el eval set (es rápida).
rss0 = rss_mb()
sc_sin, det_sin = harness(EVAL_SET, lambda q: shopper_gguf.sistema(q, modo='avanzado', usar_hyde=False),
                          JUEZ_D, ids_validos, nombre='Después (GGUF int4 CPU, SIN HyDE) [n=18]')
sc_sin['rss_pico_mb'] = round(rss_mb(), 0); sc_sin['tam_disco_gb'] = round(tam_gguf_gb, 3)
sc_sin['n_casos'] = len(EVAL_SET)
print(sc_hyde); print(sc_sin)''')

md(r'''## 5. Tabla antes / después (misma máquina, mismo juez, memoria homogénea)''')

code(r'''cols = ['sistema','n_casos','D1_citacion_outfits','D2_juez_medio','D3_cumplimiento',
        'latencia_p50_s','latencia_p95_s','tam_disco_gb','rss_pico_mb']
for sc in (sc_antes, sc_hyde, sc_sin):
    sc.setdefault('rss_pico_mb', None)
tabla = pd.DataFrame([sc_antes, sc_hyde, sc_sin])[cols]
os.makedirs('final/resultados', exist_ok=True)
tabla.to_csv('final/resultados/eficiencia_antes_despues.csv', index=False, encoding='utf-8')
print('Guardado: final/resultados/eficiencia_antes_despues.csv')
tabla''')

md(r'''## 6. Diagnóstico de latencia por etapa

Desglosamos dónde se va el tiempo (embeddings/recuperación vs generación). Confirma que la
generación del LLM es el cuello de botella y cuánto pesa HyDE (segunda generación).''')

code(r'''def latencias_etapa(shopper, hyde):
    acc = {}
    for e in EVAL_SET[:6]:
        s = shopper.sistema(e['input'], modo='avanzado', usar_hyde=hyde)
        for k, v in (s.get('latencias') or {}).items():
            acc.setdefault(k, []).append(v)
    return {k: round(float(np.median(v)), 2) for k, v in acc.items()}
print('GGUF con HyDE  (mediana por etapa, s):', latencias_etapa(shopper_gguf, True))
print('GGUF sin HyDE  (mediana por etapa, s):', latencias_etapa(shopper_gguf, False))''')

md(r'''## 7. Pruebas que NO miden exactitud (sobre la versión eficiente)

"Un modelo más chico puede acertar lo mismo y perder la capacidad de decir *no sé*". Repetimos la
**válvula** y una prueba de **invarianza** sobre el GGUF, y comparamos la válvula también sobre fp32
para saber si un fallo es culpa de la cuantización o ya venía de antes.''')

code(r'''# (a) Válvula sobre la versión eficiente. Con el pre-filtro determinista de rag_core, la válvula
# ya NO depende solo del LLM. Probamos varios casos fuera de catálogo.
casos_fuera = [e['input'] for e in EVAL_SET if e['tipo'] == 'fuera_de_corpus']
casos_fuera += ['¿tienen esta chaqueta en talla 40?', '¿cuál es el horario de la tienda?',
                '¿hacen descuentos con tarjeta?']
filas_valv = []
for q in casos_fuera:
    s = shopper_gguf.sistema(q, modo='avanzado')
    filas_valv.append({'consulta': q[:55], 'valvula': shopper_gguf.activo_valvula(s['respuesta']),
                       'por_prefiltro': s.get('valvula_pre', False)})
df_valv = pd.DataFrame(filas_valv)
df_valv.to_csv('final/resultados/valvula_eficiente.csv', index=False, encoding='utf-8')
print('Válvula activada en %d/%d casos fuera de catálogo' % (df_valv['valvula'].sum(), len(df_valv)))
df_valv''')

code(r'''# (b) Invarianza: >=5 grupos de parafraseos; umbral de aprobación definido ANTES = Jaccard >= 0.5
grupos = [
 ['abrigo largo de lana para un evento formal en invierno',
  'un abrigo de lana largo para una ocasión formal de invierno',
  'quiero un abrigo de lana largo y elegante para invierno formal'],
 ['outfit de hombre estilo techwear con cargo para el frío',
  'ropa masculina técnica de exteriores con pantalones cargo para clima frío',
  'un look de hombre outdoor con cargo, abrigado'],
 ['look minimalista de mujer para la oficina en tonos oscuros',
  'conjunto sobrio y minimalista para trabajar, colores oscuros, de mujer',
  'algo elegante y minimalista para ir a trabajar, oscuro'],
 ['vestido para una cita, femenino y elegante',
  'un look femenino y elegante para una cita',
  'algo bonito y elegante de mujer para salir en una cita'],
 ['suéter de rombos estilo universitario',
  'un jersey con patrón de rombos, estilo preppy',
  'prenda de punto con rombos, aire académico'],
]
UMBRAL = 0.5
filas_inv = []
for g in grupos:
    sets = [set(shopper_gguf.sistema(p, modo='avanzado')['outfits']) for p in g]
    inter = set.intersection(*sets); union = set.union(*sets)
    j = len(inter) / max(1, len(union))
    filas_inv.append({'consulta_base': g[0][:45], 'jaccard': round(j, 2), 'pasa': j >= UMBRAL})
df_inv = pd.DataFrame(filas_inv)
df_inv.to_csv('final/resultados/invarianza_eficiente.csv', index=False, encoding='utf-8')
print('Invarianza (umbral Jaccard >= %.1f): pasan %d/%d' % (UMBRAL, df_inv['pasa'].sum(), len(df_inv)))
df_inv''')

md(r'''## 8. Veredicto del presupuesto (criterio por criterio, incluyendo latencia y tamaño)''')

code(r'''d_d2 = sc_antes['D2_juez_medio'] - sc_sin['D2_juez_medio']   # usamos la versión SIN HyDE (la desplegable)
d_d1 = sc_antes['D1_citacion_outfits'] - sc_sin['D1_citacion_outfits']
d_d3 = sc_antes['D3_cumplimiento'] - sc_sin['D3_cumplimiento']
speedup = sc_antes['latencia_p50_s'] / max(0.001, sc_sin['latencia_p50_s'])

crit_calidad = (d_d2 <= 1.0) and (d_d1 <= 0.10) and (d_d3 <= 0.10)
crit_latencia = speedup >= 3.0
crit_tam = sc_sin['tam_disco_gb'] < 1.3

print('=== VEREDICTO POR CRITERIO (versión desplegable: GGUF int4 sin HyDE) ===')
print('Calidad  : perdida D2=%.2f (<=1.0), D1=%.3f (<=0.10), D3=%.3f (<=0.10) -> %s'
      % (d_d2, d_d1, d_d3, 'OK' if crit_calidad else 'NO'))
print('Latencia : p50 %.1fs -> %.1fs  (speedup %.1fx, objetivo >=3x) -> %s'
      % (sc_antes['latencia_p50_s'], sc_sin['latencia_p50_s'], speedup, 'OK' if crit_latencia else 'NO'))
print('Tamaño   : %.2f GB -> %.2f GB (objetivo <1.3 GB) -> %s'
      % (sc_antes['tam_disco_gb'], sc_sin['tam_disco_gb'], 'OK' if crit_tam else 'NO'))
global_ok = crit_calidad and crit_latencia and crit_tam
print()
print('VEREDICTO GLOBAL:', 'SE CUMPLIO el presupuesto' if global_ok else
      'NO se cumplio en todos los criterios -> documentar la decision en reporte.md')
if not USAR_GEMINI:
    print('\\nLIMITACION: sin Gemini, D2 de "antes" y "despues" se midio con jueces distintos; '
          'no es comparable. Correr con GEMINI_API_KEY para un D2 valido entre filas.')''')

md(r'''## 9. Tabla lista para el reporte (markdown)''')

code(r'''def md_table(df):
    enc = '| ' + ' | '.join(df.columns) + ' |'
    sep = '| ' + ' | '.join(['---']*len(df.columns)) + ' |'
    filas = ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in df.values]
    return '\n'.join([enc, sep] + filas)
print(md_table(tabla))''')


if __name__ == "__main__":
    build()
