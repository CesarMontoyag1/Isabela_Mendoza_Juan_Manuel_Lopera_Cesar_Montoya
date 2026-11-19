# -*- coding: utf-8 -*-
"""
harness.py — La vara de evaluación (Módulo 2 re-instanciada para QA, Módulo 3).

Mismo harness para TODAS las filas de las tablas del proyecto final (tabla evolutiva y tabla
antes/después de la técnica de eficiencia). 3 dimensiones:

  - D1 (clásica): precisión de CITACIÓN DE OUTFITS REALES — fracción de outfits citados en la
    respuesta que existen y fueron recuperados (fuera de corpus: 1 si NO cita, que es lo correcto).
  - D2 (juez): LLM 1-5 de calidad de recomendación.
  - D3 (dominio): cumple el criterion (juez >= 4) + regla dura de válvula fuera de corpus.

Además mide latencia p50 y p95 (requisito del Módulo 5) sobre el MISMO eval set.

El juez se pasa como función `juez_fn(system_prompt, user_prompt) -> texto`. Por defecto usa el
mismo generador del sistema, PERO se puede (y se recomienda) pasar un juez DISTINTO (p. ej. Gemini)
para que la evaluación discrimine entre sistemas — esto atiende la crítica del Módulo 3 de que el
juez era el mismo que generaba. Ver `hacer_juez_gemini`.
"""

from __future__ import annotations

import json
import re
import numpy as np


VALVULA = "No encuentro esa información en el catálogo"
_ID_RE = re.compile(r"[BC]_b\d+")

RUBRICA = (
    "Eres un juez experto evaluando a un personal shopper de moda basado en un catálogo. Te doy la "
    "PREGUNTA, la RESPUESTA ESPERADA y la RESPUESTA DEL SISTEMA. Califica la RESPUESTA DEL SISTEMA "
    "de 1 a 5:\n"
    "5 = recomienda outfits del catálogo coherentes, los cita y no inventa; o activa correctamente "
    "la válvula de escape cuando la pregunta está fuera del catálogo.\n"
    "4 = buena con un detalle flojo.\n3 = parcial (recupera algo pero se desvía o no cita).\n"
    "2 = pobre o poco fundamentada.\n"
    "1 = inventa prendas/datos, o NO activa la válvula cuando debía.\n"
    "Responde SOLO un JSON: {\"puntaje\": <1-5>, \"motivo\": \"...\"}."
)


def _parse_json(texto):
    if not texto:
        return {}
    t = texto.strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j < i:
        return {}
    try:
        o = json.loads(t[i:j + 1])
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def activo_valvula(resp):
    return VALVULA.lower() in (resp or "").lower()


def d1_citacion(eval_set, respuestas, outfits_recuperados, ids_validos):
    scores = []
    for e, r, recup in zip(eval_set, respuestas, outfits_recuperados):
        citados = set(_ID_RE.findall(r or ""))
        reales = {c for c in citados if c in ids_validos}
        if e["tipo"] == "fuera_de_corpus":
            scores.append(1.0 if not citados else 0.0)
        else:
            if not citados:
                scores.append(0.0)
            else:
                rs = set(recup or [])
                scores.append(sum(1 for c in reales if c in rs) / len(citados))
    return scores


def d2_juez(eval_set, respuestas, juez_fn):
    puntajes = []
    for e, r in zip(eval_set, respuestas):
        up = ("PREGUNTA: " + e["input"] + "\n\nRESPUESTA ESPERADA: " + e["expected_response"] +
              "\n\nRESPUESTA DEL SISTEMA: " + (r or "") + "\n\nJSON:")
        obj = _parse_json(juez_fn(RUBRICA, up))
        try:
            p = int(obj.get("puntaje"))
        except Exception:
            p = 3
        puntajes.append(min(5, max(1, p)))
    return puntajes


def d3_cumplimiento(eval_set, respuestas, d2, umbral=4):
    out = []
    for e, r, p in zip(eval_set, respuestas, d2):
        if e["tipo"] == "fuera_de_corpus":
            out.append(1 if activo_valvula(r) else 0)
        else:
            out.append(1 if (p >= umbral and not activo_valvula(r)) else 0)
    return out


def harness(eval_set, sistema_fn, juez_fn, ids_validos, nombre="sistema"):
    """Corre las 3 dimensiones + latencias p50/p95 sobre el eval set.
    sistema_fn(pregunta) -> dict con respuesta/outfits/latencia_s.
    juez_fn(system, user) -> texto (idealmente un modelo DISTINTO al generador).
    """
    respuestas, outfits, lat = [], [], []
    for e in eval_set:
        s = sistema_fn(e["input"])
        respuestas.append(s.get("respuesta", ""))
        outfits.append(s.get("outfits", []))
        lat.append(s.get("latencia_s", 0.0))

    d1 = d1_citacion(eval_set, respuestas, outfits, ids_validos)
    d2 = d2_juez(eval_set, respuestas, juez_fn)
    d3 = d3_cumplimiento(eval_set, respuestas, d2)

    sc = {
        "sistema": nombre,
        "D1_citacion_outfits": round(float(np.mean(d1)), 3),
        "D2_juez_medio": round(float(np.mean(d2)), 3),
        "D3_cumplimiento": round(float(np.mean(d3)), 3),
        "latencia_p50_s": round(float(np.percentile(lat, 50)), 3),
        "latencia_p95_s": round(float(np.percentile(lat, 95)), 3),
    }
    det = {"respuestas": respuestas, "outfits": outfits, "d1": d1, "d2": d2, "d3": d3, "lat": lat}
    return sc, det


def cargar_eval_set(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# --- Jueces ---

def hacer_juez_mismo_generador(shopper):
    """Juez = el mismo generador del sistema (lo que criticó el docente en M3). Para comparar."""
    return lambda sp, up: shopper.generar(sp, up, max_new_tokens=160)


class _FallosJuez:
    """Contador de fallos del juez, para que un error de API NO se convierta en un D2 falso."""
    def __init__(self):
        self.n = 0


# Modelos de juez a probar, en orden. Las cuentas nuevas solo tienen la familia 3.x; probamos
# varios y usamos el primero que responda (evita el 404 por nombre de modelo caducado).
MODELOS_JUEZ = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-lite-latest",
                "gemini-flash-latest", "gemini-2.5-flash"]


def hacer_juez_gemini(modelo=None, contador=None):
    """Juez = Gemini (modelo DISTINTO al generador) para que D2 discrimine entre sistemas.
    Requiere GEMINI_API_KEY. Devuelve None si no hay clave (entonces se usa el juez local).
    Prueba varios modelos y usa el PRIMERO que responde (robusto a nombres caducados).
    Lleva la cuenta de fallos en `contador` para reportarlos (no los enmascara como puntaje 3)."""
    import os
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types
    cli = genai.Client(api_key=key)

    candidatos = ([modelo] if modelo else []) + [m for m in MODELOS_JUEZ if m != modelo]
    modelo_ok, ultimo_error = None, None
    for cand in candidatos:
        try:
            cli.models.generate_content(
                model=cand, contents="responde: ok",
                config=types.GenerateContentConfig(temperature=0.0))
            modelo_ok = cand
            break
        except Exception as e:
            ultimo_error = e
            continue
    if modelo_ok is None:
        raise RuntimeError("Ningún modelo de juez Gemini respondió. Último error: " +
                           str(ultimo_error) + ". Revisa GEMINI_API_KEY.")
    modelo = modelo_ok
    print("[juez] Gemini activo con modelo:", modelo)

    cont = contador if contador is not None else _FallosJuez()

    def juez(sp, up):
        try:
            r = cli.models.generate_content(
                model=modelo, contents=up,
                config=types.GenerateContentConfig(system_instruction=sp, temperature=0.0))
            return r.text or ""
        except Exception as e:
            cont.n += 1
            print("[juez] fallo de Gemini (" + str(cont.n) + "): " + str(e)[:100])
            return ""  # se contará como puntaje 3, pero queda registrado el fallo
    juez.fallos = cont
    return juez
