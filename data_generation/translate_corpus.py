"""
translate_corpus.py

Traduce el corpus de moda (outfits + prendas) del INGLÉS ORIGINAL al ESPAÑOL,
por bloques, garantizando cobertura del 100% de las celdas de texto.

Por qué se traduce desde el inglés original (look_en / label_en) y NO desde los
archivos *_es ya existentes:
  - Los *_es se generaron con reemplazo de subcadenas y arrastran errores
    ("tierojo", "layerojo", mezclas inglés/español). Traducir desde el inglés
    limpio evita heredar esa basura.

Estrategia para garantizar 100% de traducción:
  1. Se traducen SOLO las columnas de texto libre; las columnas categóricas
     (gender, season, occasion) se mapean con diccionarios fijos y deterministas.
  2. La traducción LLM se hace por BLOQUES (varias celdas por llamada), con la
     respuesta en JSON indexado para no desalinear filas.
  3. Cada bloque se VERIFICA: si una celda traducida aún contiene palabras en
     inglés (según un detector de residuos), esa celda se reintenta de forma
     individual hasta MAX_REINTENTOS_CELDA.
  4. Se usa una CACHÉ en disco (traducciones.json): cada texto único se traduce
     una sola vez. Reejecutar el script reutiliza la caché -> reproducible y barato.
  5. Al final se emite un REPORTE de cobertura; si queda cualquier celda con
     residuos de inglés, el script termina con código de error para que el fallo
     sea visible y no pase silencioso.

Los IDs (outfitID, itemID, items) NUNCA se traducen ni se alteran.
Se añade un prefijo de colección a los IDs (B_ para mujer, C_ para hombre) para
que las dos colecciones puedan unirse después sin colisiones de ID.

Requiere:
    pip install google-genai

Configurar la API key (es la misma usada en generate_synthetic_dataset.py):

    Windows PowerShell:
        $env:GEMINI_API_KEY="tu-api-key"
    Linux / Colab:
        export GEMINI_API_KEY="tu-api-key"

Uso:
    python data_generation/translate_corpus.py

Alternativa sin API de pago:
    Ver la función traducir_bloque_local() al final (opus-mt / NLLB). Está
    desactivada por defecto; se documenta cómo activarla.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

SEED = 42  # Gemini no usa semilla explícita; se fija temperature=0 para determinismo.

# Carpeta donde viven los CSV del corpus.
CORPUS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "M3"

# Colecciones a traducir. Cada una: (archivo_look_en, archivo_label_en, prefijo_id, genero_humano)
# B = mujer, C = hombre. El prefijo evita colisiones de ID al unir las colecciones.
COLECCIONES = [
    {
        "nombre": "B_mujer",
        "look_en": "look_en (1).csv",
        "label_en": "label_en (1).csv",
        "prefijo": "B",
    },
    {
        "nombre": "C_hombre",
        "look_en": "look_en (2).csv",
        "label_en": "label_en (2).csv",
        "prefijo": "C",
    },
]

# Salidas unificadas (ambas colecciones juntas, ya con prefijo en los IDs).
OUT_LOOK = CORPUS_DIR / "corpus_look_es.csv"
OUT_LABEL = CORPUS_DIR / "corpus_label_es.csv"

# Caché de traducciones (texto_en -> texto_es). Reproducibilidad + ahorro.
CACHE_PATH = CORPUS_DIR / "traducciones_cache.json"

# Reporte de cobertura.
REPORTE_PATH = CORPUS_DIR / "reporte_traduccion.txt"

# Modelo traductor. Económico y suficiente para traducción.
# Se puede sobrescribir con la variable de entorno GEMINI_MODEL.
# Si el modelo indicado no existe, el script resuelve automáticamente uno
# disponible de tu cuenta (ver resolver_modelo()).
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Orden de preferencia para la resolución automática (del más disponible/barato
# al más potente). Las cuentas nuevas solo tienen la familia 3.x; los '-lite'
# suelen tener más capacidad libre y menos errores 503 que el flash grande.
MODELOS_PREFERIDOS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

# Celdas de texto por llamada al LLM. Bloques grandes = menos llamadas = menos
# exposición a picos de 503 y más rápido. 40 es un buen balance para flash.
BLOQUE = 40

# Reintentos.
MAX_REINTENTOS_BLOQUE = 3
MAX_REINTENTOS_CELDA = 4

# Pausa entre llamadas (rate limit).
PAUSA = 1.5

# Columnas de TEXTO LIBRE que se traducen con el LLM.
COLS_LOOK_TEXTO = ["look"]
COLS_LABEL_TEXTO = [
    "category",
    "title",
    "style",
    "outline",
    "materials",
    "color",
    "pattern",
    "detail",
    "donning/doffing",
]

# Columnas CATEGÓRICAS: traducción fija y determinista (sin LLM).
MAPA_GENDER = {
    "Men's": "De hombre",
    "Women's": "De mujer",
    "Unisex": "Unisex",
    "Boys'": "De niño",
    "Girls'": "De niña",
    "Kids'": "De niño",
}

MAPA_SEASON = {
    "Spring": "Primavera",
    "Summer": "Verano",
    "Autumn": "Otoño",
    "Fall": "Otoño",
    "Winter": "Invierno",
    "Spring/Summer": "Primavera/Verano",
    "Autumn/Winter": "Otoño/Invierno",
    "Fall/Winter": "Otoño/Invierno",
    "Spring/Fall": "Primavera/Otoño",
}

MAPA_OCCASION = {
    "Daily": "Diario",
    "Sports": "Deporte",
    "Travel": "Viaje",
    "School": "Escuela",
    "Social": "Eventos sociales",
    "Home": "Hogar",
    "Work": "Trabajo",
    "Commute": "Trabajo",
    "Party": "Fiesta",
    "Date": "Cita",
    "Daily/Sports": "Diario/Deporte",
    "Daily/Travel": "Diario/Viaje",
    "Daily/School": "Diario/Escuela",
    "Daily/Social": "Diario/Eventos sociales",
    "Sports/Travel": "Deporte/Viaje",
    "School/Social": "Escuela/Eventos sociales",
}


def mapear_categorica(valor: str, mapa: dict) -> str:
    """Traduce un valor categórico. Si trae '/', traduce cada parte.
    Si no está en el mapa, deja el valor tal cual (y se reportará como residuo)."""
    v = (valor or "").strip()
    if v in mapa:
        return mapa[v]
    if "/" in v:
        partes = [p.strip() for p in v.split("/")]
        traducidas = [mapa.get(p, p) for p in partes]
        return "/".join(traducidas)
    return mapa.get(v, v)


# ===========================================================================
# GLOSARIO DE DOMINIO (va en el prompt para fijar el vocabulario de moda)
# ===========================================================================

GLOSARIO = """
Glosario de moda (usa EXACTAMENTE estas traducciones cuando aplique):
- hoodie / hooded sweatshirt = sudadera con capucha
- sweatshirt = sudadera
- sweater / knit top = suéter / prenda de punto
- cardigan = cárdigan
- T-shirt / tee = camiseta
- long-sleeve = de manga larga ; short-sleeve = de manga corta
- tank top = camiseta sin mangas
- shirt = camisa ; blouse = blusa
- coat = abrigo ; trench coat = gabardina ; down jacket / puffer = chaqueta de plumón
- jacket = chaqueta ; blazer = blazer ; vest = chaleco
- trousers / pants = pantalones ; jeans = vaqueros ; wide-leg = de pierna ancha
- straight-leg = de corte recto ; cargo pants = pantalones cargo ; joggers = pantalón jogger
- shorts = pantalones cortos ; bermuda = bermudas
- skirt = falda ; pleated skirt = falda plisada ; mini skirt = minifalda ; maxi skirt = falda larga
- dress = vestido ; turtleneck = cuello alto ; crew neck = cuello redondo ; V-neck = cuello en V
- loafers = mocasines ; sneakers = zapatillas deportivas ; boots = botas ; sandals = sandalias
- beret = boina ; cap = gorra ; hat = sombrero ; scarf = bufanda ; socks = calcetines
- bag = bolso ; handbag = bolso de mano ; shoulder bag = bolso de hombro ; crossbody bag = bolso cruzado
- leather = cuero ; denim = mezclilla ; wool = lana ; cotton = algodón ; fleece = polar
- knit / knitwear = punto / prendas de punto ; sherpa = borreguito ; faux fur = piel sintética
- loose fit = corte holgado ; slim fit = corte ajustado ; regular fit = corte regular ; oversized = extragrande
- cropped = corto ; layered over = superpuesto a ; paired with = combinado con
- plaid / checkered = de cuadros ; striped = de rayas ; floral = floral ; argyle = de rombos
- solid color = color liso ; color-block = de bloques de color ; monochromatic = monocromático
- pullover (donning) = se pone por la cabeza ; button closure = cierre con botones ; zip = cremallera
- neutral tones = tonos neutros ; warm-toned = de tonos cálidos ; cool-toned = de tonos fríos
- vibe / aura / aesthetic = aire / estética ; effortless = desenfadado ; refined = refinado
- preppy = preppy (estilo universitario) ; vintage = vintage ; techwear = ropa técnica
- workwear = ropa de trabajo ; old money = estilo elegante clásico ("old money")
Nombres de estilo regionales: tradúcelos de forma natural, p. ej.
  "Japanese Cityboy style" = "estilo cityboy japonés",
  "Korean trendy style" = "estilo coreano moderno",
  "American vintage style" = "estilo vintage estadounidense".
"""

PROMPT_SISTEMA = """Eres un traductor profesional especializado en moda y e-commerce.
Traduce del INGLÉS al ESPAÑOL NEUTRO (apto para Colombia/Latinoamérica).

Reglas estrictas:
- Traduce el 100% del texto. NO dejes NINGUNA palabra en inglés.
- Produce español natural y fluido, NO traducción literal palabra por palabra.
- Mantén el tono descriptivo de catálogo de moda.
- NO agregues, quites ni inventes información; solo traduce lo que hay.
- Conserva mayúsculas iniciales y la puntuación del original cuando tenga sentido.
- Si un término no tiene equivalente claro, usa el término en español más cercano
  del glosario; nunca lo dejes en inglés.
""" + GLOSARIO


# ===========================================================================
# DETECTOR DE RESIDUOS EN INGLÉS
# ===========================================================================

# Palabras en inglés frecuentes en este corpus que delatan una traducción incompleta.
# Si una celda traducida contiene alguna (como palabra completa), se reintenta.
#
# IMPORTANTE: se EXCLUYEN de esta lista los préstamos aceptados en español de moda
# (denim, beige, vintage, casual, chic, blazer, preppy, techwear, etc.) y palabras
# que son español normal (color, top). Marcarlos como "inglés" causaba un bucle de
# retraducción infinito, porque la traducción correcta los conserva.
PALABRAS_INGLES = [
    # conectores y gramática inglesa (señal inequívoca de inglés sin traducir)
    "with", "and", "the", "for", "paired", "layered", "over", "suitable",
    "yet", "that", "both", "including", "an", "echoing", "evoke", "evokes",
    # verbos/adjetivos ingleses de descripción
    "feel", "vibe", "aura", "aesthetic", "loose", "slim", "cropped", "wide",
    "matching", "effortless", "refined", "elegant", "balancing", "balance",
    "enhance", "enhances", "reinforce", "reinforces", "highlight", "adds",
    "offer", "offers", "showing", "fluffy", "girlish", "sweet", "clean",
    "steady", "versatile", "versatility", "sophisticated", "soft",
    "textures", "textured", "smart", "monochromatic", "basics", "essential",
    "highly", "full", "wearability", "edgy", "rugged", "ruggedness",
    "distressed", "shearling", "thoughtful", "styling", "setting", "settings",
    "functional", "functionality", "utility", "spliced", "splicing", "stacked",
    "heavyweight", "wrinkle", "resistant", "flattering", "layering",
    "deconstructed", "striking", "prominent", "depth", "break", "norm",
    "commuter", "commute", "ready", "office", "scenarios", "scenario",
    # sustantivos de prenda/parte claramente ingleses
    "jacket", "pants", "trousers", "jeans", "skirt", "dress", "shirt",
    "sleeve", "sleeves", "collar", "pocket", "pockets", "button", "buttons",
    "closure", "pullover", "hooded", "hoodie", "sweater", "sweatshirt",
    "shoes", "socks", "scarf", "belt", "silhouette", "zipper",
    # colores en inglés (sus equivalentes ES son otros; 'beige' SÍ es válido en ES)
    "white", "black", "gray", "grey", "navy", "brown", "green", "red", "pink",
    # materiales/patrones en inglés (denim/wool->lana se traducen; 'denim' es válido ES)
    "knitwear", "cotton", "leather", "fabric", "patterned", "printed",
    "argyle", "turtleneck", "crossbody",
    # estaciones/ocasiones en inglés (deben quedar mapeadas por diccionario)
    "winter", "summer", "spring", "autumn", "fall",
    "sports", "travel", "school", "social", "home",
    "mens", "womens", "unisex",
]

# Préstamos y términos VÁLIDOS en español de moda: NUNCA se marcan como residuo
# aunque "parezcan" inglés. (Se dejan documentados para claridad.)
TERMINOS_VALIDOS_ES = {
    "denim", "beige", "vintage", "casual", "chic", "preppy", "techwear",
    "blazer", "top", "color", "look", "style",  # 'look'/'style' solo quedan si el
    # modelo dejó "... style/look" sin traducir; se filtran por los conectores ingleses
    # vecinos, no por sí solos.
}

# Se compila un patrón que busca cualquiera de esas palabras como token completo
# (sin importar mayúsculas). \b asegura palabra completa (evita falsos positivos
# dentro de palabras en español).
_PATRON_INGLES = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(set(PALABRAS_INGLES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Caracteres problemáticos que hay que normalizar en el ORIGINAL antes de traducir.
# (p. ej. la coma de ancho completo que aparece en look_en (2).csv: "p4，p5")
def normalizar_items(items: str) -> str:
    """Normaliza el campo items: comas de ancho completo -> coma ASCII, sin espacios raros."""
    if items is None:
        return ""
    s = items.replace("，", ",").replace("、", ",")
    partes = [p.strip() for p in s.split(",") if p.strip()]
    return ",".join(partes)


# Correcciones deterministas de palabras que el LLM tiende a dejar sin traducir
# (nombres de estilo recurrentes). Se aplican por reemplazo de PALABRA COMPLETA,
# respetando mayúscula inicial. Son términos que, por ser nombres de estilo, el
# modelo repite en inglés; corregirlos aquí garantiza el 100% sin más llamadas.
CORRECCIONES_FIJAS = {
    "commuter": "para desplazamientos urbanos",
    "smart": "elegante",
    "sophisticated": "sofisticado",
    "black": "negro",
    "slim": "ajustado",
    "shoes": "zapatos",
    "collar": "cuello",
}


def aplicar_correcciones(texto: str) -> str:
    """Reemplaza términos residuales por su equivalente en español (palabra completa)."""
    if not texto:
        return texto
    def _rep(m):
        original = m.group(0)
        trad = CORRECCIONES_FIJAS[original.lower()]
        # Conservar mayúscula inicial si la tenía.
        return trad[0].upper() + trad[1:] if original[0].isupper() else trad
    patron = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in CORRECCIONES_FIJAS) + r")\b",
        re.IGNORECASE,
    )
    return patron.sub(_rep, texto)


def residuos_ingles(texto: str) -> list[str]:
    """Devuelve la lista de palabras en inglés detectadas en el texto (vacía si está limpio).
    Filtra los términos aceptados en español (denim, beige, vintage, casual, etc.)."""
    if not texto:
        return []
    hallazgos = _PATRON_INGLES.findall(texto)
    return [h for h in hallazgos if h.lower() not in TERMINOS_VALIDOS_ES]


# ===========================================================================
# CLIENTE GEMINI
# ===========================================================================

def crear_cliente():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY.\n"
            "  PowerShell:  $env:GEMINI_API_KEY=\"tu-api-key\"\n"
            "  Linux/Colab: export GEMINI_API_KEY=\"tu-api-key\"\n\n"
            "Alternativa sin API de pago: ver traducir_bloque_local() en este archivo."
        )
    from google import genai
    return genai.Client(api_key=api_key)


def resolver_modelo(cliente) -> str:
    """Devuelve un nombre de modelo que exista en la cuenta.
    Prioriza MODEL (o GEMINI_MODEL); si no existe, prueba MODELOS_PREFERIDOS;
    si tampoco, elige el primer modelo 'flash' que soporte generateContent."""
    global MODEL

    # Lista de modelos disponibles para esta API key.
    disponibles = []
    try:
        for m in cliente.models.list():
            nombre = getattr(m, "name", "") or ""
            # Los nombres vienen como "models/gemini-...". Normalizamos.
            corto = nombre.split("/")[-1]
            acciones = getattr(m, "supported_actions", None) or \
                getattr(m, "supported_generation_methods", None) or []
            if (not acciones) or ("generateContent" in acciones):
                disponibles.append(corto)
    except Exception as e:
        print(f"  [aviso] no se pudo listar modelos ({e}); se usará '{MODEL}' tal cual.")
        return MODEL

    if not disponibles:
        print(f"  [aviso] la cuenta no devolvió modelos; se usará '{MODEL}' tal cual.")
        return MODEL

    # 1) Respetar el modelo pedido si existe.
    if MODEL in disponibles:
        return MODEL
    # 2) Probar la lista de preferidos en orden.
    for cand in MODELOS_PREFERIDOS:
        if cand in disponibles:
            print(f"  [info] '{MODEL}' no disponible; usando '{cand}'.")
            MODEL = cand
            return cand
    # 3) Último recurso: primer 'flash' disponible, o el primero de la lista.
    flash = [m for m in disponibles if "flash" in m and "embedding" not in m]
    elegido = (flash or disponibles)[0]
    print(f"  [info] '{MODEL}' no disponible; usando '{elegido}'. "
          f"Modelos detectados: {disponibles[:8]}...")
    MODEL = elegido
    return elegido


def _es_error_temporal(e: Exception) -> bool:
    """True si el error es transitorio (sobrecarga/límite) y conviene reintentar con espera.
    Los errores permanentes (404 NOT_FOUND, 400, 401/403 de permisos) NUNCA se reintentan:
    insistir no sirve y solo oculta el problema real (p. ej. modelo inexistente)."""
    s = str(e).lower()
    # Permanentes: cortar de inmediato.
    if any(p in s for p in ["404", "not_found", "not found", "400", "invalid",
                            "permission", "401", "403", "unauthenticated", "api key"]):
        return False
    # Temporales: reintentar con backoff.
    return any(code in s for code in ["503", "429", "500", "unavailable",
                                      "high demand", "overloaded", "rate limit",
                                      "resource_exhausted", "deadline", "timeout"])


def _generar_con_backoff(cliente, model, contents, config, etiqueta: str,
                         max_intentos: int = 8):
    """Llama a generate_content con reintentos y espera exponencial para errores
    temporales (503/429/500). Lanza la excepción si el error es permanente o se
    agotan los intentos."""
    espera = 4.0
    for intento in range(1, max_intentos + 1):
        try:
            return cliente.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as e:
            if _es_error_temporal(e) and intento < max_intentos:
                print(f"      [espera] {etiqueta}: error temporal (intento {intento}/{max_intentos}), "
                      f"reintento en {espera:.0f}s")
                time.sleep(espera)
                espera = min(espera * 1.8, 60.0)  # backoff exponencial, tope 60s
                continue
            raise
    raise RuntimeError(f"{etiqueta}: agotados los reintentos")


def _limpiar_json(texto: str) -> str:
    """Quita fences ```json y recorta al primer objeto JSON."""
    if not texto:
        return ""
    t = texto.strip()
    if t.startswith("```"):
        lineas = t.splitlines()
        if lineas and lineas[0].startswith("```"):
            lineas = lineas[1:]
        if lineas and lineas[-1].strip() == "```":
            lineas = lineas[:-1]
        t = "\n".join(lineas).strip()
    ini, fin = t.find("{"), t.rfind("}")
    if ini != -1 and fin != -1 and fin > ini:
        t = t[ini:fin + 1]
    return t.strip()


def traducir_bloque_llm(cliente, textos: list[str]) -> dict[int, str]:
    """Traduce un bloque de textos. Devuelve {indice: traduccion}.
    Usa un JSON indexado para no desalinear filas."""
    from google.genai import types

    entrada = {str(i): t for i, t in enumerate(textos)}
    prompt = (
        "Traduce al español cada valor del siguiente JSON. "
        "Responde ÚNICAMENTE con un objeto JSON con las MISMAS claves, "
        "donde cada valor es la traducción al español del valor original.\n\n"
        "JSON a traducir:\n" + json.dumps(entrada, ensure_ascii=False, indent=2)
    )

    for intento in range(1, MAX_REINTENTOS_BLOQUE + 1):
        try:
            resp = _generar_con_backoff(
                cliente, MODEL, prompt,
                types.GenerateContentConfig(
                    system_instruction=PROMPT_SISTEMA,
                    temperature=0.0,  # determinismo / reproducibilidad
                    response_mime_type="application/json",
                ),
                etiqueta=f"bloque intento {intento}",
            )
            data = json.loads(_limpiar_json(resp.text))
            out: dict[int, str] = {}
            for k, v in data.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                if isinstance(v, str) and v.strip():
                    out[idx] = v.strip()
            if len(out) == len(textos):
                return out
            print(f"    [aviso] bloque incompleto {len(out)}/{len(textos)} (intento {intento})")
        except Exception as e:
            print(f"    [aviso] error de bloque (intento {intento}): {e}")
        time.sleep(2)

    # Como último recurso, traducir celda por celda.
    out = {}
    for i, t in enumerate(textos):
        out[i] = traducir_celda_llm(cliente, t)
    return out


def traducir_celda_llm(cliente, texto: str) -> str:
    """Traduce una sola celda (usado para reintentos de residuos)."""
    from google.genai import types

    prompt = (
        "Traduce COMPLETAMENTE al español el siguiente texto de un catálogo de moda. "
        "No dejes ninguna palabra en inglés. Responde solo con la traducción, sin comillas:\n\n"
        + texto
    )
    for intento in range(1, MAX_REINTENTOS_CELDA + 1):
        try:
            resp = _generar_con_backoff(
                cliente, MODEL, prompt,
                types.GenerateContentConfig(
                    system_instruction=PROMPT_SISTEMA,
                    temperature=0.0,
                ),
                etiqueta=f"celda intento {intento}",
            )
            t = (resp.text or "").strip().strip('"').strip()
            if t:
                return t
        except Exception as e:
            print(f"      [aviso] error de celda (intento {intento}): {e}")
        time.sleep(2)
    return texto  # se devolverá el original; el reporte lo marcará como residuo


# ===========================================================================
# CACHÉ
# ===========================================================================

def cargar_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def guardar_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ===========================================================================
# TRADUCCIÓN DE UN CONJUNTO DE TEXTOS ÚNICOS (con caché + verificación)
# ===========================================================================

def traducir_textos(cliente, textos_unicos: list[str], cache: dict[str, str]) -> None:
    """Traduce in-place en la caché todos los textos que falten.
    Después verifica residuos y reintenta celda por celda hasta limpiar."""
    pendientes = [t for t in textos_unicos if t and t not in cache]
    print(f"  Textos únicos por traducir: {len(pendientes)} (en caché: {len(textos_unicos) - len(pendientes)})")

    # 1) Traducción por bloques.
    total_bloques = (len(pendientes) + BLOQUE - 1) // BLOQUE
    for inicio in range(0, len(pendientes), BLOQUE):
        bloque = pendientes[inicio:inicio + BLOQUE]
        nb = inicio // BLOQUE + 1
        print(f"    Bloque {nb}/{total_bloques} ({len(bloque)} celdas, "
              f"{inicio}/{len(pendientes)} hechas)...")
        traducciones = traducir_bloque_llm(cliente, bloque)
        for i, original in enumerate(bloque):
            cache[original] = traducciones.get(i, original)
        guardar_cache(cache)  # se persiste tras cada bloque: si se corta, no se pierde nada
        time.sleep(PAUSA)

    # 2) Corrección determinista de residuos conocidos (sin llamadas al LLM).
    for original in textos_unicos:
        if original and original in cache:
            corregido = aplicar_correcciones(cache[original])
            if corregido != cache[original]:
                cache[original] = corregido
    guardar_cache(cache)

    # 3) Verificación de residuos restantes + reintento por celda (máx 2 pasadas;
    #    si no limpia, se deja y queda anotado en el reporte final; NO se bloquea).
    MAX_PASADAS_VERIF = 2
    for original in textos_unicos:
        if not original:
            continue
        intentos = 0
        while residuos_ingles(cache.get(original, "")) and intentos < MAX_PASADAS_VERIF:
            intentos += 1
            res = residuos_ingles(cache[original])
            print(f"    [reintento celda {intentos}/{MAX_PASADAS_VERIF}] residuos {sorted(set(res))[:5]} -> retraduciendo")
            try:
                cache[original] = traducir_celda_llm(cliente, original)
            except Exception as e:
                print(f"      [aviso] no se pudo retraducir ahora ({e}); se deja y se anota.")
                break
            guardar_cache(cache)
            time.sleep(1)


# ===========================================================================
# LECTURA / ESCRITURA DE CSV
# ===========================================================================

def leer_csv(path: Path) -> tuple[list[str], list[dict]]:
    # utf-8-sig descarta el BOM (\ufeff) que traen los CSV exportados desde Excel,
    # evitando que la primera columna quede como '\ufeffoutfitID'.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        filas = list(reader)
        return reader.fieldnames or [], filas


def escribir_csv(path: Path, fieldnames: list[str], filas: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas)


# ===========================================================================
# PROCESAMIENTO DE UNA COLECCIÓN
# ===========================================================================

def procesar_coleccion(cliente, coleccion: dict, cache: dict[str, str]):
    nombre = coleccion["nombre"]
    prefijo = coleccion["prefijo"]
    print(f"\n=== Colección {nombre} (prefijo {prefijo}_) ===")

    look_path = CORPUS_DIR / coleccion["look_en"]
    label_path = CORPUS_DIR / coleccion["label_en"]

    cols_look, filas_look = leer_csv(look_path)
    cols_label, filas_label = leer_csv(label_path)
    print(f"  look_en: {len(filas_look)} outfits | label_en: {len(filas_label)} prendas")

    # --- Recolectar textos únicos a traducir (look + label) ---
    textos = set()
    for fila in filas_look:
        for c in COLS_LOOK_TEXTO:
            if fila.get(c):
                textos.add(fila[c])
    for fila in filas_label:
        for c in COLS_LABEL_TEXTO:
            if fila.get(c):
                textos.add(fila[c])

    traducir_textos(cliente, sorted(textos), cache)

    # --- Construir filas traducidas de LOOK ---
    out_look = []
    for fila in filas_look:
        nueva = dict(fila)
        nueva["outfitID"] = f"{prefijo}_{fila['outfitID']}"
        # Reprefijar los items y normalizar comas de ancho completo.
        items_norm = normalizar_items(fila.get("items", ""))
        nueva["items"] = ",".join(f"{prefijo}_{p}" for p in items_norm.split(",") if p)
        for c in COLS_LOOK_TEXTO:
            if fila.get(c):
                nueva[c] = cache.get(fila[c], fila[c])
        nueva["season"] = mapear_categorica(fila.get("season", ""), MAPA_SEASON)
        nueva["occasion"] = mapear_categorica(fila.get("occasion", ""), MAPA_OCCASION)
        nueva["coleccion"] = nombre
        out_look.append(nueva)

    # --- Construir filas traducidas de LABEL ---
    out_label = []
    for fila in filas_label:
        nueva = dict(fila)
        nueva["itemID"] = f"{prefijo}_{fila['itemID']}"
        for c in COLS_LABEL_TEXTO:
            if fila.get(c):
                nueva[c] = cache.get(fila[c], fila[c])
        nueva["gender"] = mapear_categorica(fila.get("gender", ""), MAPA_GENDER)
        nueva["coleccion"] = nombre
        out_label.append(nueva)

    return out_look, out_label


# ===========================================================================
# REPORTE DE COBERTURA
# ===========================================================================

def generar_reporte(out_look: list[dict], out_label: list[dict]) -> int:
    """Verifica que no queden residuos de inglés. Devuelve el número de celdas con residuos."""
    lineas = ["REPORTE DE COBERTURA DE TRADUCCIÓN", "=" * 40, ""]
    total_celdas = 0
    con_residuo = 0

    def revisar(filas, id_col, cols):
        nonlocal total_celdas, con_residuo
        problemas = []
        for fila in filas:
            for c in cols:
                val = fila.get(c, "")
                if not val:
                    continue
                total_celdas += 1
                res = residuos_ingles(val)
                if res:
                    con_residuo += 1
                    problemas.append(f"  [{fila.get(id_col)}].{c}: {sorted(set(res))} -> {val!r}")
        return problemas

    p_look = revisar(out_look, "outfitID", COLS_LOOK_TEXTO + ["season", "occasion"])
    p_label = revisar(out_label, "itemID", COLS_LABEL_TEXTO + ["gender"])

    lineas.append(f"Celdas de texto revisadas: {total_celdas}")
    lineas.append(f"Celdas con residuos de inglés: {con_residuo}")
    lineas.append(f"Cobertura: {100 * (total_celdas - con_residuo) / max(1, total_celdas):.2f}%")
    lineas.append("")

    if p_look:
        lineas.append("Residuos en LOOK:")
        lineas.extend(p_look)
        lineas.append("")
    if p_label:
        lineas.append("Residuos en LABEL:")
        lineas.extend(p_label)
        lineas.append("")
    if not p_look and not p_label:
        lineas.append("Sin residuos. Traducción al 100%. ✓")

    REPORTE_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas[:6]))
    print(f"\nReporte completo: {REPORTE_PATH}")
    return con_residuo


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("Traducción del corpus de moda EN -> ES (por bloques, cobertura garantizada)")
    print(f"Carpeta del corpus: {CORPUS_DIR}")

    faltan = [c for c in COLECCIONES
              if not (CORPUS_DIR / c["look_en"]).exists() or not (CORPUS_DIR / c["label_en"]).exists()]
    if faltan:
        print("ERROR: faltan archivos en inglés:")
        for c in faltan:
            print(f"  - {c['look_en']} / {c['label_en']}")
        sys.exit(1)

    cliente = crear_cliente()
    modelo = resolver_modelo(cliente)
    print(f"Modelo de traducción: {modelo}")
    cache = cargar_cache()

    todos_look, todos_label = [], []
    for col in COLECCIONES:
        ol, olb = procesar_coleccion(cliente, col, cache)
        todos_look.extend(ol)
        todos_label.extend(olb)

    # Escribir CSV unificados (con columna extra 'coleccion').
    cols_look_out = ["outfitID", "items", "look", "season", "occasion", "coleccion"]
    cols_label_out = ["itemID", "category", "title", "gender", "style", "outline",
                      "materials", "color", "pattern", "detail", "donning/doffing", "coleccion"]
    escribir_csv(OUT_LOOK, cols_look_out, todos_look)
    escribir_csv(OUT_LABEL, cols_label_out, todos_label)
    print(f"\n✅ {OUT_LOOK.name}: {len(todos_look)} outfits")
    print(f"✅ {OUT_LABEL.name}: {len(todos_label)} prendas")

    # Verificación final de cobertura.
    con_residuo = generar_reporte(todos_look, todos_label)
    if con_residuo > 0:
        print(f"\n⚠️  Quedaron {con_residuo} celdas con residuos de inglés. "
              f"Revisa {REPORTE_PATH.name} y reejecuta (la caché conserva lo ya traducido).")
        sys.exit(2)
    print("\n✅ Traducción verificada al 100%.")


# ===========================================================================
# ALTERNATIVA SIN API DE PAGO (opcional, documentada)
# ===========================================================================
#
# Si no se dispone de GEMINI_API_KEY, se puede traducir con un modelo open-source
# local (corre en Colab/CPU-GPU sin claves):
#
#   pip install transformers sentencepiece torch
#
#   from transformers import pipeline
#   _traductor = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
#
#   def traducir_celda_local(texto: str) -> str:
#       return _traductor(texto, max_length=512)[0]["translation_text"]
#
# Para usarla, reemplaza las llamadas a traducir_celda_llm / traducir_bloque_llm
# por una versión que use traducir_celda_local. Nota: opus-mt no conoce jerga de
# moda, así que conviene mantener el glosario como post-proceso de reemplazo por
# palabra completa (\b) sobre su salida.


if __name__ == "__main__":
    main()
