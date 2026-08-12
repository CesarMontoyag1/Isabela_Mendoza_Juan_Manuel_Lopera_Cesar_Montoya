"""
generate_synthetic_dataset.py

Genera el dataset sintético de entrenamiento: pares (mensaje de chat de un cliente,
perfil de estilo en JSON) para el fine-tuning con LoRA, utilizando la API de Google Gemini.

Cómo funciona:

1. Se generan combinaciones aleatorias de 2 a 5 atributos de la ontología.
2. Las combinaciones se agrupan en lotes para reducir el número de llamadas a Gemini.
3. Por cada lote, Gemini genera varias formas naturales de pedir cada perfil por chat.
4. Cada respuesta se valida como JSON antes de entrar al dataset.
5. Se guarda todo en data/train.jsonl y data/val.jsonl (80/20).

Requiere:
pip install google-genai

Configurar API key:

Linux / Cloudspace:
export GEMINI_API_KEY="tu-api-key"

Windows PowerShell:
$env:GEMINI_API_KEY="tu-api-key"

Uso:
python data_generation/generate_synthetic_dataset.py
"""

import json
import os
import random
import time

from google import genai


# --- Configuración ---------------------------------------------------------

SEED = 42

N_COMBINACIONES = 260

PARAFRASEOS_POR_COMBO = 5

# Número de perfiles enviados en una sola petición a Gemini.
# 10 perfiles × 5 parafraseos = hasta 50 ejemplos por llamada.
COMBINACIONES_POR_LLAMADA = 10

VAL_FRACTION = 0.2

# Modelo económico adecuado para generación de grandes cantidades de texto.
MASTER_MODEL = "gemini-3.5-flash-lite"

# Tiempo entre llamadas.
# Se mantiene una pequeña pausa para evitar problemas de rate limit.
PAUSA_ENTRE_LLAMADAS = 2


OUT_DIR = "data"

TRAIN_PATH = os.path.join(OUT_DIR, "train.jsonl")
VAL_PATH = os.path.join(OUT_DIR, "val.jsonl")


# Debe coincidir EXACTAMENTE con la ontología usada
# en el notebook de entrenamiento.

ONTOLOGIA = {
    "estilo": [
        "Casual",
        "Formal",
        "Minimalista",
        "Urbano",
        "Bohemio",
        "Deportivo",
        "Clasico",
    ],
    "ocasion": [
        "boda",
        "trabajo",
        "fin_de_semana",
        "viaje",
        "deporte",
        "evento_formal",
    ],
    "clima": [
        "calido",
        "frio",
        "templado",
    ],
    "paleta": [
        "neutros",
        "pasteles",
        "oscuros",
        "colores_vivos",
        "monocromatico",
    ],
    "fit": [
        "holgado",
        "regular",
        "ajustado",
        "oversized",
    ],
}

CAMPOS = list(ONTOLOGIA.keys())


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------

PROMPT_GENERACION = """
Actúa como un cliente colombiano de un e-commerce de moda escribiéndole
a un personal shopper por chat.

Debes generar mensajes para los perfiles de estilo proporcionados.

IMPORTANTE:

- Cada perfil tiene un identificador numérico.
- Debes generar exactamente {n} mensajes para CADA perfil.
- Cada mensaje debe corresponder únicamente al perfil que se indica.
- No mezcles atributos entre perfiles.
- No menciones los nombres exactos de las categorías tal como aparecen
  en el JSON.
- El modelo que posteriormente procesará estos mensajes debe INFERIR
  los atributos a partir del lenguaje natural.
- Usa lenguaje natural de chat.
- Puedes utilizar modismos colombianos ocasionales.
- Puedes incluir errores de tipeo leves de forma ocasional.
- Cada mensaje debe reflejar TODOS los campos presentes en su perfil,
  aunque algunos puedan expresarse de manera implícita.
- Los mensajes deben ser diferentes entre sí.
- Evita repetir exactamente las mismas estructuras.
- No escribas explicaciones.
- No agregues texto fuera del JSON.

PERFILES:

{perfiles}

RESPONDE ÚNICAMENTE CON UN OBJETO JSON con esta estructura:

{{
  "1": [
    "mensaje 1",
    "mensaje 2",
    "mensaje 3",
    "mensaje 4",
    "mensaje 5"
  ],
  "2": [
    "mensaje 1",
    "mensaje 2",
    "mensaje 3",
    "mensaje 4",
    "mensaje 5"
  ]
}}

Los números deben coincidir exactamente con los identificadores
de los perfiles proporcionados.
"""


# ---------------------------------------------------------------------------
# GENERACIÓN DE COMBINACIONES
# ---------------------------------------------------------------------------

def generar_combinaciones(n: int, rnd: random.Random) -> list[dict]:
    """
    Genera n combinaciones únicas de 2 a 5 campos de la ontología
    con valores aleatorios.
    """

    combinaciones = set()

    intentos = 0

    while len(combinaciones) < n and intentos < n * 20:

        intentos += 1

        k = rnd.randint(2, len(CAMPOS))

        campos_elegidos = tuple(
            sorted(rnd.sample(CAMPOS, k))
        )

        valores = tuple(
            rnd.choice(ONTOLOGIA[c])
            for c in campos_elegidos
        )

        combinaciones.add(
            (campos_elegidos, valores)
        )

    return [
        dict(zip(campos, valores))
        for campos, valores in combinaciones
    ]


# ---------------------------------------------------------------------------
# LIMPIEZA DE JSON
# ---------------------------------------------------------------------------

def limpiar_bloque_json(texto: str) -> str:
    """
    Limpia posibles bloques Markdown antes de intentar hacer json.loads().
    """

    if not texto:
        return ""

    texto = texto.strip()

    # ```json ... ```
    if texto.startswith("```"):
        lineas = texto.splitlines()

        if lineas and lineas[0].startswith("```"):
            lineas = lineas[1:]

        if lineas and lineas[-1].strip() == "```":
            lineas = lineas[:-1]

        texto = "\n".join(lineas).strip()

    # Buscar el primer objeto JSON si Gemini agregó texto adicional.
    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio != -1 and fin != -1 and fin > inicio:
        texto = texto[inicio:fin + 1]

    return texto.strip()


# ---------------------------------------------------------------------------
# GENERACIÓN POR LOTES
# ---------------------------------------------------------------------------

def generar_parafraseos_lote(
    client: genai.Client,
    perfiles: list[dict],
    n: int,
    max_reintentos: int = 3,
) -> dict[int, list[str]]:
    """
    Envía varios perfiles a Gemini en una sola llamada.

    Retorna:

    {
        1: ["mensaje", "mensaje", ...],
        2: ["mensaje", "mensaje", ...]
    }
    """

    perfiles_para_prompt = {}

    for indice, perfil in enumerate(perfiles, start=1):

        perfiles_para_prompt[str(indice)] = perfil

    perfiles_json = json.dumps(
        perfiles_para_prompt,
        ensure_ascii=False,
        indent=2,
    )

    prompt = PROMPT_GENERACION.format(
        perfiles=perfiles_json,
        n=n,
    )

    for intento in range(1, max_reintentos + 1):

        try:

            respuesta = client.models.generate_content(
                model=MASTER_MODEL,
                contents=prompt,
            )

            texto = limpiar_bloque_json(
                respuesta.text
            )

            resultado = json.loads(texto)

            if not isinstance(resultado, dict):
                raise ValueError(
                    "Gemini no devolvió un objeto JSON."
                )

            resultado_final = {}

            for clave, mensajes in resultado.items():

                try:
                    indice = int(clave)
                except ValueError:
                    continue

                if not isinstance(mensajes, list):
                    continue

                mensajes_validos = [
                    m.strip()
                    for m in mensajes
                    if isinstance(m, str) and m.strip()
                ]

                if mensajes_validos:
                    resultado_final[indice] = mensajes_validos[:n]

            if resultado_final:

                return resultado_final

            print(
                f"  [aviso] respuesta sin perfiles válidos "
                f"(intento {intento}/{max_reintentos})"
            )

        except Exception as e:

            print(
                f"  [aviso] error en intento "
                f"{intento}/{max_reintentos}: {e}"
            )

        # Esperar un poco antes de reintentar.
        if intento < max_reintentos:
            time.sleep(5)

    return {}


# ---------------------------------------------------------------------------
# CONSTRUIR DATASET
# ---------------------------------------------------------------------------

def construir_dataset(
    client: genai.Client,
    rnd: random.Random,
) -> list[dict]:

    combinaciones = generar_combinaciones(
        N_COMBINACIONES,
        rnd,
    )

    ejemplos = []

    # Dividir las combinaciones en lotes.
    lotes = [
        combinaciones[i:i + COMBINACIONES_POR_LLAMADA]
        for i in range(
            0,
            len(combinaciones),
            COMBINACIONES_POR_LLAMADA,
        )
    ]

    print(
        f"Se generaron {len(combinaciones)} combinaciones."
    )

    print(
        f"Se procesarán en {len(lotes)} llamadas aproximadamente."
    )

    print(
        f"Cada llamada contiene hasta "
        f"{COMBINACIONES_POR_LLAMADA} perfiles."
    )

    print()

    for numero_lote, lote in enumerate(lotes, start=1):

        print(
            f"[Lote {numero_lote}/{len(lotes)}] "
            f"Generando {len(lote)} perfiles..."
        )

        resultados = generar_parafraseos_lote(
            client,
            lote,
            PARAFRASEOS_POR_COMBO,
        )

        generados_lote = 0

        for indice_local, perfil in enumerate(
            lote,
            start=1,
        ):

            mensajes = resultados.get(
                indice_local,
                [],
            )

            if not mensajes:

                print(
                    f"  [aviso] Perfil {indice_local} "
                    f"no generó mensajes."
                )

                continue

            for texto in mensajes:

                ejemplos.append(
                    {
                        "input": texto.strip(),
                        "output": perfil,
                    }
                )

                generados_lote += 1

        print(
            f"  ✓ {generados_lote} ejemplos generados "
            f"en este lote."
        )

        # Pausa pequeña entre requests.
        if numero_lote < len(lotes):
            time.sleep(PAUSA_ENTRE_LLAMADAS)

    return ejemplos


# ---------------------------------------------------------------------------
# VALIDACIÓN Y DIVISIÓN
# ---------------------------------------------------------------------------

def validar_y_dividir(
    ejemplos: list[dict],
    val_fraction: float,
    rnd: random.Random,
):

    validos = []

    for ejemplo in ejemplos:

        if not isinstance(
            ejemplo.get("input"),
            str,
        ):
            continue

        if not ejemplo["input"].strip():
            continue

        if not isinstance(
            ejemplo.get("output"),
            dict,
        ):
            continue

        validos.append(ejemplo)

    rnd.shuffle(validos)

    n_val = int(
        len(validos) * val_fraction
    )

    val = validos[:n_val]

    train = validos[n_val:]

    return train, val


# ---------------------------------------------------------------------------
# GUARDAR JSONL
# ---------------------------------------------------------------------------

def guardar_jsonl(
    ejemplos: list[dict],
    path: str,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for ejemplo in ejemplos:

            f.write(
                json.dumps(
                    ejemplo,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "Falta la variable de entorno GEMINI_API_KEY.\n"
            "En Cloudspace/Linux:\n"
            'export GEMINI_API_KEY="tu-api-key"\n\n'
            "En Colab:\n"
            "import os\n"
            'os.environ["GEMINI_API_KEY"] = "tu-api-key"'
        )

    client = genai.Client(
        api_key=api_key
    )

    rnd = random.Random(
        SEED
    )

    numero_llamadas = (
        N_COMBINACIONES
        + COMBINACIONES_POR_LLAMADA
        - 1
    ) // COMBINACIONES_POR_LLAMADA

    ejemplos_objetivo = (
        N_COMBINACIONES
        * PARAFRASEOS_POR_COMBO
    )

    print(
        f"Generando {N_COMBINACIONES} combinaciones "
        f"x {PARAFRASEOS_POR_COMBO} parafraseos"
    )

    print(
        f"Objetivo: ~{ejemplos_objetivo} ejemplos"
    )

    print(
        f"Modelo: {MASTER_MODEL}"
    )

    print(
        f"Perfiles por llamada: "
        f"{COMBINACIONES_POR_LLAMADA}"
    )

    print(
        f"Llamadas estimadas: "
        f"{numero_llamadas}"
    )

    print()

    ejemplos = construir_dataset(
        client,
        rnd,
    )

    print()
    print(
        f"✓ {len(ejemplos)} ejemplos generados "
        f"antes de dividir train/val."
    )

    train, val = validar_y_dividir(
        ejemplos,
        VAL_FRACTION,
        rnd,
    )

    guardar_jsonl(
        train,
        TRAIN_PATH,
    )

    guardar_jsonl(
        val,
        VAL_PATH,
    )

    print()
    print(
        f"✅ {TRAIN_PATH}: "
        f"{len(train)} ejemplos"
    )

    print(
        f"✅ {VAL_PATH}: "
        f"{len(val)} ejemplos"
    )


if __name__ == "__main__":
    main()