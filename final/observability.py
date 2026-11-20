import atexit
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RESULTS_DIR = Path(__file__).parent / "resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "logs_observabilidad.jsonl"
WANDB_STATE_PATH = RESULTS_DIR / "wandb_state.json"

_LOCK = threading.Lock()
_WANDB_RUN = None
_WANDB_INIT_INTENTADO = False
_WANDB_ERROR: Optional[str] = None
_CONTADOR_PETICIONES = 0


def _anonimizar_texto(texto: str) -> str:
    """Genera un hash SHA-256 truncado para correlacionar sin guardar PII ni texto crudo."""
    if not texto:
        return "empty"
    return hashlib.sha256(texto.strip().lower().encode("utf-8")).hexdigest()[:12]


def _guardar_estado_wandb(conectado: bool, run_url: Optional[str] = None, error: Optional[str] = None) -> None:
    """Persiste el estado de W&B en disco para compartirlo entre workers ZeroGPU y la UI."""
    estado = {
        "conectado": conectado,
        "proyecto": os.getenv("WANDB_PROJECT", "personal-shopper-moda"),
        "run_url": run_url,
        "error": error,
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        WANDB_STATE_PATH.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _leer_estado_wandb() -> Dict[str, Any]:
    """Lee el último estado de conexión de W&B registrado por algún worker."""
    api_key_presente = bool(os.getenv("WANDB_API_KEY", "").strip())
    if not api_key_presente:
        return {
            "wandb_api_key_configurada": False,
            "wandb_conectado": False,
            "wandb_proyecto": None,
            "wandb_run_url": None,
            "wandb_error": "Falta configurar WANDB_API_KEY en Secrets del Space",
        }

    if WANDB_STATE_PATH.exists():
        try:
            data = json.loads(WANDB_STATE_PATH.read_text(encoding="utf-8"))
            return {
                "wandb_api_key_configurada": True,
                "wandb_conectado": bool(data.get("conectado", False)),
                "wandb_proyecto": data.get("proyecto", os.getenv("WANDB_PROJECT", "personal-shopper-moda")),
                "wandb_run_url": data.get("run_url"),
                "wandb_error": data.get("error"),
            }
        except Exception:
            pass

    return {
        "wandb_api_key_configurada": True,
        "wandb_conectado": _WANDB_RUN is not None,
        "wandb_proyecto": os.getenv("WANDB_PROJECT", "personal-shopper-moda"),
        "wandb_run_url": getattr(_WANDB_RUN, "url", None) if _WANDB_RUN is not None else None,
        "wandb_error": _WANDB_ERROR or "En espera de la primera petición para iniciar sesión en W&B",
    }


def _get_wandb_run():
    """Inicializa una sesión de W&B dentro del worker activo si WANDB_API_KEY está configurada."""
    global _WANDB_RUN, _WANDB_INIT_INTENTADO, _WANDB_ERROR
    if _WANDB_INIT_INTENTADO:
        return _WANDB_RUN

    _WANDB_INIT_INTENTADO = True
    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if not api_key:
        _WANDB_ERROR = "WANDB_API_KEY no encontrada en variables de entorno"
        _guardar_estado_wandb(conectado=False, error=_WANDB_ERROR)
        return None

    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        import wandb

        wandb.login(key=api_key, relogin=True)

        proyecto = os.getenv("WANDB_PROJECT", "personal-shopper-moda").strip()
        entity = os.getenv("WANDB_ENTITY", "").strip() or None
        run_name = os.getenv("WANDB_RUN_NAME", f"hf-zerogpu-{uuid.uuid4().hex[:6]}")

        _WANDB_RUN = wandb.init(
            project=proyecto,
            entity=entity,
            name=run_name,
            group="hf-spaces-prod",
            job_type="production-inference",
            config={
                "llm_modelo": "Qwen/Qwen2.5-1.5B-Instruct",
                "vlm_modelo": "Qwen/Qwen2-VL-2B-Instruct",
                "embedder": "intfloat/multilingual-e5-small",
                "retrieval": "hybrid_bm25_dense_hyde_rrf",
                "hardware": "HuggingFace-ZeroGPU",
            },
            reinit=False,
        )

        # Definir eje X limpio por número de petición
        _WANDB_RUN.define_metric("inferencia/peticion_num")
        _WANDB_RUN.define_metric("inferencia/*", step_metric="inferencia/peticion_num")

        run_url = getattr(_WANDB_RUN, "url", None)
        _WANDB_ERROR = None
        _guardar_estado_wandb(conectado=True, run_url=run_url, error=None)

        atexit.register(lambda: _WANDB_RUN.finish() if _WANDB_RUN is not None else None)
    except Exception as exc:
        _WANDB_RUN = None
        _WANDB_ERROR = f"{type(exc).__name__}: {exc}"
        _guardar_estado_wandb(conectado=False, error=_WANDB_ERROR)

    return _WANDB_RUN


def _calcular_agregados(lineas: List[str], ultimos_n: int = 100) -> Dict[str, Any]:
    """Calcula percentiles de latencia y tasas operativas a partir del log JSONL."""
    registros = []
    for ln in lineas[-ultimos_n:]:
        try:
            registros.append(json.loads(ln))
        except Exception:
            continue

    if not registros:
        return {
            "registros": [],
            "latencia_p50_ms": 0.0,
            "latencia_p95_ms": 0.0,
            "tasa_herramienta_pct": 0.0,
            "tasa_valvula_pct": 0.0,
            "tasa_cache_pct": 0.0,
        }

    lats = [r["latencia_ms"] for r in registros if "latencia_ms" in r]
    herr = [int(r.get("uso_herramienta", False)) for r in registros]
    valv = [int(r.get("valvula_activada", False)) for r in registros]
    cach = [int(r.get("cache_hit", False)) for r in registros]

    return {
        "registros": registros,
        "latencia_p50_ms": round(float(np.percentile(lats, 50)), 1) if lats else 0.0,
        "latencia_p95_ms": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
        "tasa_herramienta_pct": round(100.0 * float(np.mean(herr)), 1),
        "tasa_valvula_pct": round(100.0 * float(np.mean(valv)), 1),
        "tasa_cache_pct": round(100.0 * float(np.mean(cach)), 1),
    }


def registrar_evento(
    *,
    consulta: str,
    tiene_imagen: bool,
    es_moda: bool,
    uso_herramienta: bool,
    valvula_activada: bool,
    outfits_recuperados: List[str],
    latencia_s: float,
    cache_hit: bool = False,
    estado: str = "ok",
    detalle_error: str = "",
) -> Dict[str, Any]:
    """Escribe un evento JSON estructurado local y lo envía a Weights & Biases sin PII."""
    global _CONTADOR_PETICIONES, _WANDB_ERROR
    _CONTADOR_PETICIONES += 1

    evento = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4())[:8],
        "query_hash": _anonimizar_texto(consulta),
        "query_chars": len(consulta or ""),
        "tiene_imagen": bool(tiene_imagen),
        "es_moda": bool(es_moda),
        "uso_herramienta": bool(uso_herramienta),
        "valvula_activada": bool(valvula_activada),
        "num_outfits": len(outfits_recuperados),
        "cache_hit": bool(cache_hit),
        "latencia_ms": round(latencia_s * 1000.0, 1),
        "estado": estado,
    }
    if detalle_error:
        evento["error_tipo"] = detalle_error[:120]

    with _LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        lineas = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()

    agregados = _calcular_agregados(lineas, ultimos_n=100)
    total_global = len(lineas)

    run = _get_wandb_run()
    if run is not None:
        try:
            run.log(
                {
                    "inferencia/peticion_num": total_global,
                    "inferencia/latencia_s": round(latencia_s, 3),
                    "inferencia/latencia_ms": round(latencia_s * 1000.0, 1),
                    "inferencia/uso_herramienta": int(uso_herramienta),
                    "inferencia/valvula_activada": int(valvula_activada),
                    "inferencia/cache_hit": int(cache_hit),
                    "inferencia/tiene_imagen": int(tiene_imagen),
                    "inferencia/es_moda": int(es_moda),
                    "inferencia/num_outfits": len(outfits_recuperados),
                    "inferencia/error": int(estado == "error"),
                    "resumen/latencia_p50_ms": agregados["latencia_p50_ms"],
                    "resumen/latencia_p95_ms": agregados["latencia_p95_ms"],
                    "resumen/tasa_herramienta_pct": agregados["tasa_herramienta_pct"],
                    "resumen/tasa_valvula_pct": agregados["tasa_valvula_pct"],
                },
                commit=True,
            )
            run.summary.update({
                "total_peticiones": total_global,
                "latencia_p50_ms": agregados["latencia_p50_ms"],
                "latencia_p95_ms": agregados["latencia_p95_ms"],
                "tasa_herramienta_pct": agregados["tasa_herramienta_pct"],
                "tasa_valvula_pct": agregados["tasa_valvula_pct"],
                "tasa_cache_pct": agregados["tasa_cache_pct"],
            })
        except Exception as exc:
            _WANDB_ERROR = f"{type(exc).__name__}: {exc}"
            _guardar_estado_wandb(conectado=False, run_url=getattr(run, "url", None), error=_WANDB_ERROR)

    return evento


def obtener_resumen_metricas(ultimos_n: int = 100) -> Dict[str, Any]:
    """Lee los últimos N registros del log JSONL y el estado de W&B para el panel de observabilidad."""
    estado_wandb = _leer_estado_wandb()

    if not LOG_PATH.exists():
        return {
            **estado_wandb,
            "total_peticiones": 0,
            "latencia_p50_ms": 0.0,
            "latencia_p95_ms": 0.0,
            "ultimos_logs": [],
        }

    with _LOCK:
        lineas = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()

    agregados = _calcular_agregados(lineas, ultimos_n=ultimos_n)
    registros = agregados["registros"]

    return {
        **estado_wandb,
        "total_peticiones": len(lineas),
        "latencia_p50_ms": agregados["latencia_p50_ms"],
        "latencia_p95_ms": agregados["latencia_p95_ms"],
        "tasa_herramienta_pct": agregados["tasa_herramienta_pct"],
        "tasa_valvula_pct": agregados["tasa_valvula_pct"],
        "tasa_cache_pct": agregados["tasa_cache_pct"],
        "ultimos_logs": registros[-10:][::-1],
    }