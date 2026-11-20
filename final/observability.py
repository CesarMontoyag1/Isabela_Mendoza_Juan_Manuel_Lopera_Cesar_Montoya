import atexit
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

LOG_PATH = Path(__file__).parent / "resultados" / "logs_observabilidad.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_WANDB_RUN = None
_WANDB_INIT_INTENTADO = False
_CONTADOR_PETICIONES = 0


def _anonimizar_texto(texto: str) -> str:
    """Genera un hash SHA-256 truncado para correlacionar sin guardar PII ni texto crudo."""
    if not texto:
        return "empty"
    return hashlib.sha256(texto.strip().lower().encode("utf-8")).hexdigest()[:12]


def _get_wandb_run():
    """Inicializa una única sesión persistente de W&B si WANDB_API_KEY está configurada."""
    global _WANDB_RUN, _WANDB_INIT_INTENTADO
    if _WANDB_INIT_INTENTADO:
        return _WANDB_RUN

    _WANDB_INIT_INTENTADO = True
    if not os.getenv("WANDB_API_KEY"):
        return None

    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        import wandb

        _WANDB_RUN = wandb.init(
            project=os.getenv("WANDB_PROJECT", "personal-shopper-moda"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=os.getenv("WANDB_RUN_NAME", f"prod-hf-space-{uuid.uuid4().hex[:6]}"),
            job_type="production-inference",
            reinit=False,
        )
        atexit.register(lambda: _WANDB_RUN.finish() if _WANDB_RUN is not None else None)
    except Exception:
        _WANDB_RUN = None

    return _WANDB_RUN


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
    global _CONTADOR_PETICIONES
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

    run = _get_wandb_run()
    if run is not None:
        try:
            run.log({
                "inferencia/latencia_s": round(latencia_s, 3),
                "inferencia/latencia_ms": round(latencia_s * 1000.0, 1),
                "inferencia/uso_herramienta": int(uso_herramienta),
                "inferencia/valvula_activada": int(valvula_activada),
                "inferencia/cache_hit": int(cache_hit),
                "inferencia/tiene_imagen": int(tiene_imagen),
                "inferencia/es_moda": int(es_moda),
                "inferencia/num_outfits": len(outfits_recuperados),
                "inferencia/error": int(estado == "error"),
                "inferencia/peticiones_totales": _CONTADOR_PETICIONES,
            })
        except Exception:
            pass

    return evento


def obtener_resumen_metricas(ultimos_n: int = 100) -> Dict[str, Any]:
    """Lee los últimos N registros del log JSONL para el panel de observabilidad."""
    wandb_activo = bool(os.getenv("WANDB_API_KEY"))
    if not LOG_PATH.exists():
        return {
            "wandb_activo": wandb_activo,
            "total_peticiones": 0,
            "latencia_p50_ms": 0.0,
            "latencia_p95_ms": 0.0,
            "ultimos_logs": [],
        }

    with _LOCK:
        lineas = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()

    registros = []
    for ln in lineas[-ultimos_n:]:
        try:
            registros.append(json.loads(ln))
        except Exception:
            continue

    if not registros:
        return {
            "wandb_activo": wandb_activo,
            "total_peticiones": 0,
            "latencia_p50_ms": 0.0,
            "latencia_p95_ms": 0.0,
            "ultimos_logs": [],
        }

    lats = [r["latencia_ms"] for r in registros if "latencia_ms" in r]
    herr = [int(r.get("uso_herramienta", False)) for r in registros]
    valv = [int(r.get("valvula_activada", False)) for r in registros]

    return {
        "wandb_activo": wandb_activo,
        "wandb_proyecto": os.getenv("WANDB_PROJECT", "personal-shopper-moda") if wandb_activo else None,
        "total_peticiones": len(lineas),
        "latencia_p50_ms": round(float(np.percentile(lats, 50)), 1) if lats else 0.0,
        "latencia_p95_ms": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
        "tasa_herramienta_pct": round(100.0 * float(np.mean(herr)), 1),
        "tasa_valvula_pct": round(100.0 * float(np.mean(valv)), 1),
        "ultimos_logs": registros[-10:][::-1],
    }