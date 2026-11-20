# -*- coding: utf-8 -*-
"""
observabilidad.py — Logging estructurado + observabilidad en MLflow para el sistema desplegado.

Requisitos del proyecto final (Módulo 5):
  - Logging estructurado: cada petición deja un registro JSON con, al menos, hora, latencia,
    qué componentes se usaron (RAG, herramienta, visión) y si se activó la válvula.
    SIN datos personales (no se guarda el texto de la pregunta, solo su longitud y un hash corto).
  - Observabilidad en MLflow: cada petición se registra como una métrica en una run de MLflow,
    con latencia por petición y métricas de uso (peticiones, herramienta, válvula, errores).
    SOLO se activa si existe MLFLOW_TRACKING_URI; si no, el sistema sigue funcionando y solo
    escribe el log JSON local. Nunca se envía a MLflow el texto de la pregunta ni errores crudos.

Variables de entorno:
  - MLFLOW_TRACKING_URI    : URI del servidor de MLflow (requerido para activar el backend).
  - MLFLOW_EXPERIMENT_NAME : nombre del experimento (def. 'personal-shopper-moda').
  - MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD / MLFLOW_TRACKING_TOKEN:
    credenciales del servidor (las lee directamente la librería MLflow).
  - RAG_LOG_PATH           : ruta del JSONL local (def. 'final/logs/peticiones.jsonl').

Uso:
    obs = Observador()          # MLflow si hay MLFLOW_TRACKING_URI, si no solo log local
    obs.registrar(evento)       # escribe JSON + sube métricas a MLflow
    obs.cerrar()                # cierra la run al apagar el servicio
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone


LOG_PATH = os.environ.get("RAG_LOG_PATH", "final/logs/peticiones.jsonl")
EXPERIMENT_DEFAULT = "personal-shopper-moda"
SERVICIO = "personal-shopper-moda"


def _hash_corto(texto: str) -> str:
    """Hash no reversible y corto de la pregunta, para agrupar sin guardar el contenido."""
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()[:12]


def _float(evento, clave, default=0.0):
    """Convierte un campo del evento a float sin dejarnos caer por un valor inesperado."""
    valor = evento.get(clave, default)
    try:
        return float(valor) if valor is not None else default
    except (TypeError, ValueError):
        return default


def _nombre_metrica(clave):
    """Nombre válido de métrica MLflow para una latencia por etapa."""
    limpio = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(clave)).strip("_")
    return "latencia_etapa_" + (limpio or "otra")


def construir_evento(pregunta, salida, modo, con_imagen=False):
    """Arma el registro estructurado de una petición. NO incluye el texto de la pregunta
    (solo su longitud y un hash), para no guardar datos personales."""
    componentes = []
    if modo in ("ingenuo", "avanzado", "final"):
        componentes.append("RAG")
    if modo == "avanzado" or modo == "final":
        componentes.append("hibrida+HyDE")
    if salida.get("uso_herramienta"):
        componentes.append("herramienta")
    if con_imagen:
        componentes.append("vision")

    valvula = bool(salida.get("valvula")) if "valvula" in salida else None
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "modo": modo,
        "latencia_s": round(float(salida.get("latencia_s", 0.0)), 3),
        "componentes": componentes,
        "uso_herramienta": bool(salida.get("uso_herramienta", False)),
        "filtro_campos": sorted(list((salida.get("filtro") or {}).keys())),
        "valvula_activada": valvula,
        "n_outfits": len(salida.get("outfits", [])),
        "con_imagen": bool(con_imagen),
        "pregunta_len": len((pregunta or "")),
        "pregunta_hash": _hash_corto(pregunta),
    }


class Observador:
    """Escribe logs JSON por petición y, si hay MLflow configurado, también sube métricas."""

    def __init__(self, mlflow_tracking_uri=None, mlflow_experiment=None, log_path=LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.n = 0
        self._mlflow = None
        self.mlflow_activo = False
        self._lat_acumulada = 0.0
        self._valvulas = 0
        self._herramienta = 0
        self._errores = 0
        self._errores_mlflow = 0

        uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        experimento = (mlflow_experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                       or EXPERIMENT_DEFAULT)
        if uri:
            self._mlflow = self._iniciar_mlflow(uri, experimento)
            self.mlflow_activo = self._mlflow is not None
        else:
            print("[obs] MLflow desactivado (sin MLFLOW_TRACKING_URI); solo log local JSON.")

    def _iniciar_mlflow(self, uri, experimento):
        """Abre una run de serving. Si falla, el sistema sigue con el log local solamente."""
        try:
            import mlflow
        except ImportError as e:
            print("[obs] MLflow no instalado (" + str(e) + "); solo log local JSON.")
            return None
        try:
            mlflow.set_tracking_uri(uri)
            if experimento:
                mlflow.set_experiment(experimento)
            nombre = "servicio-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            mlflow.start_run(run_name=nombre)
            mlflow.set_tags({"servicio": SERVICIO, "tipo": "serving",
                             "experimento": experimento, "log_local": self.log_path})
            print("[obs] MLflow activo:", uri, "| experimento:", experimento, "| run:", nombre)
            return mlflow
        except Exception as e:
            print("[obs] MLflow no disponible (" + str(e) + "); solo log local JSON.")
            return None

    def registrar(self, evento):
        """Escribe el evento como una línea JSON y, si hay MLflow, lo sube como métrica."""
        self.n += 1
        latencia = _float(evento, "latencia_s")
        self._lat_acumulada += latencia
        if evento.get("valvula_activada"):
            self._valvulas += 1
        if evento.get("uso_herramienta"):
            self._herramienta += 1
        if evento.get("error"):
            self._errores += 1
        if not evento.get("ts"):
            evento["ts"] = datetime.now(timezone.utc).isoformat()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")

        if self._mlflow is not None:
            self._registrar_en_mlflow(evento, latencia)

    def _registrar_en_mlflow(self, evento, latencia):
        """Sube SOLO métricas/etiquetas agregadas (nunca texto de la pregunta ni errores crudos)."""
        componentes = evento.get("componentes") or []
        metricas = {
            "latencia_s": latencia,
            "peticiones_totales": float(self.n),
            "latencia_media_s": round(self._lat_acumulada / self.n, 3),
            "tasa_herramienta": round(self._herramienta / self.n, 3),
            "tasa_valvula": round(self._valvulas / self.n, 3),
            "tasa_error": round(self._errores / self.n, 3),
            "n_outfits": _float(evento, "n_outfits"),
            "uso_herramienta": _float(evento, "uso_herramienta"),
            "valvula_activada": _float(evento, "valvula_activada"),
            "con_imagen": _float(evento, "con_imagen"),
            "rag_usado": float("RAG" in componentes),
            "vision_usado": float("vision" in componentes),
        }
        if "latencia_total_s" in evento:
            metricas["latencia_total_s"] = _float(evento, "latencia_total_s")
        etapas = evento.get("latencias_etapa")
        if isinstance(etapas, dict):
            for clave, valor in etapas.items():
                try:
                    metricas[_nombre_metrica(clave)] = float(valor)
                except (TypeError, ValueError):
                    continue
        etiquetas = {
            "ultimo_modo": str(evento.get("modo") or "desconocido"),
            "componentes": ",".join(str(c) for c in componentes) or "ninguno",
        }
        try:
            self._mlflow.log_metrics(metricas, step=self.n)
            self._mlflow.set_tags(etiquetas)
        except Exception as e:
            self._errores_mlflow += 1
            if self._errores_mlflow <= 3:
                print("[obs] fallo al subir a MLflow:", e)

    def resumen(self):
        return {
            "peticiones": self.n,
            "latencia_media_s": round(self._lat_acumulada / self.n, 3) if self.n else 0.0,
            "valvulas": self._valvulas,
            "herramienta_usada": self._herramienta,
            "errores": self._errores,
            "mlflow_activo": self.mlflow_activo,
        }

    def cerrar(self):
        """Cierra la run de MLflow (si la hay) para que la métrica final quede persistida."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.end_run()
        except Exception as e:
            print("[obs] fallo al cerrar la run de MLflow:", e)
        finally:
            self._mlflow = None
            self.mlflow_activo = False
