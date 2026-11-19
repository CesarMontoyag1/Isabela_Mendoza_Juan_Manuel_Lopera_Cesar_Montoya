# -*- coding: utf-8 -*-
"""
observabilidad.py — Logging estructurado + observabilidad en W&B para el sistema desplegado.

Requisitos del proyecto final (Módulo 5):
  - Logging estructurado: cada petición deja un registro JSON con, al menos, hora, latencia,
    qué componentes se usaron (RAG, herramienta, visión) y si se activó la válvula.
    SIN datos personales (no se guarda el texto de la pregunta, solo su longitud y un hash corto).
  - Observabilidad en W&B: cada petición se registra como una fila/métrica en un proyecto de W&B,
    con latencia por petición y al menos una métrica de uso. W&B es OPCIONAL: si no hay API key,
    el sistema sigue funcionando y solo escribe el log JSON local.

Uso:
    obs = Observador(wandb_project="personal-shopper-moda")   # intenta W&B si hay WANDB_API_KEY
    obs.registrar(evento)                                     # escribe JSON + sube a W&B
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone


LOG_PATH = os.environ.get("RAG_LOG_PATH", "final/logs/peticiones.jsonl")


def _hash_corto(texto: str) -> str:
    """Hash no reversible y corto de la pregunta, para agrupar sin guardar el contenido."""
    return hashlib.sha256((texto or "").encode("utf-8")).hexdigest()[:12]


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
    """Escribe logs JSON por petición y, si hay W&B configurado, también sube métricas."""

    def __init__(self, wandb_project=None, wandb_entity=None, log_path=LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.n = 0
        self._wandb = None
        self._lat_acumulada = 0.0
        self._valvulas = 0
        self._herramienta = 0
        if wandb_project and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                self._wandb = wandb.init(project=wandb_project, entity=wandb_entity,
                                         job_type="serving", reinit=True,
                                         config={"servicio": "personal-shopper-moda"})
                print("[obs] W&B activo:", wandb_project)
            except Exception as e:
                print("[obs] W&B no disponible (" + str(e) + "); solo log local.")
        else:
            print("[obs] W&B desactivado (sin WANDB_API_KEY); solo log local JSON.")

    def registrar(self, evento):
        """Escribe el evento como una línea JSON y, si hay W&B, lo sube como métrica."""
        self.n += 1
        self._lat_acumulada += evento.get("latencia_s", 0.0)
        if evento.get("valvula_activada"):
            self._valvulas += 1
        if evento.get("uso_herramienta"):
            self._herramienta += 1

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")

        if self._wandb is not None:
            try:
                self._wandb.log({
                    "latencia_s": evento["latencia_s"],
                    "n_outfits": evento["n_outfits"],
                    "uso_herramienta": int(evento["uso_herramienta"]),
                    "valvula_activada": int(bool(evento.get("valvula_activada"))),
                    "con_imagen": int(evento["con_imagen"]),
                    # métricas de uso acumuladas
                    "peticiones_totales": self.n,
                    "latencia_media_s": round(self._lat_acumulada / self.n, 3),
                    "tasa_herramienta": round(self._herramienta / self.n, 3),
                })
            except Exception as e:
                print("[obs] fallo al subir a W&B:", e)

    def resumen(self):
        return {
            "peticiones": self.n,
            "latencia_media_s": round(self._lat_acumulada / self.n, 3) if self.n else 0.0,
            "valvulas": self._valvulas,
            "herramienta_usada": self._herramienta,
        }

    def cerrar(self):
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
