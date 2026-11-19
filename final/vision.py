# -*- coding: utf-8 -*-
"""
vision.py — Componente visual (Módulo 4) del personal shopper.

Patrón visual: el usuario sube una FOTO de una prenda/outfit y el sistema la "lee" para
convertirla en una consulta de texto que alimenta el RAG. Así el cliente puede decir
"quiero algo parecido a esto" con una imagen en vez de con palabras.

Implementación: image captioning con BLIP (`Salesforce/blip-image-captioning-base`), un modelo
ligero que corre en Colab/Spaces gratuito. El caption (en inglés) se traduce a una consulta de
moda en español mediante un mapeo de términos frecuentes (sin depender de una API de traducción).

Degradación con gracia: si el modelo de visión no puede cargarse (poca memoria en Spaces), la
función lo indica y el sistema sigue operando solo con texto.

IMPORTANTE (ética / requisito del enunciado): una imagen GENERADA se marca como generada. Este
componente solo DESCRIBE imágenes que sube el usuario; no genera imágenes.
"""

from __future__ import annotations

# Glosario mínimo EN->ES de términos de moda para convertir el caption de BLIP en consulta.
_GLOSARIO = {
    "jacket": "chaqueta", "coat": "abrigo", "sweater": "suéter", "hoodie": "sudadera con capucha",
    "shirt": "camisa", "t-shirt": "camiseta", "tshirt": "camiseta", "dress": "vestido",
    "skirt": "falda", "pants": "pantalones", "trousers": "pantalones", "jeans": "vaqueros",
    "shorts": "pantalones cortos", "coat": "abrigo", "suit": "traje", "blazer": "blazer",
    "boots": "botas", "shoes": "zapatos", "sneakers": "zapatillas", "scarf": "bufanda",
    "hat": "sombrero", "bag": "bolso", "vest": "chaleco", "cardigan": "cárdigan",
    "black": "negro", "white": "blanco", "blue": "azul", "red": "rojo", "green": "verde",
    "brown": "marrón", "gray": "gris", "grey": "gris", "beige": "beige", "pink": "rosa",
    "navy": "azul marino", "plaid": "de cuadros", "striped": "de rayas", "denim": "de mezclilla",
    "leather": "de cuero", "wool": "de lana", "woman": "mujer", "man": "hombre",
    "wearing": "con", "person": "", "a": "", "the": "", "in": "", "and": "y", "with": "con",
}


class VisionCaptioner:
    """Describe imágenes de moda con BLIP y genera una consulta de texto en español."""

    def __init__(self, device=None, model_name="Salesforce/blip-image-captioning-base",
                 verbose=True):
        self.verbose = verbose
        self.model_name = model_name
        self.ok = False
        self._device = device
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            if self._device is None:
                try:
                    import torch
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self._device = "cpu"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self._device)
            self.model.eval()
            self.ok = True
            self._log("Componente visual BLIP cargado en " + str(self._device) + ".")
        except Exception as e:
            self._log("Componente visual NO disponible (" + str(e) + "). Se opera solo con texto.")

    def _log(self, *a):
        if self.verbose:
            print("[vision]", *a)

    def describir(self, imagen):
        """imagen: PIL.Image. Devuelve el caption en inglés (o None si no hay modelo)."""
        if not self.ok or imagen is None:
            return None
        import torch
        inputs = self.processor(imagen, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def caption_a_consulta(self, caption):
        """Convierte el caption en inglés en una consulta de moda en español (mapeo de términos)."""
        if not caption:
            return None
        palabras = []
        for w in caption.lower().replace(",", " ").split():
            palabras.append(_GLOSARIO.get(w, w))
        consulta = " ".join(p for p in palabras if p).strip()
        return "Busco un outfit parecido a: " + consulta

    def imagen_a_consulta(self, imagen):
        """Atajo: imagen -> (caption_en, consulta_es). (None, None) si no hay modelo."""
        cap = self.describir(imagen)
        if cap is None:
            return None, None
        return cap, self.caption_a_consulta(cap)
