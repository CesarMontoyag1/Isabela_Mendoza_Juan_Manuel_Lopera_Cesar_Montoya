# Documentación del Dataset de Entrenamiento: Perfiles de Estilo E-commerce

## 1. Fuente
El texto de este dataset es de naturaleza sintética, generado automatizadamente a través de la API de Google Gemini. 
* **Proceso de recolección:** Se construyó un pipeline en Python (`generate_synthetic_dataset.py`) que genera combinaciones aleatorias de atributos de una ontología de moda predefinida. Estas combinaciones se enviaron al modelo `gemini-3.5-flash-lite` utilizando técnicas de prompt engineering para generar diversas formas naturales y coloquiales de solicitar dichos perfiles a través de un chat de e-commerce.

## 2. Tamaño y Partición (Split)
* **Volumen Total:** Se generaron 260 combinaciones únicas de perfiles, con 5 parafraseos por cada combinación, resultando en un máximo teórico de 1,300 ejemplos.
* **Partición (Split):** Se utilizó una división automatizada del 80/20.
  * **Train (`train.jsonl`):** 80% de los datos generados.
  * **Validation (`val.jsonl`):** 20% de los datos generados.

## 3. Idioma(s) y Licencia
* **Idioma:** Español. El dataset incluye intencionalmente lenguaje natural de chat, modismos colombianos ocasionales y errores de tipeo leves para simular la interacción real de usuarios en un e-commerce local.
* **Licencia:** Datos de uso académico generados para el curso SI4006. Sin restricciones de privacidad al ser datos sintéticos.

## 4. Tarea Principal
Este dataset está diseñado para una tarea de **Extracción de Información (Information Extraction)** y formateo estructurado.
* **Input (Entrada):** Un mensaje de texto en lenguaje natural simulando un cliente escribiéndole a un "personal shopper".
* **Output Esperado (Salida):** Un objeto JSON estricto que mapea el texto del usuario a una ontología de 5 campos: `estilo`, `ocasion`, `clima`, `paleta`, y `fit`. El modelo debe inferir los atributos a partir del contexto si no se mencionan explícitamente.

## 5. Sesgos o Limitaciones Conocidas
* **Sesgo dialectal:** Al instruir al modelo generador para usar modismos colombianos, el modelo afinado podría tener un menor rendimiento (menor *accuracy* de extracción) si se enfrenta a jergas o dialectos de otros países hispanohablantes.
* **Limitación de Dominio (Out-of-Domain):** El dataset está estrictamente limitado a 5 variables de moda. Si un usuario introduce consultas sobre presupuesto, tallas exactas numéricas, o envíos, el modelo probablemente fallará al intentar forzar esos datos dentro de la estructura JSON predefinida.
* **Dependencia de la Ontología:** El modelo asume que solo existen las categorías definidas (ej. en `clima` solo existe `calido`, `frio`, `templado`). Consultas ambiguas o intermedias podrían generar alucinaciones en los valores de las llaves JSON.
