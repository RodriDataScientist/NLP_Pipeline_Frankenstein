# 📖 NLP Pipeline — *Frankenstein* (Mary Shelley)

> Práctica E1 · Procesamiento y Representación de Texto  
> Ingeniería de Datos e IA · Mayo 2026

Pipeline reproducible de NLP aplicado a *Frankenstein* (Project Gutenberg) que abarca ingesta, preprocesamiento, representación vectorial y análisis semántico comparativo.

---

## 📋 Tabla de contenido

- [Descripción](#descripción)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Puntos extra implementados](#puntos-extra-implementados)
- [Dataset](#dataset)

---

## Descripción

Se implementa un pipeline de cuatro etapas sobre la novela *Frankenstein*:

| Parte | Script | Descripción |
|-------|--------|-------------|
| 1 | `parte1_ingesta.py` | Lectura del HTML, limpieza de cabecera/pie Gutenberg, segmentación en cartas, capítulos, párrafos y oraciones |
| 2 | `parte2_preprocesamiento.py` | Tokenización, minusculización, eliminación de stopwords/puntuación/números, lematización (NLTK) |
| 3 | `parte3_representaciones.py` | Bag of Words, TF-IDF y embeddings densos de 768D (modelo multilingüe español/inglés) |
| 4 | `parte4_analisis.py` | Similitud coseno, clustering K-Means, PCA, t-SNE, análisis iniciales vs. finales, SQLite |

---

## Estructura del proyecto

```
.
├── data/
│   ├── frankenstein.html          # Libro original (Project Gutenberg)
│   ├── frankenstein_clean.txt     # Texto limpio sin cabecera/pie
│   ├── segments.json              # Capítulos → párrafos + oraciones
│   ├── preprocessed.json          # Tokens limpios por capítulo
│   ├── lemma_vs_stem.json         # Comparativa morfológica por capítulo
│   ├── bow_matrix.npz             # Matriz BoW comprimida (28 × 6371)
│   ├── tfidf_matrix.npz           # Matriz TF-IDF comprimida (28 × 6371)
│   ├── embeddings_matrix.npz      # Embeddings 768D (28 × 768)
│   ├── vectorization_meta.json    # Metadatos de representaciones
│   ├── computational_costs.json   # Tiempos y memoria por método
│   ├── analysis_results.json      # Similitudes, clusters, ini vs. fin
│   └── results.db                 # SQLite: similitudes + clusters + costos
├── parte1_ingesta.py
├── parte2_preprocesamiento.py
├── parte3_representaciones.py
├── parte4_analisis.py
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.10+
- Las dependencias están listadas en `requirements.txt`

Paquetes principales:

```
nltk
beautifulsoup4
scikit-learn
sentence-transformers
numpy
matplotlib
seaborn
torch
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repo>.git
cd <repo>

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el archivo del libro
#    Descarga el HTML desde https://www.gutenberg.org/ebooks/84
#    y guárdalo como:
mkdir -p data
mv frankenstein.html data/
```

---

## Uso

Ejecutar los scripts **en orden** desde la raíz del proyecto:

```bash
python parte1_ingesta.py
python parte2_preprocesamiento.py
python parte3_representaciones.py
python parte4_analisis.py
```

Cada script es independiente y lee los artefactos generados por el anterior. Al finalizar, la carpeta `data/` contendrá todos los archivos procesados y la carpeta `figures/` las visualizaciones.

> **Nota:** La primera ejecución de `parte3_representaciones.py` descarga el modelo `paraphrase-multilingual-mpnet-base-v2` (~1 GB). Las ejecuciones posteriores usan la caché local.

---

## Resultados

### Corpus

- **28 secciones**: 4 cartas + 24 capítulos
- **Vocabulario preprocesado**: 6 371 términos únicos
- **Tokens totales** (sin stopwords): ~31 000

### Comparativa de representaciones

| Propiedad | BoW | TF-IDF | Embeddings 768D |
|-----------|-----|--------|-----------------|
| Dimensiones | 6 371 | 6 371 | 768 |
| Esparsidad | 87.9 % | 87.9 % | 0.0 % |
| Memoria | 1.361 MB | 1.361 MB | 0.082 MB |
| Tiempo construcción | 0.022 s | 0.018 s | 0.756 s |
| Captura semántica | ✗ | ✗ | ✓ |
| Interpretable | ✓ | ✓ | ✗ |

### Top similitudes coseno (embeddings)

| Par | Similitud |
|-----|-----------|
| Letter 1 ↔ Letter 3 | 0.800 |
| Letter 1 ↔ Letter 2 | 0.749 |
| Chapter 5 ↔ Chapter 20 | 0.736 |

### Análisis iniciales vs. finales

|  | BoW | TF-IDF | Embeddings |
|--|-----|--------|------------|
| Cohesión intra-inicial (Letters 1–3) | 0.281 | 0.156 | **0.766** |
| Cohesión intra-final (Ch. 22–24) | 0.513 | 0.329 | 0.476 |
| Similitud inter (prólogo ↔ desenlace) | 0.288 | 0.147 | 0.415 |

---

## Puntos extra implementados

- [x] **Comparación capítulos iniciales vs. finales** — similitud coseno intra e inter grupo
- [x] **Lematización vs. Stemming** — comparación de vocabulario y tiempo entre WordNet, Porter y Snowball por capítulo
- [x] **Persistencia en JSON y SQLite** — todos los artefactos guardados; base de datos con tablas `cosine_similarities`, `chapter_clusters` y `computational_costs`
- [x] **Costos computacionales** — medición de tiempo y memoria para cada representación

---

## Dataset

**Fuente:** [Project Gutenberg — Frankenstein (eBook #84)](https://www.gutenberg.org/ebooks/84)  
**Autora:** Mary Wollstonecraft Shelley  
**Formato descargado:** HTML  
**Licencia del texto:** Dominio público (Project Gutenberg License)
