"""
Parte 3 – Representación Vectorial del Texto
=============================================
Implementa tres representaciones:
  1. Bag of Words (BoW)
  2. TF-IDF
  3. Word Embeddings 768D  (modelo multilingüe en español/inglés)
     Modelo: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
             (768 dimensiones, soporta español e inglés)

Puntos extra incluidos:
  • Análisis de costos computacionales de cada representación
  • Persistencia de matrices en JSON y NPZ (numpy)

Entrada:  data/preprocessed.json  (de parte2)
          data/segments.json      (de parte1, para texto original)
Salidas:
  data/bow_matrix.npz
  data/tfidf_matrix.npz
  data/embeddings_matrix.npz
  data/vectorization_meta.json
  data/computational_costs.json
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PREPROCESSED_JSON = DATA_DIR / "preprocessed.json"
SEGMENTS_JSON = DATA_DIR / "segments.json"

BOW_NPZ = DATA_DIR / "bow_matrix.npz"
TFIDF_NPZ = DATA_DIR / "tfidf_matrix.npz"
EMBED_NPZ = DATA_DIR / "embeddings_matrix.npz"
META_JSON = DATA_DIR / "vectorization_meta.json"
COSTS_JSON = DATA_DIR / "computational_costs.json"


# ── Utilidades ────────────────────────────────────────────────────────────────

def sparsity(matrix: np.ndarray) -> float:
    """Proporción de ceros en la matriz."""
    total = matrix.size
    zeros = np.sum(matrix == 0)
    return zeros / total if total > 0 else 0.0


def memory_mb(matrix: np.ndarray) -> float:
    return matrix.nbytes / (1024 ** 2)


# ── 1. Bag of Words ──────────────────────────────────────────────────────────

def build_bow(corpus: list[str]) -> tuple[np.ndarray, CountVectorizer, float]:
    """
    corpus: lista de strings (un string por capítulo, ya preprocesado)
    Retorna: (matriz densa, vectorizador, tiempo_segundos)
    """
    t0 = time.perf_counter()
    vec = CountVectorizer()
    X = vec.fit_transform(corpus)
    elapsed = time.perf_counter() - t0
    return X.toarray(), vec, elapsed


# ── 2. TF-IDF ────────────────────────────────────────────────────────────────

def build_tfidf(corpus: list[str]) -> tuple[np.ndarray, TfidfVectorizer, float]:
    """
    Retorna: (matriz densa, vectorizador, tiempo_segundos)
    """
    t0 = time.perf_counter()
    vec = TfidfVectorizer()
    X = vec.fit_transform(corpus)
    elapsed = time.perf_counter() - t0
    return X.toarray(), vec, elapsed


# ── 3. Embeddings 768D (sentence-transformers) ───────────────────────────────

def build_embeddings(
    texts_original: list[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
) -> tuple[np.ndarray, float]:
    """
    Genera embeddings de 768D usando un modelo multilingüe
    (soporta inglés, español y 50+ idiomas).
    
    Usamos el texto ORIGINAL (no preprocesado) para que el transformer
    aproveche el contexto completo.

    Retorna: (matriz [n_chapters x 768], tiempo_segundos)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Instala sentence-transformers:\n"
            "  pip install sentence-transformers"
        )

    print(f"      Cargando modelo: {model_name}")
    model = SentenceTransformer(model_name)

    t0 = time.perf_counter()
    # encode devuelve np.ndarray de shape (n, 768)
    embeddings = model.encode(
        texts_original,
        show_progress_bar=True,
        batch_size=8,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    return embeddings, elapsed


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    t0_total = time.time()
    print("=" * 60)
    print("PARTE 3 – Representación Vectorial del Texto")
    print("=" * 60)

    # Cargar datos preprocesados
    if not PREPROCESSED_JSON.exists():
        raise FileNotFoundError(
            f"No se encontró {PREPROCESSED_JSON}. "
            "Ejecuta primero parte2_preprocesamiento.py"
        )
    if not SEGMENTS_JSON.exists():
        raise FileNotFoundError(
            f"No se encontró {SEGMENTS_JSON}. "
            "Ejecuta primero parte1_ingesta.py"
        )

    with open(PREPROCESSED_JSON, encoding="utf-8") as f:
        preprocessed = json.load(f)
    with open(SEGMENTS_JSON, encoding="utf-8") as f:
        segments = json.load(f)

    n_chapters = len(preprocessed)
    print(f"[INFO] Capítulos a vectorizar: {n_chapters}")

    # Corpus preprocesado (tokens unidos en string)
    corpus_clean = [
        " ".join(ch["all_tokens"]) for ch in preprocessed
    ]

    # Corpus original (párrafos unidos, para embeddings)
    corpus_original = [
        " ".join(seg["paragraphs"]) for seg in segments
    ]

    titles = [ch["title"] for ch in preprocessed]

    costs = {}

    # ── BoW ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Construyendo Bag of Words …")
    bow_matrix, bow_vec, t_bow = build_bow(corpus_clean)
    np.savez_compressed(BOW_NPZ, matrix=bow_matrix)
    costs["bow"] = {
        "time_s": round(t_bow, 6),
        "shape": list(bow_matrix.shape),
        "vocab_size": len(bow_vec.vocabulary_),
        "sparsity": round(sparsity(bow_matrix), 6),
        "memory_mb": round(memory_mb(bow_matrix), 4),
        "interpretable": True,
        "captures_frequency": True,
        "captures_semantics": False,
    }
    print(f"      Shape: {bow_matrix.shape}  |  Sparsity: {sparsity(bow_matrix):.4%}")
    print(f"      Vocab: {len(bow_vec.vocabulary_):,}  |  Tiempo: {t_bow:.4f}s")
    print(f"      → Guardado en {BOW_NPZ}")

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    print("\n[2/3] Construyendo TF-IDF …")
    tfidf_matrix, tfidf_vec, t_tfidf = build_tfidf(corpus_clean)
    np.savez_compressed(TFIDF_NPZ, matrix=tfidf_matrix)
    costs["tfidf"] = {
        "time_s": round(t_tfidf, 6),
        "shape": list(tfidf_matrix.shape),
        "vocab_size": len(tfidf_vec.vocabulary_),
        "sparsity": round(sparsity(tfidf_matrix), 6),
        "memory_mb": round(memory_mb(tfidf_matrix), 4),
        "interpretable": True,
        "captures_frequency": True,
        "captures_semantics": False,
    }
    print(f"      Shape: {tfidf_matrix.shape}  |  Sparsity: {sparsity(tfidf_matrix):.4%}")
    print(f"      Vocab: {len(tfidf_vec.vocabulary_):,}  |  Tiempo: {t_tfidf:.4f}s")
    print(f"      → Guardado en {TFIDF_NPZ}")

    # ── Embeddings 768D ──────────────────────────────────────────────────────
    print("\n[3/3] Generando Embeddings 768D (multilingüe español/inglés) …")
    print("      Modelo: paraphrase-multilingual-mpnet-base-v2")
    embeddings, t_embed = build_embeddings(corpus_original)
    np.savez_compressed(EMBED_NPZ, matrix=embeddings)
    costs["embeddings_768d"] = {
        "time_s": round(t_embed, 6),
        "shape": list(embeddings.shape),
        "vocab_size": "N/A (denso)",
        "sparsity": round(sparsity(embeddings), 6),
        "memory_mb": round(memory_mb(embeddings), 4),
        "interpretable": False,
        "captures_frequency": False,
        "captures_semantics": True,
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "dimensions": 768,
        "language_support": "50+ idiomas (incluye español e inglés)",
    }
    print(f"      Shape: {embeddings.shape}  |  Sparsity: {sparsity(embeddings):.4%}")
    print(f"      Tiempo: {t_embed:.4f}s")
    print(f"      → Guardado en {EMBED_NPZ}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "titles": titles,
        "n_chapters": n_chapters,
        "bow": {
            "npz_file": str(BOW_NPZ),
            "shape": list(bow_matrix.shape),
            "feature_names_sample": bow_vec.get_feature_names_out()[:20].tolist(),
        },
        "tfidf": {
            "npz_file": str(TFIDF_NPZ),
            "shape": list(tfidf_matrix.shape),
            "feature_names_sample": tfidf_vec.get_feature_names_out()[:20].tolist(),
        },
        "embeddings": {
            "npz_file": str(EMBED_NPZ),
            "shape": list(embeddings.shape),
            "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        },
    }

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n      → Metadata guardada en {META_JSON}")

    with open(COSTS_JSON, "w", encoding="utf-8") as f:
        json.dump(costs, f, ensure_ascii=False, indent=2)
    print(f"      → Costos computacionales guardados en {COSTS_JSON}")

    # ── Tabla comparativa ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA DE PROPIEDADES")
    print("=" * 60)
    headers = ["Métrica", "BoW", "TF-IDF", "Embeddings 768D"]
    rows = [
        ["Dimensiones", str(bow_matrix.shape[1]), str(tfidf_matrix.shape[1]), "768"],
        ["Sparsity",
         f"{sparsity(bow_matrix):.2%}",
         f"{sparsity(tfidf_matrix):.2%}",
         f"{sparsity(embeddings):.2%}"],
        ["Memoria (MB)",
         f"{memory_mb(bow_matrix):.3f}",
         f"{memory_mb(tfidf_matrix):.3f}",
         f"{memory_mb(embeddings):.3f}"],
        ["Tiempo (s)",
         f"{t_bow:.4f}",
         f"{t_tfidf:.4f}",
         f"{t_embed:.4f}"],
        ["Semántica", "No", "No", "Sí"],
        ["Interpretable", "Sí", "Sí", "No"],
    ]
    col_w = [18, 14, 14, 18]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("-" * sum(col_w))
    for row in rows:
        print("".join(c.ljust(w) for c, w in zip(row, col_w)))

    elapsed = time.time() - t0_total
    print(f"\n✓ Parte 3 completada en {elapsed:.2f}s")
    print("=" * 60)

    return {
        "bow": bow_matrix,
        "tfidf": tfidf_matrix,
        "embeddings": embeddings,
        "titles": titles,
        "bow_vec": bow_vec,
        "tfidf_vec": tfidf_vec,
    }


if __name__ == "__main__":
    main()
