"""
Parte 2 – Preprocesamiento con NLTK
=====================================
Pipeline reutilizable de limpieza y normalización.
Incluye puntos extra:
  • Comparación lematización vs stemming
  • Medición de costos computacionales

Entrada:  data/segments.json  (generado por parte1_ingesta.py)
Salidas:
  data/preprocessed.json
  data/lemma_vs_stem.json
"""

import json
import re
import string
import time
from pathlib import Path

import nltk

# Descargas necesarias
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
SEGMENTS_JSON = DATA_DIR / "segments.json"
PREPROCESSED_JSON = DATA_DIR / "preprocessed.json"
LEMMA_VS_STEM_JSON = DATA_DIR / "lemma_vs_stem.json"

# ── Recursos NLTK ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
PORTER = PorterStemmer()
SNOWBALL = SnowballStemmer("english")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ── Funciones de preprocesamiento ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenización a nivel de palabra con NLTK."""
    return word_tokenize(text)


def to_lowercase(tokens: list[str]) -> list[str]:
    """Convierte todos los tokens a minúsculas."""
    return [t.lower() for t in tokens]


def remove_punctuation(tokens: list[str]) -> list[str]:
    """Elimina puntuación y tokens vacíos resultantes."""
    cleaned = [t.translate(PUNCT_TABLE) for t in tokens]
    return [t for t in cleaned if t]


def remove_numbers(tokens: list[str]) -> list[str]:
    """Elimina tokens que sean exclusivamente numéricos."""
    return [t for t in tokens if not t.isnumeric()]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Elimina stopwords del inglés."""
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lematización con WordNetLemmatizer (NLTK)."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def stem_porter(tokens: list[str]) -> list[str]:
    """Stemming con Porter Stemmer."""
    return [PORTER.stem(t) for t in tokens]


def stem_snowball(tokens: list[str]) -> list[str]:
    """Stemming con Snowball (más agresivo que Porter)."""
    return [SNOWBALL.stem(t) for t in tokens]


# ── Pipeline principal ────────────────────────────────────────────────────────

def preprocess_text(text: str, use_stemming: bool = False) -> dict:
    """
    Aplica el pipeline completo a un texto.
    Retorna tokens en cada etapa para poder mostrar comparaciones.
    """
    steps = {}
    steps["original"] = text

    tokens = tokenize(text)
    steps["after_tokenize"] = tokens

    tokens = to_lowercase(tokens)
    steps["after_lowercase"] = tokens

    tokens = remove_punctuation(tokens)
    steps["after_remove_punct"] = tokens

    tokens = remove_numbers(tokens)
    steps["after_remove_numbers"] = tokens

    tokens = remove_stopwords(tokens)
    steps["after_remove_stopwords"] = tokens

    if use_stemming:
        tokens = stem_porter(tokens)
        steps["after_stemming_porter"] = tokens
    else:
        tokens = lemmatize(tokens)
        steps["after_lemmatization"] = tokens

    steps["final_tokens"] = tokens
    return steps


def preprocess_chapter(chapter: dict) -> dict:
    """Preprocesa todos los párrafos de un capítulo y devuelve versión limpia."""
    processed_paragraphs = []
    for para in chapter["paragraphs"]:
        result = preprocess_text(para)
        processed_paragraphs.append(result["final_tokens"])

    # Tokens finales del capítulo completo (todos los párrafos unidos)
    full_text = " ".join(chapter["paragraphs"])
    full_result = preprocess_text(full_text)

    return {
        "chapter_idx": chapter["chapter_idx"],
        "title": chapter["title"],
        "processed_paragraphs": processed_paragraphs,
        "all_tokens": full_result["final_tokens"],
        "vocab_size": len(set(full_result["final_tokens"])),
        "token_count": len(full_result["final_tokens"]),
    }


# ── Comparación lemma vs stem (punto extra) ───────────────────────────────────

def compare_lemma_vs_stem(chapters: list[dict]) -> list[dict]:
    """
    Para cada capítulo mide:
      - Tiempo de lematización vs stemming Porter vs stemming Snowball
      - Vocabulario resultante de cada método
      - Ejemplos de diferencias
    """
    results = []
    for ch in chapters:
        text = " ".join(ch["paragraphs"])
        tokens_base = remove_stopwords(
            remove_numbers(
                remove_punctuation(
                    to_lowercase(tokenize(text))
                )
            )
        )

        # Lematización
        t0 = time.perf_counter()
        lemmas = lemmatize(tokens_base)
        t_lemma = time.perf_counter() - t0

        # Stemming Porter
        t0 = time.perf_counter()
        stems_p = stem_porter(tokens_base)
        t_porter = time.perf_counter() - t0

        # Stemming Snowball
        t0 = time.perf_counter()
        stems_s = stem_snowball(tokens_base)
        t_snowball = time.perf_counter() - t0

        # Ejemplos donde difieren (primeros 10)
        diffs = []
        for orig, lem, stp, sts in zip(tokens_base, lemmas, stems_p, stems_s):
            if lem != stp or lem != sts:
                diffs.append(
                    {"original": orig, "lemma": lem, "porter": stp, "snowball": sts}
                )
            if len(diffs) >= 10:
                break

        results.append(
            {
                "chapter_idx": ch["chapter_idx"],
                "title": ch["title"],
                "token_count": len(tokens_base),
                "vocab_lemma": len(set(lemmas)),
                "vocab_porter": len(set(stems_p)),
                "vocab_snowball": len(set(stems_s)),
                "time_lemma_s": round(t_lemma, 6),
                "time_porter_s": round(t_porter, 6),
                "time_snowball_s": round(t_snowball, 6),
                "sample_differences": diffs,
            }
        )
    return results


# ── Ejecución ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("PARTE 2 – Preprocesamiento con NLTK")
    print("=" * 60)

    if not SEGMENTS_JSON.exists():
        raise FileNotFoundError(
            f"No se encontró {SEGMENTS_JSON}. Ejecuta primero parte1_ingesta.py"
        )

    with open(SEGMENTS_JSON, encoding="utf-8") as f:
        chapters = json.load(f)
    print(f"[1/4] Cargados {len(chapters)} capítulos desde {SEGMENTS_JSON}")

    # Preprocesar todos los capítulos
    print("[2/4] Aplicando pipeline de preprocesamiento …")
    processed = []
    for ch in chapters:
        processed.append(preprocess_chapter(ch))
        print(f"      ✓ {ch['title'][:50]:<50}  tokens={processed[-1]['token_count']:>5}")

    with open(PREPROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"      → Guardado en {PREPROCESSED_JSON}")

    # Comparación lemma vs stem (punto extra)
    print("[3/4] Comparando lematización vs stemming (punto extra) …")
    comparison = compare_lemma_vs_stem(chapters)
    with open(LEMMA_VS_STEM_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"      → Guardado en {LEMMA_VS_STEM_JSON}")

    # Mostrar ejemplo antes/después
    print("\n[4/4] Ejemplo antes/después (primer capítulo, primer párrafo)")
    print("-" * 60)
    sample_para = chapters[0]["paragraphs"][0] if chapters[0]["paragraphs"] else ""
    steps = preprocess_text(sample_para)

    print(f"ORIGINAL:\n  {steps['original'][:300]}\n")
    print(f"DESPUÉS DE TOKENIZAR:\n  {steps['after_tokenize'][:15]} …\n")
    print(f"DESPUÉS DE MINÚSCULAS:\n  {steps['after_lowercase'][:15]} …\n")
    print(f"DESPUÉS DE ELIMINAR PUNTUACIÓN:\n  {steps['after_remove_punct'][:15]} …\n")
    print(f"DESPUÉS DE ELIMINAR NÚMEROS:\n  {steps['after_remove_numbers'][:15]} …\n")
    print(f"DESPUÉS DE ELIMINAR STOPWORDS:\n  {steps['after_remove_stopwords'][:15]} …\n")
    print(f"DESPUÉS DE LEMATIZACIÓN:\n  {steps['after_lemmatization'][:15]} …\n")

    print("-" * 60)
    print("Resumen comparación lemma vs stem (totales por capítulo):")
    print(f"{'Capítulo':<35} {'Tokens':>7} {'VocLema':>8} {'VocPort':>8} {'VocSnow':>8} {'tLema(s)':>9} {'tPort(s)':>9}")
    for r in comparison:
        print(
            f"  {r['title'][:33]:<33} {r['token_count']:>7} "
            f"{r['vocab_lemma']:>8} {r['vocab_porter']:>8} {r['vocab_snowball']:>8} "
            f"{r['time_lemma_s']:>9.5f} {r['time_porter_s']:>9.5f}"
        )

    elapsed = time.time() - t0
    print(f"\n✓ Parte 2 completada en {elapsed:.2f}s")
    print("=" * 60)

    return processed, comparison


if __name__ == "__main__":
    main()
