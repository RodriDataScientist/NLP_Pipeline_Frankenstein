"""
Parte 1 – Ingesta y segmentación del texto
==========================================
Lee el archivo HTML de Frankenstein (data/frankenstein.html),
elimina cabecera/pie de Gutenberg y segmenta en capítulos,
párrafos y oraciones.

Salidas:
    data/frankenstein_clean.txt
    data/segments.json
"""

import json
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
HTML_FILE = DATA_DIR / "frankenstein.html"
CLEAN_TXT = DATA_DIR / "frankenstein_clean.txt"
SEGMENTS_JSON = DATA_DIR / "segments.json"


# ── 1. Leer HTML y extraer texto plano ───────────────────────────────────────
def html_to_text(html_path: Path) -> str:
    """Extrae texto limpio del HTML de Gutenberg."""
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Eliminar scripts y estilos
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n")


# ── 2. Eliminar cabecera y pie de Gutenberg ──────────────────────────────────
GUTENBERG_START_PATTERNS = [
    r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG",
    r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG",
    r"START OF THE PROJECT GUTENBERG",
]
GUTENBERG_END_PATTERNS = [
    r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",
    r"\*\*\*END OF (THE|THIS) PROJECT GUTENBERG",
    r"End of (the )?Project Gutenberg",
]


def strip_gutenberg(text: str) -> str:
    """Elimina la cabecera y el pie legal de Gutenberg."""
    start_idx = 0
    for pat in GUTENBERG_START_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # El contenido empieza después de esa línea
            start_idx = text.find("\n", m.end()) + 1
            break

    end_idx = len(text)
    for pat in GUTENBERG_END_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            end_idx = m.start()
            break

    clean = text[start_idx:end_idx].strip()
    return clean


# ── 3. Segmentar en capítulos ─────────────────────────────────────────────────
# Frankenstein usa patrones como "Chapter I", "Chapter 1", "CHAPTER I", etc.
CHAPTER_PATTERN = re.compile(
    r"^\s*(chapter\s+[IVXLCDM\d]+[^\n]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Frankenstein tiene también cartas/letters antes de los capítulos
LETTER_PATTERN = re.compile(
    r"^\s*(letter\s+\d+[^\n]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

def split_chapters_html(html_path: Path):
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    chapters = []
    for div in soup.find_all("div", class_="chapter"):
        title_tag = div.find("h2")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)

        # Filtrar tabla de contenidos
        if "contents" in title.lower():
            continue

        paragraphs = [
            p.get_text(strip=True)
            for p in div.find_all("p")
            if p.get_text(strip=True)
        ]

        # Evitar capítulos vacíos por seguridad
        if not paragraphs:
            continue

        sentences = sent_tokenize(" ".join(paragraphs))

        chapters.append({
            "title": title,
            "paragraphs": paragraphs,
            "sentences": sentences,
        })

    return chapters

# ── Pipeline principal ────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("PARTE 1 – Ingesta y segmentación")
    print("=" * 60)

    if not HTML_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {HTML_FILE}. "
            "Coloca el archivo HTML en la carpeta data/."
        )

    # Paso 1: HTML → texto
    print(f"[1/4] Leyendo {HTML_FILE} …")
    raw_text = html_to_text(HTML_FILE)
    print(f"      Caracteres extraídos: {len(raw_text):,}")

    # Paso 2: Eliminar cabecera/pie
    print("[2/4] Eliminando cabecera y pie de Gutenberg …")
    clean_text = strip_gutenberg(raw_text)
    print(f"      Caracteres después de limpieza: {len(clean_text):,}")

    # Guardar texto limpio
    CLEAN_TXT.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_TXT.write_text(clean_text, encoding="utf-8")
    print(f"      → Guardado en {CLEAN_TXT}")

    # Paso 3: Segmentar
    print("[3/4] Segmentando en capítulos / cartas …")
    chapters = split_chapters_html(HTML_FILE)
    print(f"      Secciones encontradas: {len(chapters)}")
    for ch in chapters:
        print(
            f"        · {ch['title'][:60]:<60} "
            f"párrs={len(ch['paragraphs']):>3}  "
            f"oracs={len(ch['sentences']):>4}"
        )

    # Paso 4: Persistir JSON
    print("[4/4] Guardando segments.json …")
    # Serializar (omitimos raw_text para no duplicar)
    output = []
    for i, ch in enumerate(chapters):
        output.append(
            {
                "chapter_idx": i,
                "title": ch["title"],
                "num_paragraphs": len(ch["paragraphs"]),
                "num_sentences": len(ch["sentences"]),
                "paragraphs": ch["paragraphs"],
                "sentences": ch["sentences"],
            }
        )
    with open(SEGMENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"      → Guardado en {SEGMENTS_JSON}")

    elapsed = time.time() - t0
    print(f"\n✓ Parte 1 completada en {elapsed:.2f}s")
    print("=" * 60)

    # Ejemplo de salida
    if chapters:
        ch0 = chapters[0]
        print(f"\nEjemplo – {ch0['title']}")
        print(f"  Primer párrafo: {ch0['paragraphs'][0][:200]}…")
        print(f"  Primera oración: {ch0['sentences'][0][:150]}…")

    return chapters


if __name__ == "__main__":
    main()
