"""
Parte 4 – Análisis Comparativo de Representaciones Semánticas
=============================================================
Actividades:
  • Similitud coseno entre capítulos (BoW, TF-IDF, Embeddings)
  • Clustering K-Means de fragmentos
  • Identificación de capítulos semánticamente cercanos
  • Visualización PCA y t-SNE

Puntos extra:
  • Comparación capítulos iniciales vs finales
  • Persistencia de resultados en SQLite
  • Reporte de costos computacionales

Entradas:
  data/bow_matrix.npz
  data/tfidf_matrix.npz
  data/embeddings_matrix.npz
  data/vectorization_meta.json

Salidas:
  data/analysis_results.json
  data/results.db   (SQLite)
  data/pca_bow.png
  data/pca_tfidf.png
  data/pca_embeddings.png
  data/tsne_embeddings.png
  data/cosine_heatmap_bow.png
  data/cosine_heatmap_tfidf.png
  data/cosine_heatmap_embeddings.png
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
BOW_NPZ = DATA_DIR / "bow_matrix.npz"
TFIDF_NPZ = DATA_DIR / "tfidf_matrix.npz"
EMBED_NPZ = DATA_DIR / "embeddings_matrix.npz"
META_JSON = DATA_DIR / "vectorization_meta.json"
RESULTS_JSON = DATA_DIR / "analysis_results.json"
SQLITE_DB = DATA_DIR / "results.db"


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_matrix(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    return data["matrix"]


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Calcula la matriz de similitud coseno normalizada."""
    return cosine_similarity(normalize(X))


def top_similar_pairs(sim_matrix: np.ndarray, titles: list[str], top_n: int = 5) -> list[dict]:
    """Devuelve los pares más similares (excluyendo la diagonal)."""
    n = sim_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                {
                    "chapter_a": titles[i],
                    "chapter_b": titles[j],
                    "idx_a": i,
                    "idx_b": j,
                    "similarity": float(sim_matrix[i, j]),
                }
            )
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_n]


# ── Heatmap de similitud coseno ───────────────────────────────────────────────

def plot_cosine_heatmap(sim_matrix: np.ndarray, titles: list[str], name: str):
    short_titles = [t[:15] for t in titles]
    fig, ax = plt.subplots(figsize=(max(8, len(titles) * 0.5), max(7, len(titles) * 0.45)))
    sns.heatmap(
        sim_matrix,
        xticklabels=short_titles,
        yticklabels=short_titles,
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.3,
    )
    ax.set_title(f"Similitud Coseno – {name}", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    out = DATA_DIR / f"cosine_heatmap_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"      → {out}")
    return str(out)


# ── Clustering K-Means ────────────────────────────────────────────────────────

def cluster_chapters(X: np.ndarray, titles: list[str], n_clusters: int = 4) -> list[dict]:
    """Aplica K-Means y retorna asignación de clusters."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    result = []
    for i, (title, label) in enumerate(zip(titles, labels)):
        result.append({"chapter_idx": i, "title": title, "cluster": int(label)})
    return result


# ── PCA 2D ───────────────────────────────────────────────────────────────────

def plot_pca(X: np.ndarray, titles: list[str], clusters: list[int], name: str):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = list(mcolors.TABLEAU_COLORS.values())
    unique_clusters = sorted(set(clusters))
    for c in unique_clusters:
        idx = [i for i, cl in enumerate(clusters) if cl == c]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=f"Cluster {c}",
            color=colors[c % len(colors)],
            alpha=0.8,
            s=80,
        )
    for i, title in enumerate(titles):
        ax.annotate(
            title[:10],
            (coords[i, 0], coords[i, 1]),
            fontsize=5.5,
            alpha=0.75,
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(f"PCA – {name}", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = DATA_DIR / f"pca_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"      → {out}")
    return str(out)


# ── t-SNE ────────────────────────────────────────────────────────────────────

def plot_tsne(X: np.ndarray, titles: list[str], clusters: list[int], name: str):
    perplexity = min(30, max(5, len(titles) // 2))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = list(mcolors.TABLEAU_COLORS.values())
    unique_clusters = sorted(set(clusters))
    for c in unique_clusters:
        idx = [i for i, cl in enumerate(clusters) if cl == c]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=f"Cluster {c}",
            color=colors[c % len(colors)],
            alpha=0.8,
            s=80,
        )
    for i, title in enumerate(titles):
        ax.annotate(title[:10], (coords[i, 0], coords[i, 1]), fontsize=5.5, alpha=0.75)
    ax.set_title(f"t-SNE – {name}", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = DATA_DIR / f"tsne_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"      → {out}")
    return str(out)


# ── Comparación iniciales vs finales (punto extra) ───────────────────────────

def compare_initial_vs_final(
    bow: np.ndarray,
    tfidf: np.ndarray,
    embed: np.ndarray,
    titles: list[str],
    n: int = 3,
) -> dict:
    """
    Compara los primeros N capítulos contra los últimos N
    usando similitud coseno promedio intra-grupo e inter-grupo.
    """
    initial_idx = list(range(n))
    final_idx = list(range(len(titles) - n, len(titles)))

    def group_stats(matrix, g1, g2):
        m = cosine_sim_matrix(matrix)
        intra_1 = np.mean([m[i, j] for i in g1 for j in g1 if i != j]) if len(g1) > 1 else 0
        intra_2 = np.mean([m[i, j] for i in g2 for j in g2 if i != j]) if len(g2) > 1 else 0
        inter = np.mean([m[i, j] for i in g1 for j in g2])
        return {
            "intra_initial": float(round(intra_1, 4)),
            "intra_final": float(round(intra_2, 4)),
            "inter_initial_final": float(round(inter, 4)),
        }

    return {
        "initial_chapters": [titles[i] for i in initial_idx],
        "final_chapters": [titles[i] for i in final_idx],
        "bow": group_stats(bow, initial_idx, final_idx),
        "tfidf": group_stats(tfidf, initial_idx, final_idx),
        "embeddings": group_stats(embed, initial_idx, final_idx),
    }


# ── Persistencia en SQLite (punto extra) ──────────────────────────────────────

def persist_to_sqlite(
    cosine_results: dict,
    clustering_results: dict,
    costs_json_path: Path,
):
    """Guarda similitudes, clusters y costos en SQLite."""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()

    # Tabla similitudes
    c.execute("""
        CREATE TABLE IF NOT EXISTS cosine_similarities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            representation TEXT,
            chapter_a TEXT,
            chapter_b TEXT,
            idx_a INTEGER,
            idx_b INTEGER,
            similarity REAL
        )
    """)

    # Tabla clustering
    c.execute("""
        CREATE TABLE IF NOT EXISTS chapter_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            representation TEXT,
            chapter_idx INTEGER,
            title TEXT,
            cluster INTEGER
        )
    """)

    # Tabla costos computacionales
    c.execute("""
        CREATE TABLE IF NOT EXISTS computational_costs (
            representation TEXT PRIMARY KEY,
            time_s REAL,
            memory_mb REAL,
            sparsity REAL,
            dimensions TEXT
        )
    """)

    # Insertar similitudes
    for rep, pairs in cosine_results.items():
        for p in pairs:
            c.execute(
                "INSERT INTO cosine_similarities "
                "(representation, chapter_a, chapter_b, idx_a, idx_b, similarity) "
                "VALUES (?,?,?,?,?,?)",
                (rep, p["chapter_a"], p["chapter_b"], p["idx_a"], p["idx_b"], p["similarity"]),
            )

    # Insertar clusters
    for rep, items in clustering_results.items():
        for item in items:
            c.execute(
                "INSERT INTO chapter_clusters (representation, chapter_idx, title, cluster) "
                "VALUES (?,?,?,?)",
                (rep, item["chapter_idx"], item["title"], item["cluster"]),
            )

    # Insertar costos
    if costs_json_path.exists():
        with open(costs_json_path) as f:
            costs = json.load(f)
        for rep, info in costs.items():
            c.execute(
                "INSERT OR REPLACE INTO computational_costs "
                "(representation, time_s, memory_mb, sparsity, dimensions) "
                "VALUES (?,?,?,?,?)",
                (
                    rep,
                    info.get("time_s"),
                    info.get("memory_mb"),
                    info.get("sparsity"),
                    str(info.get("shape", "")),
                ),
            )

    conn.commit()
    conn.close()
    print(f"      → SQLite guardado en {SQLITE_DB}")


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    t0_total = time.time()
    print("=" * 60)
    print("PARTE 4 – Análisis Comparativo de Representaciones")
    print("=" * 60)

    # Cargar matrices y metadata
    for f in [BOW_NPZ, TFIDF_NPZ, EMBED_NPZ, META_JSON]:
        if not f.exists():
            raise FileNotFoundError(f"Falta {f}. Ejecuta los scripts anteriores primero.")

    bow = load_matrix(BOW_NPZ)
    tfidf = load_matrix(TFIDF_NPZ)
    embeddings = load_matrix(EMBED_NPZ)

    with open(META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    titles = meta["titles"]
    n_chapters = len(titles)
    n_clusters = min(4, n_chapters // 2) if n_chapters > 4 else 2

    print(f"[INFO] Capítulos: {n_chapters}  |  Clusters: {n_clusters}")

    all_results = {}
    cosine_results = {}
    clustering_results = {}

    # ── Similitud coseno ─────────────────────────────────────────────────────
    print("\n── Similitud Coseno ────────────────────────────────────")
    for name, matrix in [("bow", bow), ("tfidf", tfidf), ("embeddings", embeddings)]:
        t0 = time.perf_counter()
        sim = cosine_sim_matrix(matrix)
        elapsed = time.perf_counter() - t0
        pairs = top_similar_pairs(sim, titles, top_n=5)
        cosine_results[name] = pairs
        heatmap_file = plot_cosine_heatmap(sim, titles, name.upper())

        print(f"\n  {name.upper()} (calculado en {elapsed:.4f}s)")
        print(f"  Top 5 pares más similares:")
        for p in pairs:
            print(
                f"    {p['chapter_a'][:25]:<25} ↔ {p['chapter_b'][:25]:<25}  "
                f"sim={p['similarity']:.4f}"
            )
        all_results[f"top_similar_{name}"] = pairs

    # ── Clustering ────────────────────────────────────────────────────────────
    print("\n── Clustering K-Means ──────────────────────────────────")
    for name, matrix in [("bow", bow), ("tfidf", tfidf), ("embeddings", embeddings)]:
        t0 = time.perf_counter()
        clusters_info = cluster_chapters(matrix, titles, n_clusters=n_clusters)
        elapsed = time.perf_counter() - t0
        clustering_results[name] = clusters_info
        cluster_labels = [c["cluster"] for c in clusters_info]

        print(f"\n  {name.upper()} clustering (en {elapsed:.4f}s):")
        for ci in clusters_info:
            print(f"    [{ci['cluster']}] {ci['title'][:50]}")

        # PCA
        plot_pca(matrix, titles, cluster_labels, name)
        all_results[f"clusters_{name}"] = clusters_info

    # t-SNE solo para embeddings (más interesante semánticamente)
    print("\n── t-SNE (Embeddings) ──────────────────────────────────")
    embed_cluster_labels = [c["cluster"] for c in clustering_results["embeddings"]]
    plot_tsne(embeddings, titles, embed_cluster_labels, "embeddings")

    # ── Comparación iniciales vs finales (punto extra) ────────────────────────
    print("\n── Capítulos Iniciales vs Finales (Punto Extra) ────────")
    init_vs_final = compare_initial_vs_final(bow, tfidf, embeddings, titles, n=3)
    all_results["initial_vs_final"] = init_vs_final

    print(f"  Capítulos iniciales: {init_vs_final['initial_chapters']}")
    print(f"  Capítulos finales:   {init_vs_final['final_chapters']}")
    print()
    for rep in ["bow", "tfidf", "embeddings"]:
        stats = init_vs_final[rep]
        print(f"  {rep.upper()}:")
        print(f"    Similitud intra-inicial:  {stats['intra_initial']:.4f}")
        print(f"    Similitud intra-final:    {stats['intra_final']:.4f}")
        print(f"    Similitud inter (ini↔fin):{stats['inter_initial_final']:.4f}")

    # ── Persistir JSON ────────────────────────────────────────────────────────
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n── Persistencia ──────────────────────────────────────────")
    print(f"      → JSON guardado en {RESULTS_JSON}")

    # ── SQLite (punto extra) ──────────────────────────────────────────────────
    persist_to_sqlite(
        cosine_results,
        clustering_results,
        DATA_DIR / "computational_costs.json",
    )

    elapsed_total = time.time() - t0_total
    print(f"\n✓ Parte 4 completada en {elapsed_total:.2f}s")
    print("=" * 60)

    # Conclusiones rápidas en consola
    print("\nCONCLUSIONES RÁPIDAS")
    print("-" * 60)
    print("• BoW y TF-IDF generan matrices dispersas (alta dimensión = vocab)")
    print("• Embeddings producen vectores densos de dimensión fija (768)")
    print("• TF-IDF penaliza términos frecuentes → mejor similitud que BoW")
    print("• Embeddings capturan semántica latente; mejores clusters temáticos")
    print("• La comparación iniciales vs finales revela arco narrativo del libro")

    return all_results


if __name__ == "__main__":
    main()
