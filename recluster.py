"""
recluster.py — Rigenera SLM registry + slm_data/ dai chunk già presenti in
ChromaDB, senza ri-scaricare/ri-chunkare/ri-embeddare i 2000 paper arXiv.

Perché serve: la prima ingestion (ASSIGN_THRESHOLD=0.55, MERGE_THRESHOLD=0.88,
vedi app.py) ha prodotto un clustering degenerato — un solo SLM conteneva il
98.9% di tutti i chunk (100431 / 101542). Un secondo tentativo con l'assegnazione
incrementale chunk-per-chunk (soglia 0.75) ha invece prodotto l'estremo opposto:
~13-14 SLM per paper, proiettati a ~27.000 SLM su 2000 paper — troppi per
merge_close_slms, che è O(n²) per ogni singolo merge trovato (e ripete la
scansione O(n²) da capo per ogni merge successivo).

Il problema di fondo: il testo derivato da LaTeX di arXiv (token @xmath/@xcite
ripetuti ovunque) forma nello spazio di embedding di bge-m3 un blob denso senza
cluster naturalmente ben separati a livello di singolo chunk — non esiste una
soglia fissa che eviti sia il collasso (soglia bassa) sia l'esplosione in
migliaia di micro-cluster (soglia alta) con l'assegnazione incrementale.

Questo script sostituisce l'assegnazione incrementale con un clustering batch
(MiniBatchKMeans) sull'intero corpus di embedding in un colpo solo: si sceglie
direttamente il numero di cluster target (N_CLUSTERS) invece di una soglia di
similarità, ed è scalabile (pochi minuti anche su 100k+ vettori), a differenza
del merge O(n²) che non lo è.

Uso:
    python3 recluster.py
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, List

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

from router import (
    REGISTRY_PATH, SLM_DATA_DIR, save_registry, merge_close_slms,
    refresh_all_summaries, make_spacy_summary_fn, load_registry,
)

# ── Config ───────────────────────────────────────────────────────────────
HF_COLLECTION_NAME  = "scientific_papers_arxiv"
N_CLUSTERS          = 150    # numero di SLM target (invece di una soglia di similarità)
# Tetto di chunk per SLM: ~250K token / 193 token-per-chunk (misurati sul
# corpus) ≈ 1300 chunk. Ogni cluster resta così caricabile PER INTERO in un
# contesto da 256K, e cluster più piccoli/tematici danno profili keyword
# c-TF-IDF più nitidi per il routing. I cluster K-Means che superano il tetto
# vengono ri-clusterizzati ricorsivamente al loro interno (K-Means non
# garantisce bilanciamento: nel run precedente il massimo era 7.5× la media).
MAX_CHUNKS_PER_SLM  = 1300
MERGE_THRESHOLD     = 0.95   # merge finale dei cluster K-Means quasi identici (ora disattivato, vedi main)
SPACY_SUMMARY_MODEL = "en_core_web_lg"
FETCH_BATCH_SIZE    = 5000   # paginazione lettura ChromaDB (no filtro ids → nessun limite SQL)


def _reset_clustering_state() -> None:
    """
    Backup + azzera registry.json e slm_data/. I chunk in ChromaDB non vengono
    toccati. Non sovrascrive un backup già esistente da un tentativo precedente.
    """
    registry_path = Path(REGISTRY_PATH)
    backup_path = registry_path.with_suffix(".json.bak")
    if registry_path.exists():
        if backup_path.exists():
            print(f"[recluster] {backup_path} già presente, non lo sovrascrivo; rimuovo solo {REGISTRY_PATH}")
            registry_path.unlink()
        else:
            registry_path.replace(backup_path)
            print(f"[recluster] {REGISTRY_PATH} esistente spostato in {backup_path}")

    if SLM_DATA_DIR.exists():
        backup_dir = SLM_DATA_DIR.with_name(SLM_DATA_DIR.name + "_bak")
        if backup_dir.exists():
            print(f"[recluster] {backup_dir}/ già presente, non lo sovrascrivo; svuoto solo {SLM_DATA_DIR}/")
            shutil.rmtree(SLM_DATA_DIR)
        else:
            SLM_DATA_DIR.replace(backup_dir)
            print(f"[recluster] {SLM_DATA_DIR}/ esistente spostato in {backup_dir}/")

    SLM_DATA_DIR.mkdir(exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump({}, f)


def _fetch_all_chunks(collection) -> List[Dict]:
    """
    Legge TUTTI i chunk della collection paginando con limit/offset (non con
    ids=[...], che sbatte contro il limite di parametri SQL di ChromaDB su
    100k+ id).
    """
    chunks: List[Dict] = []
    offset = 0
    while True:
        data = collection.get(
            limit=FETCH_BATCH_SIZE, offset=offset,
            include=["documents", "embeddings"],
        )
        ids = data.get("ids")
        ids = list(ids) if ids is not None else []
        if not ids:
            break

        documents = data.get("documents")
        documents = documents if documents is not None else []
        embeddings = data.get("embeddings")
        embeddings = embeddings if embeddings is not None else []

        for i, chunk_id in enumerate(ids):
            chunks.append({
                "id":        chunk_id,
                "text":      documents[i] if i < len(documents) else "",
                "embedding": embeddings[i],
            })

        offset += FETCH_BATCH_SIZE
        print(f"[recluster] letti {len(chunks)} chunk da ChromaDB...", flush=True)

    return chunks


def _split_oversized_groups(
    groups: List[List[int]],
    embeddings: np.ndarray,
    max_size: int = MAX_CHUNKS_PER_SLM,
) -> List[List[int]]:
    """
    Ri-clusterizza ricorsivamente ogni gruppo che supera max_size chunk.

    K-Means non garantisce cluster bilanciati; questo passo trasforma il tetto
    di dimensione (⇒ ogni SLM caricabile per intero in un contesto 256K) da
    speranza a invariante strutturale.

    Args:
        groups:     lista di gruppi, ciascuno lista di indici in `embeddings`.
        embeddings: matrice (n_chunks, dim) di tutti gli embedding.

    Returns:
        Lista di gruppi tutti di dimensione <= max_size.
    """
    import math

    result: List[List[int]] = []
    queue = list(groups)
    while queue:
        group = queue.pop()
        if len(group) <= max_size:
            result.append(group)
            continue

        k = math.ceil(len(group) / max_size) + 1
        sub_embs = embeddings[group]
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
        sub_labels = km.fit_predict(sub_embs)

        sub_groups: Dict[int, List[int]] = {}
        for local_i, lab in enumerate(sub_labels):
            sub_groups.setdefault(int(lab), []).append(group[local_i])

        if len(sub_groups) <= 1:
            # K-Means non riesce a separare (chunk quasi identici): accetta
            # il gruppo com'è per evitare un loop infinito.
            result.append(group)
            continue

        queue.extend(sub_groups.values())

    return result


def _build_registry_from_groups(chunks: List[Dict], groups: List[List[int]], embeddings: np.ndarray) -> None:
    """
    Scrive registry.json + slm_data/*.json dai gruppi finali, un SLM per gruppo.
    Il centroide è ricalcolato dagli embedding reali dei membri (dopo gli
    split i centri K-Means originali non corrispondono più ai gruppi).
    """
    registry: Dict = {}
    for group in groups:
        slm_name = f"slm_{uuid.uuid4().hex[:8]}"
        chunk_ids = [chunks[i]["id"] for i in group]

        centroid = embeddings[group].mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        centroid = (centroid / norm).tolist() if norm > 0 else centroid.tolist()

        chunks_path = SLM_DATA_DIR / f"{slm_name}_chunks.json"
        with open(chunks_path, "w") as f:
            json.dump([{"id": cid} for cid in chunk_ids], f, indent=2)

        registry[slm_name] = {
            "collection":        HF_COLLECTION_NAME,
            "chunks_json":       str(chunks_path),
            "chunk_count":       len(chunk_ids),
            "centroid_embedding": centroid,
            "topic_summary":     "",
            "keywords":          [],
            "summary_embedding": [],
        }

    save_registry(registry)
    print(f"[recluster] {len(registry)} SLM scritti da {len(chunks)} chunk "
          f"(K iniziale={N_CLUSTERS}, tetto={MAX_CHUNKS_PER_SLM} chunk/SLM).")


def main() -> None:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_collection(HF_COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{HF_COLLECTION_NAME}' non trovata in ./chroma_db. "
            f"Esegui prima l'ingestion (python app.py)."
        ) from e

    print(f"[recluster] {collection.count()} chunk nella collection '{HF_COLLECTION_NAME}'.")
    print(f"[recluster] N_CLUSTERS={N_CLUSTERS}  MERGE_THRESHOLD={MERGE_THRESHOLD}")

    _reset_clustering_state()

    print("[recluster] Carico embedding model e modello spaCy...")
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    spacy_fn = make_spacy_summary_fn(SPACY_SUMMARY_MODEL)

    chunks = _fetch_all_chunks(collection)
    print(f"[recluster] {len(chunks)} chunk totali letti, avvio MiniBatchKMeans...")

    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=4096, n_init=3)
    labels = km.fit_predict(embeddings)

    initial_groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        initial_groups.setdefault(int(lab), []).append(i)

    print(f"[recluster] {len(initial_groups)} cluster K-Means iniziali; "
          f"split di quelli oltre {MAX_CHUNKS_PER_SLM} chunk...")
    final_groups = _split_oversized_groups(list(initial_groups.values()), embeddings)

    _build_registry_from_groups(chunks, final_groups, embeddings)

    # NB: niente merge_close_slms qui. Nel run precedente (K=120) il merge a
    # soglia 0.95 ha eseguito 48 fusioni ed è la causa principale dei cluster
    # giganti (fino a 14.9k chunk): fondere due cluster da 1000 chunk viola il
    # tetto MAX_CHUNKS_PER_SLM appena imposto dallo split. Con K-Means la
    # granularità la decide già N_CLUSTERS.
    #
    # print("[recluster] Merge cluster quasi identici...")
    # merges = merge_close_slms(
    #     threshold=MERGE_THRESHOLD, embedding_model=embedding_model, chroma_client=chroma_client
    # )
    # print(f"[recluster] {len(merges)} merge effettuati.")

    print("[recluster] Estrazione keyword/summary con spaCy...")
    refresh_all_summaries(embedding_model, chroma_client, summary_fn=spacy_fn)

    registry = load_registry()
    sizes = sorted((v["chunk_count"] for v in registry.values()), reverse=True)
    total_chunks = sum(sizes)
    print(f"\n[recluster] Completato: {len(registry)} SLM, {total_chunks} chunk totali.")
    print(f"[recluster] Distribuzione (dal più grande): {sizes}")
    if sizes:
        print(f"[recluster] Cluster più grande: {sizes[0]} chunk ({sizes[0] / total_chunks * 100:.1f}% del totale)")


if __name__ == "__main__":
    main()