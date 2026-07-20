"""
refresh_summaries.py — Rigenera SOLO keyword + summary_embedding degli SLM
esistenti, senza toccare il clustering (registry/slm_data restano gli stessi
cluster, cambia solo il loro profilo semantico).

Usa il pipeline migliorato di router.py:
  - _clean_arxiv_text: rimozione artefatti LaTeX/arXiv prima di spaCy
  - filtri keyword arXiv (_ARXIV_NOISE_WORDS)
  - pesatura c-TF-IDF cross-cluster (keyword distintive del cluster, non
    semplicemente frequenti) — vedi refresh_all_summaries in router.py

Uso:
    python3 refresh_summaries.py
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from router import refresh_all_summaries, make_spacy_summary_fn, load_registry

SPACY_SUMMARY_MODEL = "en_core_web_lg"


def main() -> None:
    registry = load_registry()
    if not registry:
        raise RuntimeError("Registry vuoto: esegui prima ingestion + clustering.")

    print(f"[refresh] {len(registry)} SLM nel registry, rigenero keyword + summary embedding...")

    embedding_model = SentenceTransformer("BAAI/bge-m3")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    spacy_fn = make_spacy_summary_fn(SPACY_SUMMARY_MODEL)

    n = refresh_all_summaries(embedding_model, chroma_client, summary_fn=spacy_fn)
    print(f"[refresh] Completato: {n} SLM aggiornati.")

    # Anteprima keyword dei cluster più grandi, per ispezione rapida
    registry = load_registry()
    biggest = sorted(registry.items(), key=lambda kv: -kv[1]["chunk_count"])[:8]
    for slm_name, entry in biggest:
        print(f"\n{slm_name}  ({entry['chunk_count']} chunk)")
        print(f"  keywords: {json.dumps(entry.get('keywords', [])[:15], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
