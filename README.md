# CLARA — Clustered Language-Agent Retrieval Architecture

A **service-oriented multi-agent** Retrieval-Augmented Generation system.
Each semantic region of a document is encapsulated as an autonomous **Small
Language Model (SLM) agent service** exposing a uniform Domain-Specific
Language (DSL) interface. A semantic router selects the top-*K* most
relevant cluster agents per query; the **primary** agent handles retrieval
and generation, while **secondary** agents contribute evidence through a
dedicated inter-agent operation.

![CLARA Architecture](CLARA_Architecture_official.png)

---

## Architecture

```
┌────────────────────────── INGESTION ──────────────────────────┐

  PDF → Markdown → adaptive chunking → bge-m3 embeddings
                                  │
                                  ▼
                        ChromaDB collection
                                  │
                                  ▼
                       cluster assignment
                  (per-chapter, or cosine on centroid)
                                  │
                                  ▼
                       merge close clusters
                                  │
                                  ▼
                       keyword + summary embedding
                                  │
                                  ▼
                       registry.json + slm_data/

┌──────────────────────────── QUERY ────────────────────────────┐

  Query q
       │
       ▼
   Orchestrator: bge-m3 → q_emb
       │
       ▼
   semantic routing — cosine(q_emb, P_j.summary_embedding)
       │              top-K cluster agents (default K = 3)
       ▼
   primary  A*  = argmax score
   secondary A^- = the remaining K − 1 agents
       │
       ├──► A*   .local_retrieve(q)        → Cˆ*    (over M_j only)
       │    A*   .select_evidence(q, Cˆ*)  → E*
       │
       ├──► A_1^-.provide_evidence(q)      → E_1^-
       ├──► A_2^-.provide_evidence(q)      → E_2^-
       │     ...
       ▼
   orchestrator: Γ = E* ∪ E_1^- ∪ … ∪ E_{K-1}^-
       │
       ▼
   A*.generate(q, Γ) → answer
```

### Cluster-Agent DSL contract

```
service ClusterAgent {
  memory   M_j : ChunkSet
  profile  P_j : SemanticProfile
  invariant locality : all retrieval operations access only M_j

  operation local_retrieve(q)        -> RankedChunks
  operation select_evidence(q, C)    -> EvidenceSet
  operation provide_evidence(q)      -> EvidenceSet
  operation generate(q, Gamma)       -> Answer
      requires invocation_role = primary
}
```

### Retrieval modes compared in the benchmark

| Mode       | Routing       | Pool          | Notes                                         |
|------------|---------------|---------------|-----------------------------------------------|
| **StdRAG** | none          | full corpus   | flat baseline                                 |
| **CLARA**  | top-K agents  | ~6–15 chunks  | DSL-driven primary + `provide_evidence` flow  |

---

## Stack

| Layer              | Component                                          |
|--------------------|----------------------------------------------------|
| PDF parsing        | `pymupdf4llm`                                      |
| Chunking           | `langchain-text-splitters` (Recursive)             |
| Embedding model    | `BAAI/bge-m3` (1024-D, multilingual)               |
| Vector store       | ChromaDB persistent client (cosine)                |
| Keyword extraction | spaCy `it_core_news_lg` (NER + POS)                |
| Routing            | cosine over `summary_embedding`                    |
| Lexical retrieval  | `rank_bm25` (per-agent, bounded to *M_j*)          |
| Fusion             | Reciprocal Rank Fusion                             |
| DSL service        | `ClusterAgent`                                     |
| Orchestrator       | `Orchestrator`                                     |
| Generation LLM     | Qwen3.5-4B, INT8 on CUDA / fp32 on CPU             |
| UI                 | Gradio                                             |

## Source files

| File              | Role                                                                  |
|-------------------|-----------------------------------------------------------------------|
| `chunking.py`     | PDF → Markdown, doc-type detection, adaptive chunk profiles           |
| `router.py`       | SLM registry, assignment, merge, summary refresh, semantic routing    |
| `agent.py`        | `ClusterAgent` — DSL service                                          |
| `orchestrator.py` | `Orchestrator` — routing, role assignment, Γ aggregation, generation  |
| `SLMAgent.py`     | Prompt construction (chat-template helper)                            |
| `app.py`          | Gradio UI + ingestion + streaming query                               |
| `benchmark.py`    | StdRAG vs CLARA quantitative comparison on 75 queries                 |

---

## SLM registry

```json
{
  "slm_7d833ad1": {
    "collection":         "storia",
    "chunks_json":        "slm_data/slm_7d833ad1_chunks.json",
    "chunk_count":        2,
    "centroid_embedding": [0.0094, 0.0458, ...],
    "topic_summary":      "## 2.1. The route to the East ...",
    "keywords":           ["Henry the Navigator", "East", "Indies", ...],
    "summary_embedding":  [...]
  }
}
```

---

## Quickstart

### Installation

```bash
pip install -r requirements.txt
python -m spacy download it_core_news_lg   # legacy PDF pipeline (chunking.process_pdf)
python -m spacy download en_core_web_lg    # HF dataset ingestion (armanc/scientific_papers, English)
```

### Run the UI

```bash
python app.py
# → http://localhost:7860
```

On first launch (empty `registry.json`), `app.py` streams
`armanc/scientific_papers` (config `arxiv`, split `train`, capped at
`MAX_DOCS` papers — see the constants at the top of `app.py`) straight from
Hugging Face, chunks + embeds + clusters it into SLMs, then serves the Query
tab. Subsequent launches skip re-ingestion as long as `registry.json` is
already populated.

### Run the benchmark

```bash
python benchmark.py
python evaluate_quality.py
```

---

## Project structure

```
.
├── chunking.py         # PDF → Markdown, adaptive profiles
├── router.py           # registry, assignment, merge, summary, routing
├── agent.py            # ClusterAgent + DSL
├── orchestrator.py     # Orchestrator workflow
├── SLMAgent.py         # prompt construction
├── app.py              # Gradio UI + streaming query
├── benchmark.py        # StdRAG vs CLARA comparison
├── evaluate_quality.py # RAGAS evaluation
│
├── registry.json       # generated at runtime
├── slm_data/           # generated at runtime
├── chroma_db/          # generated at runtime
│
├── requirements.txt
└── LICENSE
```

`registry.json`, `slm_data/`, and `chroma_db/` are gitignored: they are
rebuilt on first ingestion.

---

## License

See `LICENSE`.
