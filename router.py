"""
router.py — SLM registry, assignment, routing, and merge logic.

SLM (Semantic Language Map): a semantic cluster of chunks from ChromaDB.
Primary representation: centroid_embedding (mean of all chunk embeddings).
Secondary representation: summary_embedding (embedding of LLM-generated topic_summary).
"""

import json
import re
import uuid
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer


REGISTRY_PATH = "registry.json"
SLM_DATA_DIR = Path("slm_data")

# How often (in chunks added) the summary is refreshed during incremental ingestion.
SLM_SUMMARY_EVERY_N: int = 20

SUMMARY_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
# Model-agnostic summary function type.
# Input:  list of chunk texts (representative subset)
# Output: (topic_summary: str, keywords: List[str])
SummaryFn = Callable[[List[str]], Tuple[str, List[str]]]

# Module-level cache: loaded once on first use
_summary_tok = None
_summary_mdl = None
_kw_llm      = None   # KeyLLM TextGeneration instance, built from _summary_mdl

_KW_PROMPT = (
    "Extract 10-15 index tags from the text below.\n"
    "Return ONLY a comma-separated list. No explanations. No labels. No rules.\n\n"
    "Good example output:\n"
    "James Watt, macchina a vapore, 1769, carbone, Rivoluzione Industriale, "
    "George Stephenson, locomotiva, siderurgia, proletariato, enclosures\n\n"
    "If the text contains only page numbers, URLs, image captions, index entries, "
    "or formatting symbols with no real content, return exactly: SKIP\n\n"
    "Text:\n[DOCUMENT]\n\n"
    "Tags:\n"
)

# ── I/O helpers ────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    SLM_DATA_DIR.mkdir(exist_ok=True)


def load_registry() -> Dict:
    path = Path(REGISTRY_PATH)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_registry(registry: Dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ── NEW: Centroid computation ───────────────────────────────────────────

def _update_centroid(
    old_centroid: Optional[List[float]],
    new_embedding: np.ndarray,
    n: int,
) -> List[float]:
    """
    Incrementally update a normalized centroid by adding one new embedding.

    Args:
        old_centroid: current centroid as a float list, or None if first chunk.
        new_embedding: embedding of the incoming chunk (1-D float32 array).
        n: chunk count AFTER adding the new chunk.

    Returns:
        Updated, L2-normalized centroid as a float list.
    """
    if old_centroid is None or n == 1:
        vec = new_embedding.astype(np.float32)
    else:
        c = np.array(old_centroid, dtype=np.float32)
        vec = (c * (n - 1) + new_embedding) / n
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


# ── All-chunk text retrieval ───────────────────────────────────────────

# ChromaDB's SQLite backend caps bound parameters per query (SQLITE_MAX_VARIABLE_NUMBER,
# commonly 999). SLM clusters can grow past that after enough merges, so any
# col.get(ids=...) over a whole cluster must be paginated.
CHROMA_GET_BATCH_SIZE: int = 500


def _batched(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _get_all_chunk_texts(
    chunk_ids: List[str],
    chroma_client,
    collection_name: str,
) -> List[str]:
    """
    Retrieve the text of EVERY chunk in the cluster from ChromaDB.

    Using all chunks (instead of a representative subset) ensures that the
    keywords extracted by the summary LLM cover the full semantic breadth of
    the cluster, not just its core or boundary.

    Fetched in batches of CHROMA_GET_BATCH_SIZE ids to stay under ChromaDB's
    per-query bound-parameter limit for large clusters.

    Returns:
        List of chunk text strings (order not guaranteed).
    """
    try:
        col = chroma_client.get_collection(collection_name)
    except Exception:
        return []

    texts: List[str] = []
    for batch_ids in _batched(chunk_ids, CHROMA_GET_BATCH_SIZE):
        data = col.get(ids=batch_ids, include=["documents"])
        texts.extend(data.get("documents") or [])
    return texts


def unload_summary_model() -> None:
    """
    Deallocate the Qwen summary model from memory.
    Call after refresh_all_summaries to free GPU/CPU RAM.
    """
    import gc
    import torch

    global _summary_tok, _summary_mdl, _kw_llm

    _kw_llm = None

    if _summary_mdl is not None:
        del _summary_mdl
        _summary_mdl = None

    if _summary_tok is not None:
        del _summary_tok
        _summary_tok = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[router] Modello summary deallocato.")


# ── Qwen summary function ──────────────────────────────────────────────

_NOISE_PREFIXES = ("note:", "the summary", "the keyword", "a knowledge", "while the")

# Single-word noise terms from book structure/index
_NOISE_SINGLE_WORDS = frozenset([
    "approfondimenti", "multimediali", "sintesi", "mappe", "concettuali",
    "sezione", "volume", "capitolo", "pagina", "temi", "idea", "ricerca",
    "premesse", "elenco", "presentazione", "immagini", "utilizzate",
    "applicazioni", "caratteri", "protagonisti", "conseguenze", "invenzioni",
    "protagonista", "riassunto", "riepilogo", "scheda", "glossario",
])

# PDF artifact prefixes (line-break markers, encoding artifacts)
_KW_ARTIFACT_PREFIXES = ("br ", "c3 ", "c2 ", "br\n", "_", "@", "\\")

# arXiv/LaTeX artifact noise (single-word keywords to drop): placeholder tokens
# left by the scientific_papers preprocessing (@xmath/@xcite → xmath/xcite once
# '@' is stripped), structural words, and figure/table references.
_ARXIV_NOISE_WORDS = frozenset([
    "xmath", "xcite", "noop", "eq", "eqs", "fig", "figs", "figure", "table",
    "section", "sec", "chunk", "ref", "refs", "cite", "et al", "al",
    "ptabularcr", "tabular", "rectangle", "circle", "arrow", "dash",
    "doibase", "link", "http", "https", "www",
])


# ── arXiv text cleaning (pre keyword-extraction) ───────────────────────

# @xmath12, @xcite, @xnmath … placeholder tokens from the arXiv dump
_ARXIV_PLACEHOLDER = re.compile(r'@x?[a-z]+\d*')
# residual latex commands: \frac, \nonumber, \operatorname …
_LATEX_COMMAND     = re.compile(r'\\[a-zA-Z]+\*?')
# table column specs standing alone: "ccccc", "llccc", "rcl" …
_TABLE_COLSPEC     = re.compile(r'\b[lcr]{3,}\b')
# runs of underscores / repeated separators
_UNDERSCORE_RUN    = re.compile(r'_{2,}')


def _clean_arxiv_text(text: str) -> str:
    """
    Strip LaTeX/arXiv artifacts from chunk text before keyword extraction.

    The armanc/scientific_papers dump replaces math and citations with
    placeholder tokens (@xmath12, @xcite) and keeps raw table rows; spaCy
    happily extracts these as 'entities', polluting the SLM keyword profiles
    (observed in registry.json: '@noop', 'ptabularcr', whole table rows).
    Dropping table-like lines and placeholders leaves only the prose the
    keyword extraction is meant to see.
    """
    lines = []
    for line in text.split("\n"):
        # righe-tabella: molti separatori & o + (righe dati di tabelle latex)
        if line.count("&") >= 3 or line.count(" + ") >= 3:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    cleaned = _ARXIV_PLACEHOLDER.sub(" ", cleaned)
    cleaned = _LATEX_COMMAND.sub(" ", cleaned)
    cleaned = _TABLE_COLSPEC.sub(" ", cleaned)
    cleaned = _UNDERSCORE_RUN.sub(" ", cleaned)
    cleaned = re.sub(r'[{}$^~|]', " ", cleaned)
    cleaned = re.sub(r'\s{2,}', " ", cleaned)
    return cleaned.strip()


# Forma ammessa per una keyword: solo lettere (accentate incluse), cifre,
# spazi e trattini. Esclude label LaTeX (underscore), URL, formule (=, /, :),
# riferimenti tipo "eq.([...]" — tutto junk che il c-TF-IDF altrimenti premia
# perché massimamente "raro".
_KW_ALLOWED_CHARS = re.compile(r"^[a-zA-Z0-9À-ÿ\s\-]+$")
_KW_MAX_WORDS = 5
# Nessuna parola inglese/francese comune supera i 20 caratteri: token più
# lunghi sono quasi sempre label LaTeX concatenate ("maintheoremanditsconsequence").
_KW_MAX_WORD_LEN = 20
# Parole funzione francesi: ≥2 in una keyword ⇒ frammento di frase, non un
# termine ("dans la proposition suivante"). Una sola è ammessa: "la sr cuo
# systems" (lantanio!) è legittimo.
_FRENCH_FUNCTION_WORDS = frozenset([
    "dans", "une", "le", "la", "les", "des", "du", "est", "par", "pour",
    "dun", "dune", "qui", "que", "sur", "avec", "nous", "sont", "comme",
    "donc", "cette", "ces", "aux", "ont",
])


def _is_meaningful_kw(kw: str) -> bool:
    """Return True if kw is a keyword worth keeping — not noise, not a fragment."""
    kw = kw.strip()
    if len(kw) < 4:
        return False
    if len(kw) > 60 or "\n" in kw:
        return False
    if not _KW_ALLOWED_CHARS.match(kw):
        return False
    if len(kw.split()) > _KW_MAX_WORDS:
        return False
    if any(len(w) > _KW_MAX_WORD_LEN for w in kw.split()):
        return False
    if sum(1 for w in kw.lower().split() if w in _FRENCH_FUNCTION_WORDS) >= 2:
        return False
    # Starts with a digit: allow 4-digit years (1000-2100), reject page refs (1-3 digits)
    if kw[0].isdigit():
        first_token = kw.split()[0]
        if not (len(first_token) == 4 and first_token.isdigit()):
            return False
    # Ends with a 2-3 digit number (page refs: "restaurazione 153", "napoleon 146")
    # Allow 4-digit years: "luglio 1789", "pace Augusta 1555"
    if re.search(r'\s\d{2,3}$', kw):
        return False
    # PDF artifact prefixes
    if any(kw.lower().startswith(p) for p in _KW_ARTIFACT_PREFIXES):
        return False
    words = kw.lower().split()
    # All words are very short (e.g. "la xv", "il br", "dal xv")
    if all(len(w) <= 2 for w in words):
        return False
    # Single-word structural noise
    if len(words) == 1 and words[0] in _NOISE_SINGLE_WORDS:
        return False
    # arXiv/LaTeX artifact noise: drop keywords made only of placeholder /
    # structural tokens (xmath, xcite, fig, table…)
    if all(w in _ARXIV_NOISE_WORDS for w in words):
        return False
    # Keyword containing placeholder fragments anywhere ("xmath12", "eq. four")
    if any(w.startswith(("xmath", "xcite")) for w in words):
        return False
    return True


def _parse_raw_keywords(raw: str) -> List[str]:
    raw = raw.strip()
    
    # Modello ha riconosciuto chunk junk
    if raw.upper().startswith("SKIP"):
        return []
    
    result: List[str] = []
    for kw in raw.replace("\n", ",").split(","):
        kw = kw.strip()
        if not kw or len(kw) > 60:
            continue
        if kw.lower().startswith(_NOISE_PREFIXES):
            continue
        if not _is_meaningful_kw(kw):
            continue
        result.append(kw)
    return result


def _merge_keywords_with_counts(keywords: List[str]) -> List[Tuple[str, int]]:
    """
    Deduplicate and merge a flat list of keywords collected across all chunks,
    returning (keyword, count) pairs sorted by frequency descending.

    Steps:
      1. Case-insensitive dedup — count occurrences, keep the most common casing.
      2. Substring absorption — if keyword A (normalized) is a substring of keyword B,
         A is dropped and its count is added to B (the more specific form wins).
      3. Sort by frequency descending so the most cross-chunk keywords come first.

    I conteggi servono alla pesatura c-TF-IDF cross-cluster in
    refresh_all_summaries (frequenza nel cluster × rarità negli altri cluster).
    """
    # Step 1: normalize and count
    freq: Dict[str, int] = {}       # norm → count
    canonical: Dict[str, str] = {}  # norm → best original form (first seen)
    for kw in keywords:
        norm = kw.lower().strip().rstrip(".,;:")
        if not norm:
            continue
        if norm not in freq:
            freq[norm] = 0
            canonical[norm] = kw
        freq[norm] += 1

    # Step 2: substring absorption (longer/more-specific absorbs shorter/generic)
    norms = sorted(freq.keys(), key=len, reverse=True)  # longest first
    absorbed: set = set()
    for i, longer in enumerate(norms):
        if longer in absorbed:
            continue
        for shorter in norms[i + 1:]:
            if shorter in absorbed:
                continue
            if shorter in longer:  # e.g. "napoleon" ⊂ "napoleon bonaparte"
                freq[longer] += freq[shorter]
                absorbed.add(shorter)

    # Step 3: build result sorted by frequency, apply final noise filter
    result = [
        (canonical[n], freq[n])
        for n in norms
        if n not in absorbed and _is_meaningful_kw(canonical[n])
    ]
    result.sort(key=lambda x: -x[1])
    return result


def _merge_keywords(keywords: List[str]) -> List[str]:
    """Come _merge_keywords_with_counts, ma restituisce solo le keyword (API storica)."""
    return [kw for kw, _ in _merge_keywords_with_counts(keywords)]


# Maximum prompts per GPU forward pass.
# RTX 4080 SUPER (16GB): Qwen2.5-7B float16 ≈ 14GB weights → ~2GB headroom.
# KV cache per sequence at ~870 token (720 input + 150 output) ≈ 50MB → max ~16.
# 8 is the safe ceiling accounting for PyTorch allocator fragmentation.
KEYWORD_BATCH_SIZE: int = 8


def _get_kw_pipe():
    """
    Build (or return cached) a HuggingFace text-generation pipeline
    wrapping the already-loaded Qwen model.
    Left-padding is forced so causal generation is correct when batch_size > 1.
    """
    from transformers import pipeline as hf_pipeline

    global _kw_llm
    if _kw_llm is not None:
        return _kw_llm

    _summary_tok.padding_side = "left"
    _kw_llm = hf_pipeline(
        "text-generation",
        model=_summary_mdl,
        tokenizer=_summary_tok,
        do_sample=False,
        pad_token_id=_summary_tok.eos_token_id,
        max_new_tokens=150,
        return_full_text=False,
    )
    return _kw_llm


def _qwen_summary_fn(texts: List[str]) -> Tuple[str, List[str]]:
    """
    Generate keywords for every chunk via GPU-batched HF pipeline inference.
    All prompts are built upfront; the pipeline processes KEYWORD_BATCH_SIZE
    prompts per forward pass (true GPU batching, not sequential).
    Keywords from all chunks are aggregated and deduplicated by _merge_keywords.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global _summary_tok, _summary_mdl

    if _summary_tok is None or _summary_mdl is None:
        print(f"[router] Carico {SUMMARY_MODEL} in float16...")
        _summary_tok = AutoTokenizer.from_pretrained(SUMMARY_MODEL, trust_remote_code=True)
        _summary_mdl = AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()

    topic_summary = ""

    pipe = _get_kw_pipe()
    prompts = [_KW_PROMPT.replace("[DOCUMENT]", t) for t in texts]
    all_raw: List[str] = []

    outputs = pipe(prompts, batch_size=KEYWORD_BATCH_SIZE)
    for out in outputs:
        generated = out[0]["generated_text"]
        all_raw.extend(_parse_raw_keywords(generated))

    keywords = _merge_keywords(all_raw)
    return topic_summary, keywords


# ── Summary update ─────────────────────────────────────────────────────

def update_slm_summary(
    registry: Dict,
    slm_name: str,
    embedding_model: SentenceTransformer,
    chroma_client,
    summary_fn: Optional[SummaryFn] = None,
) -> None:
    """
    Regenerate topic_summary and keywords for an SLM using ALL chunks in the cluster.
    Always uses _qwen_summary_fn unless a custom summary_fn is passed.

    All chunk texts are passed to the summary function so that keywords reflect
    the full semantic content of the cluster, not just a representative subset.
    """
    if summary_fn is None:
        summary_fn = _qwen_summary_fn
    entry = registry[slm_name]
    chunks_path = Path(entry["chunks_json"])
    if not chunks_path.exists():
        return

    with open(chunks_path) as f:
        chunk_ids = [c["id"] for c in json.load(f)]
    if not chunk_ids:
        return

    # Fetch texts for every chunk in the cluster (no centroid-based filtering).
    texts = _get_all_chunk_texts(chunk_ids, chroma_client, entry["collection"])
    if not texts:
        return

    topic_summary, keywords = summary_fn(texts)
    if not topic_summary and not keywords:
        return

    entry["topic_summary"] = topic_summary
    entry["keywords"] = keywords

    if embedding_model is not None:
        routing_text = ", ".join(keywords) if keywords else topic_summary
        summary_emb = embedding_model.encode(
            [routing_text], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()
        entry["summary_embedding"] = summary_emb


# ── NEW: Batch summary refresh (call after full ingestion) ─────────────

# Quante keyword distintive tenere per SLM dopo la pesatura c-TF-IDF.
# 25 (non 15): con il routing max-sim per keyword un profilo più ricco copre
# meglio l'ampiezza tematica di cluster da migliaia di chunk, senza il rischio
# di "diluizione" che aveva l'embedding della stringa unica.
CTFIDF_TOP_K: int = 25
# Occorrenze minime nel cluster perché una keyword partecipi al ranking
# c-TF-IDF: gli hapax (count 1-2) sono quasi sempre label/refusi che la
# pesatura per rarità premierebbe proprio in quanto unici.
CTFIDF_MIN_COUNT: int = 3


def refresh_all_summaries(
    embedding_model: SentenceTransformer,
    chroma_client,
    summary_fn: Optional[SummaryFn] = None,
) -> int:
    """
    Rigenera keyword + summary_embedding per tutti gli SLM.

    Se summary_fn espone la variante `with_counts` (make_spacy_summary_fn),
    usa una pesatura c-TF-IDF a due fasi:
      Fase A — estrae (keyword, conteggio) da ogni cluster;
      Fase B — pesa ogni keyword per frequenza nel cluster × log-rarità negli
               altri cluster (score = count × log((1+N)/(1+df))), tenendo le
               top CTFIDF_TOP_K *distintive*.
    Con il ranking per sola frequenza, parole generiche del corpus ("state",
    "system", "model") dominavano ogni cluster e i summary embedding
    risultavano quasi indistinguibili tra loro — il routing non discriminava.
    Le keyword che compaiono in tutti i cluster hanno score 0 e spariscono.

    Se summary_fn non espone i conteggi (es. Qwen), usa il vecchio flusso
    single-pass per-cluster invariato.
    """
    import gc
    import math
    import torch

    registry = load_registry()
    fn_counts = getattr(summary_fn, "with_counts", None)

    if fn_counts is None:
        # ── Vecchio flusso single-pass (Qwen o summary_fn custom) ──────────
        count = 0
        for slm_name in list(registry.keys()):
            update_slm_summary(registry, slm_name, embedding_model, chroma_client, summary_fn)
            count += 1
        save_registry(registry)

        unload_summary_model()
        gc.collect()
        torch.cuda.empty_cache()

        return count

    # ── Fase A: (keyword, count) per ogni SLM ──────────────────────────────
    per_slm_counts: Dict[str, List[Tuple[str, int]]] = {}
    for slm_name, entry in registry.items():
        chunks_path = Path(entry["chunks_json"])
        if not chunks_path.exists():
            continue
        with open(chunks_path) as f:
            chunk_ids = [c["id"] for c in json.load(f)]
        if not chunk_ids:
            continue
        texts = _get_all_chunk_texts(chunk_ids, chroma_client, entry["collection"])
        if not texts:
            continue
        per_slm_counts[slm_name] = fn_counts(texts)

    # ── Fase B: document frequency cross-cluster + reweighting ─────────────
    n_clusters = len(per_slm_counts)
    df: Dict[str, int] = {}
    for counts in per_slm_counts.values():
        for kw, _ in counts:
            norm = kw.lower()
            df[norm] = df.get(norm, 0) + 1

    for slm_name, counts in per_slm_counts.items():
        scored = [
            (kw, c * math.log((1 + n_clusters) / (1 + df[kw.lower()])))
            for kw, c in counts
            if c >= CTFIDF_MIN_COUNT
        ]
        scored.sort(key=lambda x: -x[1])
        top = [kw for kw, s in scored[:CTFIDF_TOP_K] if s > 0]
        if not top:
            # Nessuna keyword sopra la soglia o tutte presenti ovunque:
            # fallback alle più frequenti per non lasciare il profilo vuoto.
            top = [kw for kw, _ in counts[:CTFIDF_TOP_K]]

        entry = registry[slm_name]
        entry["topic_summary"] = ""
        entry["keywords"] = top

        routing_text = ", ".join(top)
        entry["summary_embedding"] = embedding_model.encode(
            [routing_text], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

        # Embedding per singola keyword, per il routing max-sim: al query time
        # il cluster è agganciato dalla keyword più vicina alla query, non da
        # una media in cui i termini si diluiscono a vicenda.
        kw_embs = embedding_model.encode(
            top, normalize_embeddings=True, convert_to_numpy=True
        )
        entry["keyword_embeddings"] = [e.tolist() for e in kw_embs]

    save_registry(registry)

    unload_summary_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return n_clusters




# ── UPDATED: SLM creation ──────────────────────────────────────────────

def create_slm(
    registry: Dict,
    chunk_id: str,
    embedding: np.ndarray,
    collection_name: str,
) -> str:
    """
    Create a new SLM seeded from a single chunk.
    Centroid is initialized to the chunk embedding.
    Summary is left empty; call update_slm_summary / refresh_all_summaries later.
    """
    _ensure_dirs()
    slm_name = f"slm_{uuid.uuid4().hex[:8]}"

    registry[slm_name] = {
        "collection":        collection_name,
        "chunks_json":       str(SLM_DATA_DIR / f"{slm_name}_chunks.json"),
        "chunk_count":       1,
        "centroid_embedding": embedding.tolist(),  # NEW
        "topic_summary":     "",                   # NEW (populated by summary_fn)
        "keywords":          [],                   # NEW
        "summary_embedding": [],                   # NEW (populated by summary_fn)
    }

    chunks_path = SLM_DATA_DIR / f"{slm_name}_chunks.json"
    with open(chunks_path, "w") as f:
        json.dump([{"id": chunk_id}], f, indent=2)

    return slm_name


# ── UPDATED: Single chunk assignment ───────────────────────────────────

def assign_chunk(
    chunk_id: str,
    embedding: np.ndarray,
    collection_name: str,
    chroma_client,
    embedding_model: Optional[SentenceTransformer] = None,
    threshold: float = 0.65,
    summary_fn: Optional[SummaryFn] = None,
) -> str:
    """
    Assign a chunk to the most similar SLM (by centroid cosine similarity),
    or create a new SLM if no match exceeds threshold.

    Centroid is updated incrementally on every assignment.
    Summary is regenerated every SLM_SUMMARY_EVERY_N chunks when summary_fn is provided.
    """
    registry = load_registry()

    best_slm: Optional[str] = None
    best_score: float = -1.0
    e_norm = float(np.linalg.norm(embedding))

    for slm_name, entry in registry.items():
        centroid = entry.get("centroid_embedding")
        if not centroid:
            continue
        c = np.array(centroid, dtype=np.float32)
        c_norm = float(np.linalg.norm(c))
        if c_norm == 0 or e_norm == 0:
            continue
        score = float(np.dot(embedding, c) / (e_norm * c_norm))
        if score > best_score:
            best_score = score
            best_slm = slm_name

    if best_slm is None or best_score < threshold:
        slm_name = create_slm(registry, chunk_id, embedding, collection_name)
        save_registry(registry)
        return slm_name

    # Add chunk to existing SLM
    entry = registry[best_slm]
    n = entry["chunk_count"] + 1
    entry["chunk_count"] = n

    # UPDATED: Incremental centroid update
    entry["centroid_embedding"] = _update_centroid(
        entry.get("centroid_embedding"), embedding, n
    )

    chunks_path = Path(entry["chunks_json"])
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    chunks.append({"id": chunk_id})
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)

    # Periodic summary refresh during ingestion (optional)
    if (
        n % SLM_SUMMARY_EVERY_N == 0
        and summary_fn is not None
        and embedding_model is not None
    ):
        update_slm_summary(registry, best_slm, embedding_model, chroma_client, summary_fn)

    save_registry(registry)
    return best_slm


# ── UPDATED: Batch chunk assignment ────────────────────────────────────

def assign_chunks(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    collection_name: str,
    chroma_client,
    threshold: float = 0.65,
    summary_fn: Optional[SummaryFn] = None,
) -> Dict[str, List[str]]:
    """
    Assign all chunks from a document to SLMs.
    Pass summary_fn to enable periodic in-flight summary updates (every SLM_SUMMARY_EVERY_N chunks).
    Call refresh_all_summaries after ingestion for a full summary pass.
    """
    assignments: Dict[str, List[str]] = {}
    for chunk in chunks:
        embedding = np.array(chunk["embedding"], dtype=np.float32)
        slm_name = assign_chunk(
            chunk_id=chunk["id"],
            embedding=embedding,
            collection_name=collection_name,
            chroma_client=chroma_client,
            embedding_model=embedding_model,
            threshold=threshold,
            summary_fn=summary_fn,
        )
        assignments.setdefault(slm_name, []).append(chunk["id"])
    return assignments


# ── Chapter-based SLM assignment ──────────────────────────────────────

def assign_chunks_by_chapter(
    chunks: List[Dict],
    collection_name: str,
) -> Dict[str, List[str]]:
    """
    Create one SLM per chapter using the 'chapter' metadata already present
    on each chunk (set by chunking.extract_chapters).

    This is the primary assignment strategy when chapter structure is detected.
    Each chapter becomes a single SLM; centroid is computed from all chunk
    embeddings in that chapter.

    Args:
        chunks:          list of dicts with keys id, text, chapter, embedding.
        collection_name: ChromaDB collection name stored in each SLM entry.

    Returns:
        assignments dict {slm_name: [chunk_id, ...]}
    """
    _ensure_dirs()

    # Group chunks by chapter title
    chapter_groups: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        chapter = chunk.get("chapter", "Document") or "Document"
        chapter_groups.setdefault(chapter, []).append(chunk)

    registry = load_registry()
    assignments: Dict[str, List[str]] = {}

    for chapter_title, chapter_chunks in chapter_groups.items():
        slm_name = f"slm_{uuid.uuid4().hex[:8]}"

        chunk_ids = [c["id"] for c in chapter_chunks]

        # Compute centroid from all chunk embeddings in this chapter
        vecs = np.array([c["embedding"] for c in chapter_chunks], dtype=np.float32)
        centroid = vecs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid = (centroid / norm).tolist() if norm > 0 else centroid.tolist()

        # Write chunk list to disk
        chunks_path = SLM_DATA_DIR / f"{slm_name}_chunks.json"
        with open(chunks_path, "w") as f:
            json.dump([{"id": cid} for cid in chunk_ids], f, indent=2)

        registry[slm_name] = {
            "collection":         collection_name,
            "chunks_json":        str(chunks_path),
            "chunk_count":        len(chunk_ids),
            "centroid_embedding": centroid,
            "topic_summary":      chapter_title,   # use chapter title as initial summary
            "keywords":           [],
            "summary_embedding":  [],
        }
        assignments[slm_name] = chunk_ids

    save_registry(registry)
    print(f"[chapter] {len(assignments)} SLM creati da {len(chunks)} chunk "
          f"({len(chapter_groups)} capitoli) in '{collection_name}'.")
    return assignments


def assign_chunks_auto(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    collection_name: str,
    chroma_client,
    threshold: float = 0.65,
) -> Tuple[Dict[str, List[str]], str]:
    """
    Primary entry point for chunk assignment.

    Strategy:
      - If multiple distinct chapters are detected → assign_chunks_by_chapter()
      - Otherwise (single 'Document' section or no chapters) → assign_chunks()

    Returns:
        (assignments, strategy_used)  where strategy_used is 'chapter' or 'cosine'.
    """
    chapters = {c.get("chapter", "Document") for c in chunks}
    # Fallback if only one chapter label (e.g. "Document") detected
    if len(chapters) <= 1:
        return assign_chunks(chunks, embedding_model, collection_name, chroma_client, threshold), "cosine"

    return assign_chunks_by_chapter(chunks, collection_name), "chapter"


# ── UPDATED: Query-time routing ────────────────────────────────────────

def find_top_n_slms(
    query_embedding: np.ndarray,
    registry: Dict,
    n: int = 2,
    mode: str = "summary",
) -> List[Tuple[str, float]]:
    """
    Route a query to the top-n SLMs by cosine similarity with the cluster's
    semantic profile.

    mode="summary"  (default): routing keyword-based (contributo del paper).
        Se l'entry ha `keyword_embeddings` (una per keyword, scritte da
        refresh_all_summaries), il punteggio del cluster è la MEDIA dei
        coseni tra la query e le singole keyword del profilo. Sweep empirico
        sulle 20 query di benchmark (recall@3/@5 dei cluster contenenti i
        top-5 chunk di StdRAG): mean 46%/51% > stringa unica 37%/43% >
        max-sim 16%/28% — il max amplifica i match spuri di keyword
        rumorose, la media li smorza; la stringa unica è penalizzata
        dall'essere out-of-distribution per bge-m3. In assenza di
        keyword_embeddings ricade sul confronto con summary_embedding
        (embedding della stringa concatenata, comportamento storico).
    mode="centroid": confronta con centroid_embedding (media degli embedding
        dei chunk). Baseline di confronto per l'ablation study.

    Args:
        query_embedding: normalized query embedding (1-D float32).
        registry:        loaded registry dict.
        n:               number of SLMs to return.
        mode:            "summary" | "centroid".

    Returns:
        List of (slm_name, score) sorted descending.
    """
    if mode not in ("summary", "centroid"):
        raise ValueError(f"mode sconosciuto: {mode!r} (usare 'summary' o 'centroid')")

    q_norm = float(np.linalg.norm(query_embedding))
    if q_norm == 0:
        raise ValueError("Query embedding is a zero vector.")

    scores: List[Tuple[str, float]] = []

    for slm_name, entry in registry.items():
        if mode == "summary" and entry.get("keyword_embeddings"):
            # Media dei coseni per keyword (embedding già normalizzati da refresh)
            kw_matrix = np.array(entry["keyword_embeddings"], dtype=np.float32)
            sims = kw_matrix @ (query_embedding / q_norm)
            score = float(sims.mean())
            scores.append((slm_name, score))
            continue

        emb_key = "summary_embedding" if mode == "summary" else "centroid_embedding"
        profile_emb = entry.get(emb_key)
        if not profile_emb:
            continue

        pv = np.array(profile_emb, dtype=np.float32)
        pv_norm = float(np.linalg.norm(pv))
        if pv_norm == 0:
            continue

        score = float(np.dot(query_embedding, pv) / (q_norm * pv_norm))
        scores.append((slm_name, score))

    if not scores:
        raise RuntimeError(f"Nessun SLM con profilo di routing valido ({mode}). Esegui prima l'ingestion.")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ── UPDATED: SLM merge ─────────────────────────────────────────────────

def merge_close_slms(
    threshold: float = 0.90,
    embedding_model: Optional[SentenceTransformer] = None,
    chroma_client=None,
    summary_fn: Optional[SummaryFn] = None,
) -> List[Dict]:
    """
    Iteratively merge SLM pairs whose centroid_embedding cosine similarity >= threshold.
    Keeps the SLM with more chunks. Merged centroid is a weighted mean of both centroids.
    Regenerates summary after merge if summary_fn is provided.
    """
    merges: List[Dict] = []

    while True:
        registry = load_registry()
        slm_names = list(registry.keys())
        if len(slm_names) < 2:
            break

        best_pair: Optional[Tuple[str, str]] = None
        best_score: float = -1.0

        for i in range(len(slm_names)):
            for j in range(i + 1, len(slm_names)):
                a, b = slm_names[i], slm_names[j]
                ca = registry[a].get("centroid_embedding")
                cb = registry[b].get("centroid_embedding")
                if ca is None or cb is None:
                    continue
                ca_arr = np.array(ca, dtype=np.float32)
                cb_arr = np.array(cb, dtype=np.float32)
                na, nb = np.linalg.norm(ca_arr), np.linalg.norm(cb_arr)
                if na == 0 or nb == 0:
                    continue
                score = float(np.dot(ca_arr, cb_arr) / (na * nb))
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        if best_pair is None or best_score < threshold:
            break

        a, b = best_pair
        count_a = registry[a]["chunk_count"]
        count_b = registry[b]["chunk_count"]
        kept, removed = (a, b) if count_a >= count_b else (b, a)

        # Merge chunk lists
        kept_path = Path(registry[kept]["chunks_json"])
        removed_path = Path(registry[removed]["chunks_json"])

        kept_chunks: List = []
        if kept_path.exists():
            with open(kept_path, "r") as f:
                kept_chunks = json.load(f)
        if removed_path.exists():
            with open(removed_path, "r") as f:
                kept_chunks += json.load(f)
            removed_path.unlink()

        with open(kept_path, "w") as f:
            json.dump(kept_chunks, f, indent=2)

        # UPDATED: Weighted centroid merge
        ca_arr = np.array(registry[kept].get("centroid_embedding", []), dtype=np.float32)
        cb_arr = np.array(registry[removed].get("centroid_embedding", []), dtype=np.float32)
        total = count_a + count_b
        merged = (ca_arr * count_a + cb_arr * count_b) / total
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged /= norm

        registry[kept]["chunk_count"] = total
        registry[kept]["centroid_embedding"] = merged.tolist()
        del registry[removed]
        save_registry(registry)

        # Regenerate summary post-merge
        if embedding_model is not None and chroma_client is not None and summary_fn is not None:
            registry = load_registry()
            update_slm_summary(registry, kept, embedding_model, chroma_client, summary_fn)
            save_registry(registry)

        merges.append({"kept": kept, "removed": removed, "similarity": round(best_score, 6)})

    return merges


# ── UPDATED: Migration ─────────────────────────────────────────────────

def migrate_centroids_to_summaries(
    embedding_model: SentenceTransformer,
    chroma_client,
) -> int:
    """
    Migrate existing SLM entries to the new centroid-based format.
    Recomputes centroid_embedding from chunk embeddings stored in ChromaDB.
    Removes stale TF-IDF artifacts (summary, last_summary_at).
    Regenerates summaries with Qwen for all migrated SLMs.
    Skips SLMs that already have centroid_embedding.

    Returns:
        Number of SLMs migrated.
    """
    registry = load_registry()
    migrated: List[str] = []

    for slm_name, entry in registry.items():
        if entry.get("centroid_embedding"):
            continue

        chunks_path = Path(entry.get("chunks_json", ""))
        if not chunks_path.exists():
            continue

        with open(chunks_path) as f:
            chunk_ids = [c["id"] for c in json.load(f)]
        if not chunk_ids:
            continue

        try:
            col = chroma_client.get_collection(entry["collection"])
        except Exception:
            continue

        embeddings: List = []
        for batch_ids in _batched(chunk_ids, CHROMA_GET_BATCH_SIZE):
            data = col.get(ids=batch_ids, include=["embeddings"])
            batch_embeddings = data.get("embeddings")
            if batch_embeddings is not None and len(batch_embeddings) > 0:
                embeddings.extend(batch_embeddings)
        if not embeddings:
            continue

        arr = np.array(embeddings, dtype=np.float32)
        centroid = arr.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        entry["centroid_embedding"] = centroid.tolist()
        entry.setdefault("topic_summary", "")
        entry.setdefault("keywords", [])
        entry.setdefault("summary_embedding", [])
        entry.pop("summary", None)
        entry.pop("last_summary_at", None)

        migrated.append(slm_name)

    if not migrated:
        return 0

    save_registry(registry)

    # Regenerate summaries for migrated SLMs
    registry = load_registry()
    for slm_name in migrated:
        update_slm_summary(registry, slm_name, embedding_model, chroma_client)
    save_registry(registry)

    return len(migrated)

def make_spacy_summary_fn(
    model_name: str = "it_core_news_lg",
    use_ner: bool = True,
    use_pos_nouns: bool = True,
) -> "SummaryFn":
    """
    Return a SummaryFn che estrae keyword con spaCy (NER + POS filter).

    Strategia:
      1. NER → entità coerenti (PER, ORG, GPE, LOC, MISC, PRODUCT, EVENT)
      2. POS filter → NOUN + PROPN non già catturati come entità
      3. _merge_keywords → dedup + substring absorption già presenti nel router
    """
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError:
        raise OSError(
            f"Modello spaCy '{model_name}' non trovato.\n"
            f"Installa con: python -m spacy download {model_name}"
        )

    print(f"[router] spaCy inizializzato con '{model_name}'.")

    # Label NER da tenere — esclude DATE, CARDINAL, ORDINAL, PERCENT (rumorose)
    _KEEP_LABELS = {"PER", "ORG", "GPE", "LOC", "MISC", "PRODUCT", "EVENT", "WORK_OF_ART"}

    def _extract_raw(texts: List[str]) -> List[str]:
        """Estrae le keyword grezze (con ripetizioni) da tutti i chunk."""
        all_raw: List[str] = []

        # Pulizia artefatti LaTeX/arXiv prima dell'estrazione: senza questa,
        # spaCy estrae @xmath/@xcite/righe di tabella come "entità".
        texts = [t for t in (_clean_arxiv_text(t) for t in texts) if t]

        # Processa tutti i chunk in batch (molto più veloce di uno alla volta)
        for doc in nlp.pipe(texts, batch_size=32, disable=["parser", "senter"]):

            if use_ner:
                for ent in doc.ents:
                    if ent.label_ in _KEEP_LABELS:
                        kw = ent.text.strip()
                        if _is_meaningful_kw(kw):
                            all_raw.append(kw)

            if use_pos_nouns:
                # Entità già estratte → non duplicare
                entity_spans = {ent.start for ent in doc.ents}
                for token in doc:
                    if token.i in entity_spans:
                        continue
                    if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                        kw = token.lemma_.strip()
                        if _is_meaningful_kw(kw):
                            all_raw.append(kw)

        return all_raw

    def _fn(texts: List[str]) -> Tuple[str, List[str]]:
        # Riusa _merge_keywords già presente nel router (dedup + absorption + freq sort)
        return "", _merge_keywords(_extract_raw(texts))

    # Variante con conteggi, usata dal flusso c-TF-IDF di refresh_all_summaries
    # (frequenza per keyword necessaria alla pesatura cross-cluster).
    _fn.with_counts = lambda texts: _merge_keywords_with_counts(_extract_raw(texts))

    return _fn

