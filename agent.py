"""
agent.py — ClusterAgent: per-cluster service exposing the CLARA DSL.

"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ── DSL types ──────────────────────────────────────────────────────────

# A Chunk is the unit of local memory exposed through the DSL.
# We keep it as a plain dict to match the rest of the project.
Chunk = Dict[str, Any]                  # {"id", "text", "embedding", "metadata"}
RankedChunk = Tuple[Chunk, float]       # (chunk, relevance_score)
EvidenceSet = List[Chunk]


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _rrf(rankings: List[List[int]], k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion of integer rankings → fused ranking."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ── ClusterAgent ───────────────────────────────────────────────────────

@dataclass
class ClusterAgent:
    name: str
    collection_name: str
    chunk_ids: List[str]
    keywords: List[str]
    summary_embedding: np.ndarray
    centroid_embedding: np.ndarray

    # Injected dependencies (not part of the persistent state).
    embedding_model: SentenceTransformer = field(repr=False)
    chroma_client: Any = field(repr=False)

    # Lazy-loaded local memory cache.
    _local_chunks: Optional[List[Chunk]] = field(default=None, init=False, repr=False)
    _bm25: Optional[BM25Okapi] = field(default=None, init=False, repr=False)
    _embeddings_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ── Construction from a registry entry ────────────────────────────

    @classmethod
    def from_registry_entry(
        cls,
        slm_name: str,
        entry: Dict,
        embedding_model: SentenceTransformer,
        chroma_client: Any,
    ) -> "ClusterAgent":
        chunks_path = Path(entry["chunks_json"])
        chunk_ids: List[str] = []
        if chunks_path.exists():
            with open(chunks_path, "r") as f:
                chunk_ids = [c["id"] for c in json.load(f)]

        return cls(
            name=slm_name,
            collection_name=entry["collection"],
            chunk_ids=chunk_ids,
            keywords=list(entry.get("keywords") or []),
            summary_embedding=np.array(entry.get("summary_embedding") or [], dtype=np.float32),
            centroid_embedding=np.array(entry.get("centroid_embedding") or [], dtype=np.float32),
            embedding_model=embedding_model,
            chroma_client=chroma_client,
        )

    # ── Local memory loader (locality invariant lives here) ───────────

    def _load_local_memory(self) -> List[Chunk]:
        if self._local_chunks is not None:
            return self._local_chunks

        if not self.chunk_ids:
            self._local_chunks = []
            self._embeddings_matrix = np.zeros((0,), dtype=np.float32)
            self._bm25 = BM25Okapi([[""]])
            return self._local_chunks

        col = self.chroma_client.get_collection(self.collection_name)
        data = col.get(
            ids=self.chunk_ids,
            include=["embeddings", "documents", "metadatas"],
        )
        embeddings = data.get("embeddings") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        ids = data.get("ids") or self.chunk_ids

        chunks: List[Chunk] = []
        emb_list: List[np.ndarray] = []
        for i, cid in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            emb = (
                np.array(embeddings[i], dtype=np.float32)
                if i < len(embeddings) and embeddings[i] is not None
                else np.zeros((1,), dtype=np.float32)
            )
            chunks.append({"id": cid, "text": text, "embedding": emb, "metadata": meta or {}})
            emb_list.append(emb)

        self._local_chunks = chunks
        self._embeddings_matrix = (
            np.vstack(emb_list) if emb_list else np.zeros((0,), dtype=np.float32)
        )
        # Tokenized corpus for BM25; protect against empty corpora.
        corpus = [_tokenize(c["text"]) for c in chunks] or [[""]]
        self._bm25 = BM25Okapi(corpus)
        return self._local_chunks

    # ── DSL operation 1: local_retrieve ───────────────────────────────

    def local_retrieve(self, query: str, top_k: int = 20) -> List[RankedChunk]:

        chunks = self._load_local_memory()
        if not chunks:
            return []

        q_emb = self.embedding_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

        # Dense ranking (cosine over normalized embeddings).
        emb_mat = self._embeddings_matrix
        if emb_mat.ndim == 2 and emb_mat.shape[0] == len(chunks):
            # Both q_emb and stored embeddings are L2-normalized.
            dense_scores = emb_mat @ q_emb
        else:
            dense_scores = np.array(
                [
                    float(np.dot(q_emb, c["embedding"]))
                    if c["embedding"].shape == q_emb.shape
                    else 0.0
                    for c in chunks
                ],
                dtype=np.float32,
            )
        dense_rank = sorted(
            range(len(chunks)), key=lambda i: float(dense_scores[i]), reverse=True
        )

        # BM25 ranking (sparse).
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_rank = sorted(
            range(len(chunks)), key=lambda i: float(bm25_scores[i]), reverse=True
        )

        # RRF fusion → single ranking; score reported for inspection is the dense score.
        fused = _rrf([dense_rank, bm25_rank])
        out: List[RankedChunk] = []
        for idx in fused[:top_k]:
            out.append((chunks[idx], float(dense_scores[idx])))
        return out

    # ── DSL operation 2: select_evidence ──────────────────────────────

    def select_evidence(
        self,
        query: str,
        ranked: List[RankedChunk],
        tau_size: int = 5,
        tau_rel: float = 0.0,
    ) -> EvidenceSet:

        del query  # unused: ranking already encodes query relevance
        evidence: EvidenceSet = []
        for chunk, score in ranked:
            if score < tau_rel:
                continue
            evidence.append(chunk)
            if len(evidence) >= tau_size:
                break
        return evidence

    # ── DSL operation 3: provide_evidence (atomic; orchestrator → secondary) ──

    def provide_evidence(
        self,
        query: str,
        tau_size: int = 3,
        tau_rel: float = 0.0,
        retrieve_top_k: int = 10,
    ) -> EvidenceSet:

        ranked = self.local_retrieve(query, top_k=retrieve_top_k)
        return self.select_evidence(query, ranked, tau_size=tau_size, tau_rel=tau_rel)

    # ── DSL operation 4: generate (primary-only) ──────────────────────

    def generate(
        self,
        query: str,
        gamma: EvidenceSet,
        slm_runtime: Tuple[Any, Any],
        *,
        role: str = "primary",
        max_new_tokens: int = 256,
        device: Optional[str] = None,
    ) -> str:

        if role != "primary":
            raise RuntimeError(
                f"generate() requires invocation_role = 'primary' but got '{role}'. "
                "Secondary agents act as evidence providers only."
            )

        import torch
        # Local import keeps agent.py importable without transformers if generate is unused.
        from SLMAgent import build_messages, _apply_chat_template_no_think

        tokenizer, model = slm_runtime
        if device is None:
            device = next(model.parameters()).device

        context_chunks = [c["text"] for c in gamma]
        text = _apply_chat_template_no_think(tokenizer, build_messages(query, context_chunks))
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # ── Introspection ─────────────────────────────────────────────────

    def memory_size(self) -> int:
        """|M_j| — number of chunks in the agent's local memory."""
        return len(self.chunk_ids)

    def __repr__(self) -> str:
        return (
            f"ClusterAgent(name={self.name!r}, "
            f"|M_j|={len(self.chunk_ids)}, "
            f"|keywords|={len(self.keywords)})"
        )
