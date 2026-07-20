"""
chunking.py — PDF parsing and adaptive chunking via LangChain.

Document type is auto-detected from content heuristics.
Each type uses tuned chunk_size and overlap parameters.
Chapter structure is preserved from the PDF as metadata.
"""

import uuid
import re
from typing import List, Dict, Tuple

import pymupdf4llm
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Chunking profiles per document type ───────────────────────────────

CHUNKING_PROFILES = {
    "paper": {
        # Scientific papers: dense, short sections, high information per sentence
        "chunk_size":    800,
        "chunk_overlap": 100,
    },
    "book": {
        # Books/textbooks: longer narrative paragraphs, lower density
        "chunk_size":    1500,
        "chunk_overlap": 150,
    },
    "technical": {
        # Technical docs, manuals, API docs: mixed prose and code/specs
        "chunk_size":    1200,
        "chunk_overlap": 200,
    },
    "generic": {
        # Blogs, articles, emails, unknown format
        "chunk_size":    1500,
        "chunk_overlap": 150,
    },
}


# ── Document type detection ────────────────────────────────────────────

def detect_doc_type(text: str) -> str:
    """
    Heuristic detection of document type from full text.

    Returns one of: 'paper', 'book', 'technical', 'generic'.
    """
    sample = text[:8000].lower()

    # Count heuristic signals
    paper_signals = sum([
        bool(re.search(r'\babstract\b', sample)),
        bool(re.search(r'\bintroduction\b', sample)),
        bool(re.search(r'\bconclusion\b', sample)),
        bool(re.search(r'\breferences\b', sample)),
        bool(re.search(r'\bmethodolog', sample)),
        bool(re.search(r'\brelated work\b', sample)),
        bool(re.search(r'\bexperiment', sample)),
        bool(re.search(r'\bdoi\b|\barxiv\b', sample)),
        len(re.findall(r'\[\d+\]', sample)) > 3,       # citation markers [1], [2]
        len(re.findall(r'\d+\.\s+[a-z]', sample)) > 4, # numbered sections
    ])

    book_signals = sum([
        bool(re.search(r'\bchapter\b', sample)),
        bool(re.search(r'\bpreface\b|\bforeword\b', sample)),
        len(re.findall(r'\n\n', text[:8000])) > 20,    # many paragraph breaks
        len(text) > 50000,                              # long document
    ])

    technical_signals = sum([
        bool(re.search(r'\bapi\b|\bendpoint\b', sample)),
        bool(re.search(r'def |class |function |import |```', sample)),
        bool(re.search(r'\bparameter[s]?\b|\bargument[s]?\b', sample)),
        bool(re.search(r'\binstallation\b|\bconfiguration\b', sample)),
        len(re.findall(r'`[^`]+`', text[:8000])) > 5,  # inline code
    ])

    scores = {
        "paper":     paper_signals,
        "book":      book_signals,
        "technical": technical_signals,
    }

    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 2:
        return best
    return "generic"


def get_chunker(doc_type: str) -> Tuple[RecursiveCharacterTextSplitter, Dict]:
    """
    Return a LangChain splitter configured for the given document type.
    Also returns the profile dict for logging.
    """
    profile = CHUNKING_PROFILES.get(doc_type, CHUNKING_PROFILES["generic"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=profile["chunk_size"],
        chunk_overlap=profile["chunk_overlap"],
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    return splitter, profile


# ── Chapter extraction ─────────────────────────────────────────────────

_MD_IMAGE = re.compile(r'!\[.*?\]\(.*?\)')  # strips ![alt](url) from markdown output


def extract_chapters(pdf_path: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Extract text from PDF using pymupdf4llm (Markdown output), split into
    chapters using heading patterns. Falls back to a single 'Document' section
    if no headings are found.

    Returns (chapters, full_text).
    """
    full_text = pymupdf4llm.to_markdown(pdf_path)
    full_text = _MD_IMAGE.sub("", full_text)  # remove image references

    chapter_pattern = re.compile(
        r'(?m)^#{1,3}\s+\*{0,2}'           # ## o ## ** (con bold opzionale)
        r'('
        r'\d{1,2}\.\s+[A-ZÀÈÌÒÙ][^\n]{3,}' # "1. IL 1300" — solo primo livello
        r'|[A-ZÀÈÌÒÙ]{3,}[^\n]+'           # "L'ETÀ DELLA RINASCITA" — tutto maiuscolo
        r')',
    )
    matches = list(chapter_pattern.finditer(full_text))

    if not matches:
        return [{"chapter": "Document", "text": full_text.strip()}], full_text

    chapters = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            chapters.append({"chapter": title, "text": text})

    return chapters, full_text


# ── Main entry point ───────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    chunk_size: int = None,
    overlap: int = None,
) -> Tuple[List[Dict], str, str]:
    """
    Parse PDF, auto-detect document type, chunk adaptively with LangChain,
    embed in batch, and store in ChromaDB.

    Args:
        chunk_size: if provided, overrides the adaptive profile (manual override).
        overlap:    if provided, overrides the adaptive profile (manual override).

    Returns:
        (chunks, doc_type, profile_summary)
        chunks: list of dicts with keys id, text, chapter, embedding.
        doc_type: detected document type string.
        profile_summary: human-readable description of parameters used.
    """
    chapters, full_text = extract_chapters(pdf_path)

    # Auto-detect document type
    doc_type = detect_doc_type(full_text)
    splitter, profile = get_chunker(doc_type)

    # Manual override if explicit values provided
    if chunk_size is not None or overlap is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or profile["chunk_size"],
            chunk_overlap=overlap or profile["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        profile = {
            "chunk_size":    chunk_size or profile["chunk_size"],
            "chunk_overlap": overlap or profile["chunk_overlap"],
        }

    profile_summary = (
        f"tipo={doc_type}  "
        f"chunk_size={profile['chunk_size']}  "
        f"overlap={profile['chunk_overlap']}"
    )

    _index_noise = re.compile(
        r'(INDICE\s+volume|Aula\s+Virtuale|Glossario|pag\.\s*\d{2,3}\s*$)',
        re.IGNORECASE | re.MULTILINE,
    )

    # Split each chapter
    raw_chunks: List[Dict[str, str]] = []
    for chapter in chapters:
        splits = splitter.split_text(chapter["text"])
        for text in splits:
            text = text.strip()
            if text and not _index_noise.search(text):
                raw_chunks.append({"chapter": chapter["chapter"], "text": text})
    print(f"[chunking] {len(raw_chunks)} chunk validi,")
   
    if not raw_chunks:
        return [], doc_type, profile_summary

    # Batch embedding
    texts = [c["text"] for c in raw_chunks]
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
    )

    chunks = []
    for i, raw in enumerate(raw_chunks):
        chunks.append({
            "id":        str(uuid.uuid4()),
            "text":      raw["text"],
            "chapter":   raw["chapter"],
            "embedding": embeddings[i].tolist(),
        })

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"chapter": c["chapter"]} for c in chunks],
    )

    return chunks, doc_type, profile_summary


# ── Plain-text entry point (no PDF/OCR parsing) ────────────────────────

def process_text(
    text: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    chapter_label: str = "Document",
    doc_type: str = "paper",
    chunk_size: int = None,
    overlap: int = None,
) -> Tuple[List[Dict], str, str]:
    """
    Chunk and embed text that has already been extracted elsewhere (e.g. a
    dataset field), skipping the PDF/Markdown parsing done by process_pdf.

    Args:
        chapter_label: value stored as the 'chapter' metadata for every chunk
                        produced from this text (there is no heading structure
                        to derive it from, unlike extract_chapters()).
        doc_type:       chunking profile to use (see CHUNKING_PROFILES); defaults
                        to 'paper' since this entry point is used for scientific
                        paper corpora.

    Returns:
        (chunks, doc_type, profile_summary) — same shape as process_pdf.
    """
    splitter, profile = get_chunker(doc_type)

    if chunk_size is not None or overlap is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or profile["chunk_size"],
            chunk_overlap=overlap or profile["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        profile = {
            "chunk_size":    chunk_size or profile["chunk_size"],
            "chunk_overlap": overlap or profile["chunk_overlap"],
        }

    profile_summary = (
        f"tipo={doc_type}  "
        f"chunk_size={profile['chunk_size']}  "
        f"overlap={profile['chunk_overlap']}"
    )

    raw_chunks = [t.strip() for t in splitter.split_text(text)]
    raw_chunks = [t for t in raw_chunks if t]
    if not raw_chunks:
        return [], doc_type, profile_summary

    embeddings = embedding_model.encode(
        raw_chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )

    chunks = []
    for i, raw_text in enumerate(raw_chunks):
        chunks.append({
            "id":        str(uuid.uuid4()),
            "text":      raw_text,
            "chapter":   chapter_label,
            "embedding": embeddings[i].tolist(),
        })

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"chapter": c["chapter"]} for c in chunks],
    )

    return chunks, doc_type, profile_summary
