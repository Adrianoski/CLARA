"""
Benchmark script for comparing standard RAG retrieval vs SLM-routed retrieval.
"""

import time
import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from router import load_registry, find_top_n_slms

# ── Config ─────────────────────────────────────────────────────────────
TOP_N_SLMS        = 15
TOP_K             = 5
RRF_K             = 60
ROUTING_MODE      = "summary"             
GENERATION_MODEL  = "Qwen/Qwen3.5-4B"    
MAX_NEW_TOKENS    = 256
MAX_INPUT_TOKENS  = 16384                  
                                           
                                           
                                           
OUTPUT_FILE       = f"{MAX_INPUT_TOKENS}_benchmark_answers_spacy_4B_3_WAYS_arxiv2000_{TOP_N_SLMS}_SLM_TOP_K_{TOP_K}_ROUTING_{ROUTING_MODE}.md"


QUERIES = [
    (
        "paper_000045",
        "What are Compact Symmetric Objects (CSOs) and what is their defining characteristic?",
        ["compact symmetric objects", "extragalactic radio sources", "central engine",
         "terminal hotspots", "subluminal speeds", "superluminal speeds"],
        (
            "Compact Symmetric Objects are a family of extragalactic radio sources "
            "(about 3% of flux-limited high-frequency samples) whose defining "
            "characteristic is high-luminosity radio components on both sides of a "
            "central engine at sub-kiloparsec scales, with little or no extended "
            "emission. CSOs show terminal hotspots moving apart at subluminal "
            "speeds, while jet components between the core and the hotspots move "
            "faster — occasionally at superluminal speeds."
        ),
    ),
    (
        "paper_000057",
        "What is the Waldmeier effect in solar cycle research?",
        ["waldmeier effect", "sunspot number", "anti - correlation", "rise time",
         "grand minima", "solar cycle"],
        (
            "The sunspot number varies with an average 11-year period, but "
            "individual cycle length and amplitude vary randomly, with stronger "
            "cycles tending to have shorter periods. The Waldmeier effect "
            "describes an anti-correlation between the rise time to maximum and "
            "the peak sunspot number (WE1), while the rise rate correlates tightly "
            "and positively with the peak number (WE2). Grand minima are extended "
            "periods of strongly reduced solar activity, such as 1645-1715."
        ),
    ),
    (
        "paper_000084",
        "What is an interval exchange transformation and how is it related to translation surfaces?",
        ["interval exchange", "translation surface", "abelian differentials",
         "riemann surfaces", "rauzy - veech", "unique ergodicity"],
        (
            "A geodesic flow in a given direction on a translation surface induces "
            "an interval exchange transformation on a transverse segment; these "
            "transformations are closely related to abelian differentials on "
            "Riemann surfaces. The Rauzy-Veech induction gives a discrete model "
            "for the Teichmüller geodesic flow, analogous to the Euclidean "
            "algorithm, and was used by Masur and Veech to independently prove "
            "Keane's conjecture on the unique ergodicity of almost all interval "
            "exchange transformations."
        ),
    ),
    (
        "paper_000111",
        "What indirect evidence existed for gravitational waves before their direct detection?",
        ["gravitational waves", "psr 1913 + 16", "orbital decay", "general relativity",
         "neutron star", "nobel - prize"],
        (
            "Before direct detection, indirect evidence for gravitational waves "
            "came from high-precision, Nobel-prize-winning measurements of the "
            "binary pulsar PSR 1913+16 and its companion neutron star: the "
            "orbital decay driven by gravitational-wave emission matched the "
            "predictions of general relativity to better than 1%."
        ),
    ),
    (
        "paper_000219",
        "How does resonant excitation improve the spectral linewidth of photons emitted by a quantum dot?",
        ["spectral linewidth", "resonantly driven", "photoluminescence",
         "cw lasers", "above saturation", "single qd"],
        (
            "Resonantly driving a single quantum dot yields a significantly "
            "narrower emitted-photon linewidth (about 0.48 GHz) than incoherent "
            "CW excitation methods such as above-bandgap (about 2.5 GHz) or "
            "p-shell (about 1.5 GHz) excitation, even at high power well above "
            "saturation."
        ),
    ),
    (
        "paper_000240",
        "How are Active Galactic Nuclei (AGN) classified into type 1 and type 2?",
        ["broad emission lines", "optical spectra", "agn1", "agn2",
         "hard x - ray telescopes", "soft x - rays"],
        (
            "AGN were first discovered in radio and then classified by their "
            "optical characteristics into type 1 (AGN1, broad emission lines "
            "present in the optical spectrum) and type 2 (AGN2, no broad lines). "
            "AGN1 samples selected in the optical or soft X-rays (by missions "
            "like Einstein and ROSAT) had well-measured evolution, while AGN2 "
            "samples were difficult to build at any wavelength before the advent "
            "of hard X-ray telescopes."
        ),
    ),
    (
        "paper_000243",
        "What physical parameters of white dwarfs can be measured from spectroscopy, and why is this harder in cataclysmic variables?",
        ["effective temperature", "surface gravity", "mass - radius relation",
         "cataclysmic variables", "polars", "synchronously rotating"],
        (
            "Effective temperature, surface gravity and magnetic field strength "
            "of field white dwarfs can be measured precisely from spectroscopy, "
            "and assuming a mass-radius relation, mass and radius follow "
            "independently of distance. Determining these properties for "
            "accreting white dwarfs in cataclysmic variables — including polars, "
            "which contain a synchronously rotating magnetic white dwarf — is a "
            "newer and harder research field, and little is known about their "
            "temperatures despite many known systems."
        ),
    ),
    (
        "paper_000276",
        "What key physics issues are addressed by studying nuclei far from the stability region?",
        ["drip lines", "radioactive beam", "shell gap", "neutron - rich",
         "weakly bound", "unstable nuclei"],
        (
            "Radioactive beam facilities producing neutron-rich or "
            "neutron-deficient nuclei let researchers map the neutron and proton "
            "drip lines, understand continuum effects on weakly bound nuclear "
            "systems, shell-gap modifications in very neutron-rich systems, "
            "nuclear properties relevant to astrophysics, and deformation, spin, "
            "pairing properties and unusual shapes in unstable nuclei."
        ),
    ),
    (
        "paper_000288",
        "Why is protecting stored biometric templates especially important compared to passwords?",
        ["biometric authentication", "biometric template", "one - to - one matching",
         "compromised", "fingerprint", "iris"],
        (
            "In biometric authentication systems, a submitted sample is matched "
            "one-to-one against a stored biometric template. Unlike a password, "
            "a compromised biometric trait (e.g. face, fingerprint, or iris) "
            "cannot be replaced, since biometric traits are considered unique — "
            "so poor protection of biometric templates raises serious security "
            "and privacy concerns."
        ),
    ),
    (
        "paper_000312",
        "What is the Neupert effect and how is it explained by the thick-target model?",
        ["neupert effect", "thick - target model", "nonthermal electrons",
         "bremsstrahlung", "lower corona", "hard x - ray"],
        (
            "The Neupert effect is the observation that the rising part of a "
            "solar flare's soft X-ray light curve resembles the time integral of "
            "the hard X-ray/microwave emission, interpreted as a causal link "
            "between thermal and nonthermal flare emission. The nonthermal "
            "thick-target model explains this: flare energy is released mainly "
            "as nonthermal electrons, and hard X-rays are produced via "
            "electron-ion bremsstrahlung as the electron beam hits the lower "
            "corona and chromosphere."
        ),
    ),
    (
        "paper_000384",
        "Why has quantum electronic transport in one-dimensional structures received renewed theoretical interest?",
        ["quantum electronic transport", "point contacts", "carbon nanotubes",
         "quantum wires", "mesoscopic systems", "phase coherence"],
        (
            "Continuous progress in fabricating small electronic circuits (point "
            "contacts, atomic chains, carbon nanotubes, quantum wires) has "
            "renewed theoretical interest in one-dimensional quantum electronic "
            "transport, building on decades of study of mesoscopic systems where "
            "electron phase coherence is preserved along the whole device; "
            "disorder from random impurity positions gives transport a "
            "stochastic character."
        ),
    ),
    (
        "paper_000393",
        "Why are standard database spatial indexes considered inadequate for astronomical queries?",
        ["spatial indexes", "relational databases", "celestial objects",
         "mcs library", "dif package", "astronomical use"],
        (
            "Built-in spatial indexes offered by relational database servers for "
            "coordinate columns are hard to use, follow a syntax different from "
            "astronomical convention, and perform inadequately for astronomical "
            "use on data like celestial coordinates. The MCS library project "
            "implemented the 'dif' package, a tool that automatically manages "
            "this kind of spatial indexing for astronomical data."
        ),
    ),
    (
        "paper_000414",
        "How do cohesive zone models (CZMs) represent material failure along surfaces?",
        ["cohesive zone models", "continuum mechanics", "material failure",
         "constitutive laws", "discontinuities", "traction"],
        (
            "Cohesive zone models are surface failure models within continuum "
            "mechanics: the continuum is enriched with discontinuities along "
            "cohesive-zone surfaces governed by traction-displacement-separation "
            "constitutive laws, where traction increases to a maximum and then "
            "decreases to zero with increasing separation. CZMs are natural when "
            "the location of the separation surface — e.g. fracture along a weak "
            "interface — is known in advance."
        ),
    ),
    (
        "paper_000432",
        "What is IEEE 802.15.4 designed for in the context of wireless body area networks?",
        ["wireless body area networks", "ieee 802.15.4", "low power",
         "wireless sensor networks", "physical layer", "patient s health"],
        (
            "IEEE 802.15.4 targets wireless body area networks (WBANs), which "
            "require low power and low data rate applications. It defines the "
            "physical layer and MAC sublayer to provide a low-cost, low-power, "
            "reliable protocol for applications such as wireless monitoring of "
            "patient health, wearable computing, and location identification."
        ),
    ),
    (
        "paper_000489",
        "What are the two phase transitions hadronic matter is expected to undergo at large temperature or density?",
        ["hadronic matter", "chiral symmetry", "crossover transitions",
         "heavy ion physics", "neutron stars", "supernuclear densities"],
        (
            "At large temperature or density, hadronic matter is expected to "
            "undergo a deconfinement transition (quarks and gluons no longer "
            "confined) and a chiral-symmetry-restoring transition; whether these "
            "coincide or are distinct — or even real phase transitions rather "
            "than crossovers — is unsettled. This matters for neutron stars, "
            "which offer a unique environment to study cold matter at "
            "supernuclear densities, though confirming a deconfined quark phase "
            "there is limited by uncertainties in modeling QCD at large "
            "densities."
        ),
    ),
    (
        "paper_000561",
        "Why might the ZEUS MHD code be expected to perform worse than upwind conservative codes on Riemann problems?",
        ["mhd code", "sod problem", "riemann problems", "upwind",
         "conservative codes", "brio and wu"],
        (
            "ZEUS is a freely available, widely used astrophysical MHD code, but "
            "unlike codes tested on problems such as the Brio & Wu MHD Riemann "
            "problem, ZEUS is neither upwind for all characteristic fields nor "
            "conservative — so it can be expected, and is shown, to perform "
            "significantly worse than upwind conservative codes on a range of "
            "simple Riemann test problems."
        ),
    ),
    (
        "paper_000612",
        "What is the color superconducting phase in cold, dense QCD matter and who first proposed it?",
        ["finite baryon density", "color superconducting phase", "color antitriplet channel",
         "bcs theory", "cold fermi sea", "cooper - pair condensate"],
        (
            "Sufficiently cold and dense baryonic matter is expected to be in a "
            "color superconducting phase, first proposed decades ago by "
            "Frautschi and Barrois based on the observation that one-gluon "
            "exchange between two quarks is attractive in the color-antitriplet "
            "channel. By BCS theory, a weak attractive interaction in a cold "
            "Fermi sea drives instability toward Cooper-pair condensation — "
            "central to understanding the cores of compact stars."
        ),
    ),
    (
        "paper_000621",
        "Why is entanglement considered fragile in realistic, macroscopic quantum systems?",
        ["quantum information processing", "long - distance quantum communication",
         "distributed quantum computation", "environment induced decoherence",
         "biological systems", "macroscopic"],
        (
            "Entanglement is the key resource that can make quantum information "
            "processing and long-distance quantum communication more powerful "
            "than classical approaches, but it is fragile under "
            "environment-induced decoherence. As systems grow toward macroscopic "
            "scale — gases, fluids, solids, or biological systems — they become "
            "open and noisy, threatening the entanglement engineering tries to "
            "preserve."
        ),
    ),
    (
        "paper_000708",
        "What is the difference between SIS and SIR epidemic models, and why do they show non-trivial dynamics despite being simple?",
        ["sis system", "sir epidemics", "susceptible individuals",
         "non - trivial equilibria", "bifurcations", "critical fluctuations"],
        (
            "Epidemic models are classically ODE systems for a host population "
            "split into susceptible and infected classes (SIS), or additionally "
            "a recovered class conferring immunity (SIR). The nonlinear "
            "infection term — a product of two population variables — means "
            "even these simple models show non-trivial equilibria arising from "
            "bifurcations, and stochastic versions show critical fluctuations at "
            "the epidemic threshold."
        ),
    ),
    (
        "paper_000750",
        "Why was it historically assumed that halo white dwarfs contribute negligibly to the total mass of the galaxy?",
        ["stellar remnants", "main sequence stars", "halo wds",
         "cno elements", "interstellar gas", "total mass"],
        (
            "White dwarfs are the most common stellar remnants and trace early "
            "galactic evolution through their density, distribution, colors and "
            "ages. Halo white dwarfs were assumed to contribute negligible mass "
            "to the galaxy partly because forming a roughly 0.6-solar-mass white "
            "dwarf releases several solar masses of gas heavily enriched in CNO "
            "elements, and local stars plus interstellar gas — which would carry "
            "such enrichment — make up only a few percent of the galaxy's total "
            "mass budget."
        ),
    ),
]
# ── Helpers ────────────────────────────────────────────────────────────

def tokenize(text: str):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def rrf(rankings, k=RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def cosine_scores(q_emb, embeddings):
    q_norm = float(np.linalg.norm(q_emb))
    scores = []
    for emb in embeddings:
        e = np.array(emb, dtype=np.float32)
        e_norm = float(np.linalg.norm(e))
        scores.append(float(np.dot(q_emb, e) / (q_norm * e_norm)) if e_norm > 0 else 0.0)
    return scores


# ── RAG standard: cerca su TUTTI i chunk di TUTTE le collection ────────

def retrieve_standard(query, q_emb, chroma_client, top_k=TOP_K):
    t0 = time.perf_counter()

    ids, embeddings, documents = [], [], []
    for col_info in chroma_client.list_collections():
        col  = chroma_client.get_collection(col_info.name)
        data = col.get(include=["embeddings", "documents"])
        ids.extend(data["ids"])
        embeddings.extend(data["embeddings"])
        documents.extend(data["documents"])

    dense_scores = cosine_scores(q_emb, embeddings)
    dense_rank   = sorted(range(len(embeddings)), key=lambda i: dense_scores[i], reverse=True)

    bm25       = BM25Okapi([tokenize(d) for d in documents])
    bm25_sc    = bm25.get_scores(tokenize(query))
    bm25_rank  = sorted(range(len(embeddings)), key=lambda i: float(bm25_sc[i]), reverse=True)

    fused = rrf([dense_rank, bm25_rank])[:top_k]
    elapsed = (time.perf_counter() - t0) * 1000

    # Gli id dei top-k servono al calcolo del routing recall@N
    # (in quale cluster vivono i chunk che StdRAG considera migliori).
    return [documents[i] for i in fused], [ids[i] for i in fused], len(embeddings), elapsed


# ── SLM-routed retrieval ───────────────────────────────────────────────

def retrieve_slm(query, q_emb, registry, chroma_client, top_k=TOP_K):
    t0 = time.perf_counter()

    top_slms = find_top_n_slms(q_emb, registry, n=TOP_N_SLMS, mode=ROUTING_MODE)

    all_chunks = []
    for slm_name, _ in top_slms:
        entry = registry[slm_name]
        chunks_path = Path(entry["chunks_json"])
        if not chunks_path.exists():
            continue
        with open(chunks_path) as f:
            chunk_ids = [c["id"] for c in json.load(f)]
        col = chroma_client.get_collection(entry["collection"])
        data = col.get(ids=chunk_ids, include=["embeddings", "documents", "metadatas"])
        for i, emb in enumerate(data["embeddings"]):
            e = np.array(emb, dtype=np.float32)
            e_norm = float(np.linalg.norm(e))
            q_norm = float(np.linalg.norm(q_emb))
            ds = float(np.dot(q_emb, e) / (q_norm * e_norm)) if e_norm > 0 else 0.0
            all_chunks.append({"text": data["documents"][i], "score": ds})

    if not all_chunks:
        return [], 0, (time.perf_counter() - t0) * 1000

    dense_rank = sorted(range(len(all_chunks)), key=lambda i: all_chunks[i]["score"], reverse=True)
    bm25       = BM25Okapi([tokenize(c["text"]) for c in all_chunks])
    bm25_sc    = bm25.get_scores(tokenize(query))
    bm25_rank  = sorted(range(len(all_chunks)), key=lambda i: float(bm25_sc[i]), reverse=True)

    fused   = rrf([dense_rank, bm25_rank])[:top_k]
    elapsed = (time.perf_counter() - t0) * 1000

    return [all_chunks[i]["text"] for i in fused], len(all_chunks), elapsed


# ── SLM-Full: routing + tutti i chunk del cluster (no retrieval interno) ──

def retrieve_slm_full(q_emb, registry, chroma_client):
    """Solo routing semantico: passa al LLM TUTTI i chunk dei top-N SLM,
    senza ulteriore filtraggio dense/BM25 intra-cluster."""
    t0 = time.perf_counter()

    top_slms = find_top_n_slms(q_emb, registry, n=TOP_N_SLMS, mode=ROUTING_MODE)

    docs = []
    for slm_name, _ in top_slms:
        entry = registry[slm_name]
        chunks_path = Path(entry["chunks_json"])
        if not chunks_path.exists():
            continue
        with open(chunks_path) as f:
            chunk_ids = [c["id"] for c in json.load(f)]
        col = chroma_client.get_collection(entry["collection"])
        data = col.get(ids=chunk_ids, include=["documents"])
        docs.extend(data["documents"])

    elapsed = (time.perf_counter() - t0) * 1000
    return docs, len(docs), elapsed


# ── Generation ────────────────────────────────────────────────────────

_gen_tok = None
_gen_mdl = None

def _load_gen_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    global _gen_tok, _gen_mdl
    if _gen_tok is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.int8 if device == "cuda" else torch.float32
        print(f"  Carico {model_name} su {device} (max_input_tokens={MAX_INPUT_TOKENS})...")
        _gen_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _gen_tok.model_max_length = MAX_INPUT_TOKENS
        _gen_mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()


# Budget in caratteri per i chunk passati al generatore (~3.5 char/token),
# con margine per system prompt, template e query. Troncare la LISTA di chunk
# prima del template — invece di lasciare che tokenizer(truncation=True) tagli
# la coda del prompt — preserva il chat template: senza questo fix, con pool
# molto grandi (SLM-Full) il marcatore finale del turno assistant veniva
# amputato e il modello produceva output degeneri (loop, 'assistant <think>'
# leakage nelle risposte).
_DOCS_CHAR_BUDGET = int(3.5 * (MAX_INPUT_TOKENS - 1024))


def _fit_docs_to_budget(docs: list, budget: int = _DOCS_CHAR_BUDGET) -> list:
    fitted, used = [], 0
    for d in docs:
        if used + len(d) > budget:
            break
        fitted.append(d)
        used += len(d)
    # almeno un chunk (eventualmente accorciato) per non mandare un prompt vuoto
    if not fitted and docs:
        fitted = [docs[0][:budget]]
    return fitted


def generate(query: str, docs: list, model_name: str) -> str:
    import torch
    from SLMAgent import build_messages, _apply_chat_template_no_think
    _load_gen_model(model_name)
    device = next(_gen_mdl.parameters()).device
    docs = _fit_docs_to_budget(docs)
    # apply_chat_template with enable_thinking=False prevents <think> tokens at the source.
    text = _apply_chat_template_no_think(_gen_tok, build_messages(query, docs))
    # truncation=True resta solo come cintura di sicurezza: dopo il budget sui
    # chunk il prompt dovrebbe già stare in MAX_INPUT_TOKENS.
    inputs = _gen_tok(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    with torch.no_grad():
        out = _gen_mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_gen_tok.eos_token_id,
        )
    answer = _gen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del inputs, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return answer


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("Carico modelli...")
    emb_model    = SentenceTransformer("BAAI/bge-m3")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    registry     = load_registry()

    all_cols     = chroma_client.list_collections()
    total_chunks = sum(chroma_client.get_collection(c.name).count() for c in all_cols)
    col_names    = [c.name for c in all_cols]

    print(f"Collections: {col_names}  |  chunk totali: {total_chunks}  |  SLM: {len(registry)}")
    print(f"Query: {len(QUERIES)}  |  top-k={TOP_K}  |  top-N SLM={TOP_N_SLMS}  |  model={GENERATION_MODEL}\n")

    def hit_rate(docs, keywords):
        joined = " ".join(docs).lower()
        hits = sum(1 for kw in keywords if kw in joined)
        return hits / len(keywords) * 100 if keywords else 0.0

    # Mappa chunk_id → SLM, per il routing recall@N
    chunk2slm = {}
    for slm_name, entry in registry.items():
        chunks_path = Path(entry["chunks_json"])
        if not chunks_path.exists():
            continue
        with open(chunks_path) as f:
            for c in json.load(f):
                chunk2slm[c["id"]] = slm_name

    def routing_recall(std_top_ids, q_emb, at_n):
        """Frazione dei top-k chunk di StdRAG il cui cluster è nel top-N del routing."""
        top = find_top_n_slms(q_emb, registry, n=at_n, mode=ROUTING_MODE)
        top_names = {name for name, _ in top}
        if not std_top_ids:
            return 0.0
        in_top = sum(1 for cid in std_top_ids if chunk2slm.get(cid) in top_names)
        return in_top / len(std_top_ids) * 100

    def classify_answer(ans: str) -> str:
        """RISPOSTA = risposta vera | NON-DISPONIBILE = rifiuto onesto | DEGENERE = output rotto."""
        a = ans.lower()
        if "assistant" in a[:400] or a.count("becomes eq") > 1 or "@xc0000" in a:
            return "DEGENERE"
        if "not available" in a or "does not contain" in a or "not contain" in a:
            return "NON-DISPONIBILE"
        return "RISPOSTA"

    std_times, slm_times, full_times = [], [], []
    std_pools, slm_pools, full_pools = [], [], []
    std_gens,  slm_gens,  full_gens  = [], [], []
    overlaps, std_hits, slm_hits, full_hits = [], [], [], []
    recalls3, recalls5 = [], []
    md_rows = []

    for idx, (paper_id, query, expected_kws, ground_truth) in enumerate(QUERIES, 1):
        print(f"\n[{idx}/{len(QUERIES)}] ({paper_id}) {query}")

        # Documenti e query sono entrambi in inglese (corpus arXiv), quindi
        # un'unica query serve sia per il retrieval (dense + BM25) sia per la
        # generazione — a differenza della vecchia coppia EN/IT usata per i
        # PDF italiani.
        q_emb = emb_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

        docs_std, ids_std, pool_std, t_std = retrieve_standard(query, q_emb, chroma_client)
        docs_slm,  pool_slm,  t_slm  = retrieve_slm(query, q_emb, registry, chroma_client)

        # SLM-Full disattivato: con i cluster attuali (centinaia-migliaia di
        # chunk) il pool senza reranking supera di gran lunga il contesto del
        # generatore e viene troncato ad-hoc, producendo risposte quasi sempre
        # sbagliate (1/20 nei run precedenti) e allungando di molto ogni query
        # (generazione più lenta di StdRAG e SLM-RAG messi insieme). Placeholder
        # a costo zero sotto, così il resto dello script (aggregati, markdown,
        # JSON) continua a funzionare senza modifiche.
        # docs_full, pool_full, t_full = retrieve_slm_full(q_emb, registry, chroma_client)
        docs_full, pool_full, t_full = [], 0, 0.0

        overlap = len(set(docs_std[:TOP_K]) & set(docs_slm[:TOP_K])) / TOP_K * 100
        hr_std  = hit_rate(docs_std,  expected_kws)
        hr_slm  = hit_rate(docs_slm,  expected_kws)
        hr_full = hit_rate(docs_full, expected_kws)
        rec3 = routing_recall(ids_std, q_emb, 3)
        rec5 = routing_recall(ids_std, q_emb, 5)

        std_times.append(t_std);   slm_times.append(t_slm);   full_times.append(t_full)
        std_pools.append(pool_std); slm_pools.append(pool_slm); full_pools.append(pool_full)
        overlaps.append(overlap)
        std_hits.append(hr_std);   slm_hits.append(hr_slm);   full_hits.append(hr_full)
        recalls3.append(rec3);     recalls5.append(rec5)
        print(f"  Routing recall — @3: {rec3:.0f}%  @5: {rec5:.0f}%  (mode={ROUTING_MODE})")

        speedup = t_std / t_slm if t_slm > 0 else float("inf")

        print(f"  Retrieval  —  StdRAG: {t_std:.1f}ms (pool={pool_std})  |  "
              f"SLM-RAG: {t_slm:.1f}ms (pool={pool_slm})  |  "
              f"SLM-Full: disattivato  speedup={speedup:.1f}x")
        print(f"  Generazione StdRAG...")
        t_gen0 = time.perf_counter()
        ans_std = generate(query, docs_std, GENERATION_MODEL)
        t_gen_std = (time.perf_counter() - t_gen0) * 1000

        print(f"  Generazione SLM-RAG...")
        t_gen0 = time.perf_counter()
        ans_slm = generate(query, docs_slm, GENERATION_MODEL)
        t_gen_slm = (time.perf_counter() - t_gen0) * 1000

        # print(f"  Generazione SLM-Full...")
        # t_gen0 = time.perf_counter()
        # ans_full = generate(query, docs_full, GENERATION_MODEL)
        # t_gen_full = (time.perf_counter() - t_gen0) * 1000
        ans_full, t_gen_full = "", 0.0

        std_gens.append(t_gen_std); slm_gens.append(t_gen_slm); full_gens.append(t_gen_full)

        print(f"  Gen times  —  StdRAG: {t_gen_std:.0f}ms  |  SLM-RAG: {t_gen_slm:.0f}ms  |  SLM-Full: disattivato")
        print(f"  StdRAG  : {ans_std[:100]}…")
        print(f"  SLM-RAG : {ans_slm[:100]}…")

        md_rows.append({
            "paper_id":     paper_id,
            "query":        query,
            "ground_truth": ground_truth,
            "ans_std":      ans_std,
            "ans_slm":      ans_slm,
            "ans_full":     ans_full,
            "cls_std":      classify_answer(ans_std),
            "cls_slm":      classify_answer(ans_slm),
            "cls_full":     "DISATTIVATO",  # SLM-Full non eseguito, vedi commento sopra
            "docs_std":     docs_std,
            "docs_slm":     docs_slm,
            "docs_full":    docs_full,
            "t_ret_std":    t_std,
            "t_ret_slm":    t_slm,
            "t_ret_full":   t_full,
            "t_gen_std":    t_gen_std,
            "t_gen_slm":    t_gen_slm,
            "t_gen_full":   t_gen_full,
            "speedup":      speedup,
            "pool_std":     pool_std,
            "pool_slm":     pool_slm,
            "pool_full":    pool_full,
            "hr_std":       hr_std,
            "hr_slm":       hr_slm,
            "hr_full":      hr_full,
            "overlap":      overlap,
            "recall3":      rec3,
            "recall5":      rec5,
        })

    # ── Salva Markdown ─────────────────────────────────────────────────
    avg_speedup        = np.mean(std_times) / np.mean(slm_times)
    avg_speedup_full   = np.mean(std_times) / np.mean(full_times) if np.mean(full_times) > 0 else 0.0
    pool_reduction     = (1 - np.mean(slm_pools)  / np.mean(std_pools)) * 100
    pool_reduction_full= (1 - np.mean(full_pools) / np.mean(std_pools)) * 100
    avg_ret_std        = np.mean(std_times)   # LAT-STD:   avg. retrieval latency of StdRAG (ms)
    avg_ret_slm        = np.mean(slm_times)   # LAT-CLARA: avg. retrieval latency of CLARA/SLM-RAG (ms)
    avg_ret_full       = np.mean(full_times)

    def _count_true_answers(key):
        return sum(1 for r in md_rows if r[key] == "RISPOSTA")

    n_q = len(md_rows)  # NQ: number of queries in the query set

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark RAG — {', '.join(col_names)}\n\n")
        f.write(f"**Modello:** {GENERATION_MODEL}  |  **top-k:** {TOP_K}  |  **top-N SLM:** {TOP_N_SLMS}  |  **routing:** {ROUTING_MODE}\n\n")
        f.write(f"| Metrica | StdRAG | SLM-RAG | SLM-Full |\n|---|---|---|---|\n")
        f.write(f"| Query (NQ) | {n_q} | {n_q} | {n_q} |\n")
        f.write(f"| Risposte vere | **{_count_true_answers('cls_std')}/{n_q}** | **{_count_true_answers('cls_slm')}/{n_q}** | disattivato |\n")
        f.write(f"| Retrieval latency (avg, ms) | **{avg_ret_std:.1f}** | **{avg_ret_slm:.1f}** | — |\n")
        f.write(f"| Speedup retrieval | — | **{avg_speedup:.1f}x** | — |\n")
        f.write(f"| Pool medio | {int(np.mean(std_pools))} chunk "
                f"| {int(np.mean(slm_pools))} chunk (-{pool_reduction:.0f}%) "
                f"| — |\n")
        f.write(f"| Generazione media | {np.mean(std_gens):.0f} ms | {np.mean(slm_gens):.0f} ms | — |\n")
        f.write(f"| Keyword hit | {np.mean(std_hits):.0f}% | **{np.mean(slm_hits):.0f}%** | — |\n")
        f.write(f"| Overlap medio top-{TOP_K} (Std vs SLM) | {np.mean(overlaps):.0f}% | — | — |\n")
        f.write(f"| Routing recall @3 / @5 | — | {np.mean(recalls3):.0f}% / {np.mean(recalls5):.0f}% | idem |\n\n")
        f.write(f"<!-- LaTeX placeholders: [TBD:NQ]={n_q}  "
                f"[TBD:LAT-STD]={avg_ret_std:.1f}ms  [TBD:LAT-CLARA]={avg_ret_slm:.1f}ms -->\n\n")
        f.write("---\n\n")

        for i, r in enumerate(md_rows, 1):
            f.write(f"## {i}. {r['query']}\n\n")
            f.write(f"*Fonte: {r['paper_id']}*\n\n")
            f.write(f"| | StdRAG | SLM-RAG | SLM-Full |\n|---|---|---|---|\n")
            f.write(f"| **Esito risposta** | {r['cls_std']} | {r['cls_slm']} | {r['cls_full']} |\n")
            f.write(f"| **Retrieval** | {r['t_ret_std']:.1f} ms | {r['t_ret_slm']:.1f} ms ({r['speedup']:.1f}x) | {r['t_ret_full']:.1f} ms |\n")
            f.write(f"| **Generazione** | {r['t_gen_std']:.0f} ms | {r['t_gen_slm']:.0f} ms | {r['t_gen_full']:.0f} ms |\n")
            f.write(f"| **Totale** | {r['t_ret_std']+r['t_gen_std']:.0f} ms "
                    f"| {r['t_ret_slm']+r['t_gen_slm']:.0f} ms "
                    f"| {r['t_ret_full']+r['t_gen_full']:.0f} ms |\n")
            f.write(f"| **Pool chunk** | {r['pool_std']} | {r['pool_slm']} | {r['pool_full']} |\n")
            f.write(f"| **Keyword hit** | {r['hr_std']:.0f}% | {r['hr_slm']:.0f}% | {r['hr_full']:.0f}% |\n")
            f.write(f"| **Routing recall @3/@5** | — | {r['recall3']:.0f}% / {r['recall5']:.0f}% | idem |\n\n")
            f.write(f"### Risposta StdRAG\n\n{r['ans_std']}\n\n")
            f.write(f"### Risposta SLM-RAG\n\n{r['ans_slm']}\n\n")
            f.write(f"### Risposta SLM-Full\n\n{r['ans_full']}\n\n")
            f.write(f"### Ground Truth\n\n{r['ground_truth']}\n\n")
            f.write("---\n\n")

    print(f"\nFile salvato: {OUTPUT_FILE}")

    # ── Salva JSON per evaluate_quality.py ────────────────────────────
    json_path = OUTPUT_FILE.replace(".md", f"_results.json")
    json_data = {
        "metadata": {
            "collections":       col_names,
            "model":             GENERATION_MODEL,
            "top_k":             TOP_K,
            "top_n_slm":         TOP_N_SLMS,
            "routing_mode":      ROUTING_MODE,
            "n_queries":         n_q,               # NQ
            "avg_ret_ms_std":    round(avg_ret_std, 1),   # LAT-STD
            "avg_ret_ms_slm":    round(avg_ret_slm, 1),   # LAT-CLARA
            "avg_ret_ms_full":   round(avg_ret_full, 1),
        },
        "results": [
            {
                "paper_id":     r["paper_id"],
                "query":        r["query"],
                "ground_truth": r["ground_truth"],
                "std": {
                    "chunks":          r["docs_std"],
                    "answer":          r["ans_std"],
                    "answer_class":    r["cls_std"],
                    "t_ret_ms":        r["t_ret_std"],
                    "t_gen_ms":        r["t_gen_std"],
                    "pool":            r["pool_std"],
                    "keyword_hit_pct": r["hr_std"],
                },
                "slm": {
                    "chunks":          r["docs_slm"],
                    "answer":          r["ans_slm"],
                    "answer_class":    r["cls_slm"],
                    "t_ret_ms":        r["t_ret_slm"],
                    "t_gen_ms":        r["t_gen_slm"],
                    "pool":            r["pool_slm"],
                    "keyword_hit_pct": r["hr_slm"],
                },
                "full": {
                    "chunks":          r["docs_full"],
                    "answer":          r["ans_full"],
                    "answer_class":    r["cls_full"],
                    "t_ret_ms":        r["t_ret_full"],
                    "t_gen_ms":        r["t_gen_full"],
                    "pool":            r["pool_full"],
                    "keyword_hit_pct": r["hr_full"],
                },
                "overlap_pct":        r["overlap"],
                "speedup":            r["speedup"],
                "routing_recall_at3": r["recall3"],
                "routing_recall_at5": r["recall5"],
            }
            for r in md_rows
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON salvato:  {json_path}")
    print(f"\nRIEPILOGO  (routing={ROUTING_MODE})")
    print(f"  NQ (numero query)     : {n_q}")
    print(f"  Risposte vere StdRAG  : {_count_true_answers('cls_std')}/{n_q}")
    print(f"  Risposte vere SLM-RAG : {_count_true_answers('cls_slm')}/{n_q}")
    print(f"  Risposte vere SLM-Full: disattivato")
    print(f"  Routing recall @3 / @5 : {np.mean(recalls3):.0f}% / {np.mean(recalls5):.0f}%")
    print(f"  Retrieval latency (avg) StdRAG    [LAT-STD]   : {avg_ret_std:.1f} ms")
    print(f"  Retrieval latency (avg) SLM-RAG   [LAT-CLARA] : {avg_ret_slm:.1f} ms")
    print(f"  Speedup retrieval SLM-RAG  : {avg_speedup:.1f}x")
    print(f"  Pool medio  StdRAG  : {int(np.mean(std_pools))} chunk")
    print(f"  Pool medio  SLM-RAG : {int(np.mean(slm_pools))} chunk  ({pool_reduction:+.0f}%)")
    print(f"  Generazione StdRAG  : {np.mean(std_gens):.0f} ms")
    print(f"  Generazione SLM-RAG : {np.mean(slm_gens):.0f} ms")
    print(f"  Overlap medio top-{TOP_K}  : {np.mean(overlaps):.0f}%")
    print(f"  Keyword hit StdRAG  : {np.mean(std_hits):.0f}%")
    print(f"  Keyword hit SLM-RAG : {np.mean(slm_hits):.0f}%")


if __name__ == "__main__":
    main()
