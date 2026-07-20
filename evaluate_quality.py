"""
evaluate_quality.py — Valutazione qualitativa RAG con RAGAS + GPT-4o come giudice.
"""

import os
import json
import argparse
import math
import numpy as np
from pathlib import Path

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

# ── I/O ────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── RAGAS dataset ──────────────────────────────────────────────────────────────

def build_samples(results: list, mode: str):
    from ragas import SingleTurnSample
    samples = []
    for r in results:
        chunks = r[mode]["chunks"]
        if not chunks:
            chunks = ["Nessun contesto recuperato."]
        samples.append(SingleTurnSample(
            user_input=r["query_en"],
            response=r[mode]["answer"],
            retrieved_contexts=chunks,
            reference=r["ground_truth"],
        ))
    return samples


# ── RAGAS evaluation ───────────────────────────────────────────────────────────

def run_ragas(samples, llm, embeddings):
    from ragas import EvaluationDataset, evaluate, RunConfig
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
        Faithfulness,
    )
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        ResponseRelevancy(),
        Faithfulness(),
    ]
    dataset = EvaluationDataset(samples=samples)
    run_config = RunConfig(max_workers=2, timeout=120)
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )


# ── Report ─────────────────────────────────────────────────────────────────────

METRIC_KEYS = {
    "llm_context_precision_with_reference": "Contextual Precision",
    "llm_context_recall":                   "Contextual Recall",
    "response_relevancy":                   "Answer Relevancy",
    "faithfulness":                         "Faithfulness",
}


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{float(val):.3f}"


def _mean(score_obj, key: str) -> float:
    try:
        return float(score_obj[key])
    except Exception:
        return float("nan")


def _per_sample(score_obj, key: str) -> list:
    try:
        return [s.get(key, float("nan")) for s in score_obj.scores]
    except Exception:
        return []


def write_report(results, score_std, score_slm, score_full, metadata, output_path):
    col   = metadata.get("collection", "?")
    model = metadata.get("model", "?")

    # aggregate means
    means_std  = {k: _mean(score_std,  k) for k in METRIC_KEYS}
    means_slm  = {k: _mean(score_slm,  k) for k in METRIC_KEYS}
    means_full = {k: _mean(score_full, k) for k in METRIC_KEYS}

    # per-sample scores
    per_std  = {k: _per_sample(score_std,  k) for k in METRIC_KEYS}
    per_slm  = {k: _per_sample(score_slm,  k) for k in METRIC_KEYS}
    per_full = {k: _per_sample(score_full, k) for k in METRIC_KEYS}

    # efficiency aggregates from JSON
    std_ret   = np.mean([r["std"]["t_ret_ms"]  for r in results])
    slm_ret   = np.mean([r["slm"]["t_ret_ms"]  for r in results])
    full_ret  = np.mean([r["full"]["t_ret_ms"] for r in results])
    std_gen   = np.mean([r["std"]["t_gen_ms"]  for r in results])
    slm_gen   = np.mean([r["slm"]["t_gen_ms"]  for r in results])
    full_gen  = np.mean([r["full"]["t_gen_ms"] for r in results])
    std_pool  = np.mean([r["std"]["pool"]      for r in results])
    slm_pool  = np.mean([r["slm"]["pool"]      for r in results])
    full_pool = np.mean([r["full"]["pool"]     for r in results])
    std_hr    = np.mean([r["std"]["keyword_hit_pct"]  for r in results])
    slm_hr    = np.mean([r["slm"]["keyword_hit_pct"]  for r in results])
    full_hr   = np.mean([r["full"]["keyword_hit_pct"] for r in results])
    speedup_slm  = std_ret / slm_ret  if slm_ret  > 0 else 0.0
    speedup_full = std_ret / full_ret if full_ret > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(f"# Quality Report — {col}\n\n")
        f.write(f"**Modello generazione:** {model}  |  "
                f"**Giudice LLM:** gpt-4o  |  "
                f"**Query:** {len(results)}\n\n")

        # ── Metriche di qualità ────────────────────────────────────────
        f.write("## Metriche di qualità (RAGAS)\n\n")
        f.write("| Metrica | StdRAG | SLM-RAG | SLM-Full | Δ (SLM−Std) | Δ (Full−Std) |\n")
        f.write("|---|---|---|---|---|---|\n")

        def _delta_str(a, b):
            if math.isnan(a) or math.isnan(b):
                return "—"
            d = b - a
            sign = "+" if d > 0 else ""
            return f"{sign}{_fmt(d)}"

        for key, label in METRIC_KEYS.items():
            s  = means_std[key]
            sl = means_slm[key]
            fu = means_full[key]
            f.write(f"| {label} | {_fmt(s)} | {_fmt(sl)} | {_fmt(fu)} "
                    f"| {_delta_str(s, sl)} | {_delta_str(s, fu)} |\n")

        f.write("\n---\n\n")

        # ── Metriche di efficienza ─────────────────────────────────────
        f.write("## Metriche di efficienza\n\n")
        f.write("| Metrica | StdRAG | SLM-RAG | SLM-Full |\n|---|---|---|---|\n")
        f.write(f"| Retrieval medio | {std_ret:.1f} ms | {slm_ret:.1f} ms | {full_ret:.1f} ms |\n")
        f.write(f"| Generazione media | {std_gen:.0f} ms | {slm_gen:.0f} ms | {full_gen:.0f} ms |\n")
        f.write(f"| Pool medio | {std_pool:.0f} chunk | {slm_pool:.0f} chunk | {full_pool:.0f} chunk |\n")
        f.write(f"| Speedup retrieval | — | **{speedup_slm:.1f}x** | **{speedup_full:.1f}x** |\n")

        f.write("\n---\n\n")

        # ── Dettaglio per query ────────────────────────────────────────
        f.write("## Dettaglio per query\n\n")

        cp = "llm_context_precision_with_reference"
        cr = "llm_context_recall"
        ar = "response_relevancy"
        fa = "faithfulness"

        f.write(
            "| # | Query "
            "| CP Std | CP SLM | CP Full "
            "| CR Std | CR SLM | CR Full "
            "| AR Std | AR SLM | AR Full "
            "| Faith Std | Faith SLM | Faith Full |\n"
        )
        f.write("|---|---|" + "---|" * 12 + "\n")

        for i, r in enumerate(results):
            label = r["query_en"][:48] + ("…" if len(r["query_en"]) > 48 else "")

            def gs(key, src, idx=i):
                lst = {"std": per_std, "slm": per_slm, "full": per_full}[src][key]
                return _fmt(lst[idx]) if idx < len(lst) else "—"

            f.write(
                f"| {i+1} | {label} "
                f"| {gs(cp,'std')} | {gs(cp,'slm')} | {gs(cp,'full')} "
                f"| {gs(cr,'std')} | {gs(cr,'slm')} | {gs(cr,'full')} "
                f"| {gs(ar,'std')} | {gs(ar,'slm')} | {gs(ar,'full')} "
                f"| {gs(fa,'std')} | {gs(fa,'slm')} | {gs(fa,'full')} |\n"
            )

        f.write("\n---\n\n")

        # ── Risposte complete per ogni query ───────────────────────────
        f.write("## Risposte per query\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"### {i}. {r['query_en']}\n\n")
            f.write(f"*{r['query_it']}*\n\n")
            f.write(f"**Ground Truth:** {r['ground_truth']}\n\n")
            f.write(f"**StdRAG:** {r['std']['answer']}\n\n")
            f.write(f"**SLM-RAG:** {r['slm']['answer']}\n\n")
            f.write(f"**SLM-Full:** {r['full']['answer']}\n\n")
            f.write("---\n\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valutazione qualitativa RAG con RAGAS + GPT-4o"
    )
    parser.add_argument(
        "--input",  default="benchmark_answers_spacy_4B_3_WAYS_5PDF_3_SLM_TOK_K_5_results.json",
        help="JSON prodotto da benchmark.py"
    )
    parser.add_argument(
        "--output", default="benchmark_answers_spacy_4B_3_WAYS_5PDF_3_SLM_TOK_K_5.md",
        help="File markdown di output"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Errore: OPENAI_API_KEY non impostata.\n"
            "  export OPENAI_API_KEY=sk-..."
        )

    if not Path(args.input).exists():
        raise SystemExit(
            f"Errore: file '{args.input}' non trovato.\n"
            "  Esegui prima: python3 benchmark.py"
        )

    data     = load_results(args.input)
    results  = data["results"]
    metadata = data.get("metadata", {})
    print(f"Query caricate: {len(results)}  |  collection: {metadata.get('collection','?')}")

    # ── Setup LLM e embeddings (RAGAS >= 0.2 API) ──────────────────────
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory

    openai_client = OpenAI(api_key=api_key, max_retries=8)
    llm = llm_factory("gpt-4o", client=openai_client, max_tokens=4096)

    from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embeddings = LangchainEmbeddingsWrapper(
        LCHuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    # ── Valutazione ────────────────────────────────────────────────────
    samples_std  = build_samples(results, "std")
    samples_slm  = build_samples(results, "slm")
    samples_full = build_samples(results, "full")

    print("\nValutazione StdRAG con RAGAS...")
    score_std  = run_ragas(samples_std,  llm, embeddings)

    print("Valutazione SLM-RAG con RAGAS...")
    score_slm  = run_ragas(samples_slm,  llm, embeddings)

    print("Valutazione SLM-Full con RAGAS...")
    score_full = run_ragas(samples_full, llm, embeddings)

    # ── Report ─────────────────────────────────────────────────────────
    write_report(results, score_std, score_slm, score_full, metadata, args.output)
    print(f"\nReport salvato: {args.output}")


if __name__ == "__main__":
    main()
