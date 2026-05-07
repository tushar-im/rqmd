# qmd → Rust Port Plan (local-first semantic search)

## Goals

- Preserve qmd’s local-first architecture: BM25 + vector retrieval + rerank + optional query expansion.
- Reduce model footprint from ~2GB toward a practical target of **300–900MB** depending on quality tier.
- Improve runtime predictability (latency + memory) versus TypeScript-only inference.
- Keep Claude Code plugin compatibility by exposing the same high-level interface.

## Proposed stack requested

You asked for:

- **Embedding:** `Qwen3-0.6B-Embedding`
- **Reranker:** `Qwen3-0.6B-Reranker`
- **Querying LLM:** `BitNet b1.58 2B4T`

This is a sensible high-quality local stack, but the performance profile depends heavily on quantization and CPU vs GPU execution.

## Expected performance impact (practical estimate)

> These are engineering estimates for a local developer machine running a Rust retrieval service; exact values must be verified with your corpus and hardware.

### 1) Throughput and latency

- **Embedding stage (index time, query time for query embedding):**
  - With a 0.6B embedding model, per-query embedding latency is generally **moderate** (typically much slower than tiny 100M models, but manageable with batching).
  - Expect index build throughput to drop vs tiny embedders, roughly **1.5–3.0× slower** depending on quantization and hardware.

- **Reranking stage (query-time):**
  - A 0.6B reranker is likely the largest latency contributor in retrieval-only flows.
  - For top-k rerank (e.g., 20–50 candidates), expect query P95 to increase by roughly **1.8–4.0×** versus lightweight distilled rerankers.

- **BitNet b1.58 2B4T querying LLM:**
  - If only used after retrieval/rerank for answer synthesis, it does **not** affect recall directly but can dominate end-to-end response time.
  - Expect answer generation latency to scale with output tokens; often this becomes the new bottleneck unless you cap max tokens and stream.

### 2) Memory footprint (RAM)

- Compared to compact default stacks, this trio likely increases runtime RAM substantially.
- With aggressive quantization + memory-mapped weights, expect a **material increase** in steady-state RSS; without quantization, memory may become a practical blocker on mid-range laptops.
- Biggest spikes appear when reranker and query LLM are both active concurrently.

### 3) Disk footprint

- Model cache size likely lands above the earlier “small default” target.
- This stack is more likely to sit in a **balanced/quality preset** than a small preset.

### 4) Quality impact

- **Retrieval quality:** likely improved over very small embedders/rerankers (better semantic matching, improved ranking depth).
- **Answer quality:** BitNet 2B can improve fluency/grounding when prompted with retrieved context, but benefit depends on prompt design and context packing quality.

## Net effect vs a lightweight baseline

Using `Qwen3-0.6B-Embedding + Qwen3-0.6B-Reranker + BitNet 2B4T` usually means:

- **Pros:** better ranking quality, better final answer quality.
- **Cons:** higher P95 latency, lower throughput, higher RAM, larger disk usage.

In practice this is typically a **quality-first preset**, not a speed-first preset.

## Recommended deployment pattern

1. Keep two shipping presets:
   - **small:** tiny embedder + distilled reranker + optional/no LLM
   - **balanced-quality:** Qwen 0.6B embedder/reranker + BitNet querying

2. Gate balanced-quality preset with SLO checks:
   - P95 query latency target
   - Max RSS target
   - Recall@20 / NDCG@10 floor

3. Optimize query path:
   - Rerank fewer docs first (e.g., 20 before 50)
   - Adaptive reranking (skip rerank for high-confidence BM25+vector agreement)
   - Stream answer tokens from BitNet and enforce output token caps

## Benchmark protocol to confirm real impact

Run A/B on the same corpus and 200–500 representative queries:

- **A (baseline small):** small embedder + small reranker + no/mini query LLM
- **B (requested stack):** Qwen3-0.6B-Embedding + Qwen3-0.6B-Reranker + BitNet 2B4T

Track:

- Retrieval: Recall@20, NDCG@10, MRR
- Performance: P50/P95 latency (embed, retrieve, rerank, generate), QPS
- Resource: peak RSS, steady RSS, model load time, disk footprint

Adopt B as default only if quality gain justifies latency/RAM increase for your target hardware.

## Suggested Rust Architecture

1. **Ingestion pipeline**
   - Parse + chunk docs.
   - Compute lexical index terms and dense vectors.
   - Persist metadata and vectors.

2. **Hybrid retrieval**
   - BM25 via `tantivy`.
   - Vector ANN via HNSW (`hnsw_rs`) or IVF-PQ (if we wire FAISS/Qdrant local mode).
   - Score fusion (RRF or weighted sum).

3. **Query expansion (optional)**
   - Default: rule-based + RM3-style pseudo-relevance expansion (no model).
   - Optional model-based expander for higher quality.

4. **Reranking**
   - Lightweight cross-encoder or late-interaction reranker.
   - Distilled model by default, larger model optional.

5. **Plugin/API layer**
   - gRPC/HTTP local service with a thin TS shim so Claude Code integration remains stable.

## Runtime/backends to support in Rust

- **ONNX Runtime** for broad model compatibility and mature quantized kernels.
- **Candle** for pure-Rust deployments and tighter binary control.
- Optional **llama.cpp** bridge for GGUF models when that yields best local perf.

## Incremental migration plan

### Phase 0 — Benchmark baseline (current qmd)

- Capture latency (P50/P95), memory RSS, and quality metrics (NDCG@10, Recall@20).
- Build a fixed evaluation corpus and query set.

### Phase 1 — Rust retrieval core

- Implement ingestion + BM25 + ANN + fusion in Rust.
- Keep existing TS orchestration and call Rust via CLI or local RPC.

### Phase 2 — Rust inference path

- Move embedding/reranking/expansion to Rust model runtime abstraction.
- Enable quantized model loading and model cache management.

### Phase 3 — Plugin compatibility + packaging

- Maintain same external plugin contract.
- Ship one installer and prebuilt binaries for macOS/Linux/Windows.
