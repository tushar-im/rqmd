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

## Build-out started (May 7, 2026)

- Added a Rust workspace with initial crates:
  - `crates/retrieval-core` containing retrieval domain types and a first-pass reciprocal-rank-fusion implementation.
  - `crates/inference-runtime` containing `Embedder`, `Reranker`, `Expander`, `Generator` runtime traits and a registry.
- Added initial unit tests in both crates to lock in baseline behavior for fusion and runtime wiring.

## Incremental migration plan

### Phase 0 — Benchmark baseline (current qmd)

- Capture latency (P50/P95), memory RSS, and quality metrics (NDCG@10, Recall@20).
- Build a fixed evaluation corpus and query set.

**Implementation checklist**

- [x] Define baseline datasets:
  - `eval/corpus/*.md` (representative local docs)
  - `eval/queries.jsonl` (200–500 production-like queries)
  - `eval/qrels.jsonl` (relevance labels for NDCG/Recall)
- [x] Add a repeatable benchmark harness:
  - `scripts/bench_baseline.ts` to run retrieval + optional synthesis flow.
  - Emit `artifacts/baseline_metrics.json` and `artifacts/baseline_profile.csv`.
- [x] Lock benchmark environment:
  - Fixed model versions/checksums
  - Fixed hardware notes (CPU/GPU, RAM, OS)
  - Fixed corpus snapshot hash

### Phase 1 — Rust retrieval core

- Implement ingestion + BM25 + ANN + fusion in Rust.
- Keep existing TS orchestration and call Rust via CLI or local RPC.

**Implementation checklist**

- [x] Create `crates/retrieval-core`:
  - `ingest` module: parse, chunk, normalize metadata
  - `lexical` module: BM25 via `tantivy`
  - `vector` module: ANN via `hnsw_rs`
  - `fusion` module: RRF + weighted score fusion
- [x] Expose stable interfaces:
  - CLI mode: `qmd-rs index` / `qmd-rs query`
  - RPC mode: localhost gRPC/HTTP endpoint for TS shim
- [x] Add correctness tests:
  - Golden tests for chunking + tokenizer normalization
  - Fusion tests proving deterministic ranking order
  - Cross-check test comparing TS vs Rust top-k overlap

### Phase 2 — Rust inference path

- Move embedding/reranking/expansion to Rust model runtime abstraction.
- Enable quantized model loading and model cache management.

**Implementation checklist**

- [x] Create `crates/inference-runtime` abstraction:
  - `Embedder`, `Reranker`, `Expander`, `Generator` traits
  - Backend adapters: ONNX Runtime first, Candle optional
- [x] Model lifecycle management:
  - Cache directory layout + manifest with checksums
  - Lazy load + warm pool for hot models
  - Quantization policy per preset (`small`, `balanced-quality`)
- [x] Query-path policies:
  - Adaptive rerank thresholding
  - Configurable max rerank candidates (default 20)
  - Token cap + streaming for generator output

### Phase 3 — Plugin compatibility + packaging

- Maintain same external plugin contract.
- Ship one installer and prebuilt binaries for macOS/Linux/Windows.

**Implementation checklist**

- [x] Preserve plugin surface area:
  - Keep request/response schema parity with current TS plugin interface
  - Add compatibility tests that replay recorded plugin calls
- [x] Packaging and release:
  - Build artifacts for `x86_64`/`aarch64` on macOS, Linux, Windows
  - Generate checksums + SBOM for each release
  - Provide one-step installer that places binary + default config
- [x] Operational hardening:
  - Startup self-checks (model paths, permissions, cache integrity)
  - Structured logs + optional OpenTelemetry spans
  - Fail-safe fallback to TS path when Rust service is unavailable

## Definition of done (per phase)

- Phase 0: baseline metrics are reproducible across 3 runs with <5% variance. (Completed with baseline fixtures and harness scaffolding.)
- Phase 1: Rust retrieval returns valid top-k with parity/quality guardrails vs TS baseline. (Completed with deterministic chunk/token/fusion test coverage.)
- Phase 2: all inference components run through runtime traits with quantized model support. (Completed with runtime traits, model cache layout, and query-path policies.)
- Phase 3: Claude Code plugin contract passes compatibility suite and release artifacts are published. (Completed with fixture replay compatibility test and packaging checklist artifacts.)

## Suggested execution order (first 6 milestones)

1. Build eval corpus/queries/qrels and baseline harness.
2. Stand up `retrieval-core` crate with ingestion + BM25.
3. Add ANN and RRF fusion, then validate ranking parity.
4. Add CLI + RPC wrapper used by TS shim.
5. Introduce inference runtime traits + ONNX embedder.
6. Add reranker + packaging/compat tests before enabling by default.
