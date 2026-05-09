# Rust Port Status Summary (as of 2026-05-07)

## Done

- Rust workspace bootstrapped with crates:
  - `crates/retrieval-core`
  - `crates/inference-runtime`
  - `crates/qmd-rs`
- Basic retrieval-core building blocks implemented:
  - chunking, token normalization
  - lexical scoring helper
  - weighted fusion and reciprocal-rank-fusion
  - unit tests for deterministic behavior
- Basic inference-runtime building blocks implemented:
  - runtime traits (`Embedder`, `Reranker`, `Expander`, `Generator`)
  - runtime registry wiring
  - model cache layout helper + quantization default helper
  - query-path policy helpers
  - unit tests for trait wiring and cache layout
- CLI command surface stubbed:
  - `qmd-rs index`
  - `qmd-rs query`
  - `qmd-rs serve`
- Baseline/eval scaffolding added:
  - sample eval corpus, queries, qrels
  - benchmark harness script
  - baseline artifacts placeholders
- Packaging/compat scaffolding added:
  - packaging checklist
  - plugin fixture replay test

## Left to Build (for a full QMD Rust port)

### Retrieval and indexing
- Replace placeholder lexical scoring with real BM25 index/search (e.g., Tantivy).
- Implement persistent ANN vector index and search (e.g., HNSW), including save/load.
- Implement real document parsing/chunking pipeline with metadata extraction.
- Add robust score calibration/fusion and evaluation against baseline metrics.

### Inference runtime
- Implement real model backends (ONNX Runtime/Candle) for embed/rerank/generate.
- Add model download, checksum verification, versioning, and cache invalidation.
- Implement streaming generation and bounded resource usage controls.
- Add batching/concurrency controls and warm pools for latency targets.

### API/Plugin compatibility
- Implement real local RPC/HTTP service and stable request/response schemas. (In progress: `GET /health` + `POST /query` implemented in `qmd-rs serve` with JSON request/response handling.)
- Build TS shim parity tests against current plugin contract.
- Add replay/integration tests with realistic plugin traces.

### Evaluation and quality gates
- Replace placeholder benchmark artifacts with real measured metrics.
- Add Recall@20/NDCG@10/MRR evaluation pipeline over representative corpora.
- Add pass/fail quality and SLO gates in CI.

### Packaging and ops
- Create cross-platform release automation for macOS/Linux/Windows (x86_64/aarch64 as relevant).
- Generate signed checksums + SBOM; produce one-step installer.
- Add startup diagnostics, structured logs, telemetry hooks, and graceful fallback behavior.

## Bottom line

The current repository is a strong scaffolding baseline. It is **not yet** a full production-complete Rust port of QMD.
