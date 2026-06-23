# RAG Answer Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce off-topic and weak-evidence answers in SoulDrive by tightening prompt control, adding multi-query retrieval and lightweight reranking, and exposing model capability status for local users.

**Architecture:** Keep the existing offline sidecar RAG flow and improve it incrementally. Retrieval will expand into a larger candidate pool, optionally rerank with a local cross-encoder, gate weak evidence more aggressively, and only then call the generator. Model selection remains local-file based, with lightweight defaults and automatic upgrade when stronger local models exist.

**Tech Stack:** Python 3.10, unittest, llama-cpp-python, sentence-transformers, FastAPI, React, TypeScript

---

## Files And Responsibilities

- `core/rag_engine.py`: prompt builder, generation retry policy, evidence-aware answer flow.
- `core/knowledge_base.py`: multi-query retrieval, candidate pool expansion, reranker integration point.
- `core/answer_quality.py`: stronger evidence gate and answer citation validation.
- `core/model_runtime.py`: default/enhanced chat-model priority and reranker model discovery.
- `core/diagnostics.py`: surface reranker and selected answer-model status in diagnostics.
- `core/mcp_server.py`: expose richer model status through existing diagnostics endpoints.
- `core/paths.py` or helper-local path resolution usage: keep model discovery local and workspace-aware.
- `core/reranker.py`: optional local cross-encoder wrapper with safe fallback when model files are missing.
- `core/query_expansion.py`: bounded multi-query expansion helper.
- `souldrive-ui/src/App.tsx`: present current answer model, reranker model, and acceleration status.
- `tests/test_rag_prompting.py`: prompt contract and retry/refusal behavior.
- `tests/test_query_expansion.py`: multi-query shape and bounds.
- `tests/test_reranker.py`: reranker loading, scoring, and safe downgrade behavior.
- `tests/test_answer_quality.py`: stricter evidence gate and citation validation.
- `tests/test_model_runtime.py`: chat model priority and reranker model detection.
- `tests/test_diagnostics.py`: diagnostics surface selected answer/reranker model state.

## Task 1: Stronger Evidence Gate

**Files:**
- Modify: `core/answer_quality.py`
- Modify: `tests/test_answer_quality.py`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- low score rejection
- medium score requiring retry
- high score acceptance
- answer validation rejecting unsupported citations when evidence exists

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_answer_quality -v`

- [ ] **Step 3: Write minimal implementation**

Add:
- a three-state gate result such as `reject`, `retry`, `accept`
- stronger thresholds based on top score and evidence count
- `validate_answer_citations(answer, evidence)` helper for retry/deny decisions

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_answer_quality -v`

## Task 2: Multi-Query Expansion

**Files:**
- Create: `core/query_expansion.py`
- Create: `tests/test_query_expansion.py`

- [ ] **Step 1: Write the failing tests**

Cover:
- expansion returns original query first
- total query variants stays within 3
- empty or repeated variants are removed
- technical Chinese/English terms are preserved

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_query_expansion -v`

- [ ] **Step 3: Write minimal implementation**

Implement a bounded heuristic helper that expands common question forms without needing another model call in v1.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_query_expansion -v`

## Task 3: Local Reranker Wrapper

**Files:**
- Create: `core/reranker.py`
- Create: `tests/test_reranker.py`
- Modify: `core/model_runtime.py`

- [ ] **Step 1: Write the failing tests**

Cover:
- detects `mmarco-mMiniLMv2-L6-H384-v1` before `bge-reranker-base`
- loads safely when model exists
- returns downgrade mode when reranker model is absent

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_reranker tests.test_model_runtime -v`

- [ ] **Step 3: Write minimal implementation**

Implement:
- reranker model name priority
- local-files-only sentence-transformers cross-encoder loader
- no-model fallback that preserves current retrieval path

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_reranker tests.test_model_runtime -v`

## Task 4: Retrieval Candidate Expansion And Reranking

**Files:**
- Modify: `core/knowledge_base.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

Cover:
- multi-query retrieval increases candidate pool before final ranking
- reranker score can override weak lexical/dense ordering
- no reranker path still works

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_retrieval -v`

- [ ] **Step 3: Write minimal implementation**

Add:
- query expansion in `search_with_evidence`
- larger pre-ranking candidate pool
- optional reranker score integration into final candidate ranking

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_retrieval -v`

## Task 5: Prompt Repair And Answer Retry

**Files:**
- Modify: `core/rag_engine.py`
- Create: `tests/test_rag_prompting.py`

- [ ] **Step 1: Write the failing tests**

Cover:
- prompt builder emits clean readable instruction text
- reject gate returns refusal immediately
- retry gate triggers one regeneration attempt at most
- unsupported citations force retry then refusal if still invalid

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_rag_prompting -v`

- [ ] **Step 3: Write minimal implementation**

Refactor:
- isolate prompt builder into small helpers
- replace broken prompt strings with clean UTF-8 Chinese
- integrate new evidence gate states
- validate answer citations before yielding final response

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_rag_prompting -v`

## Task 6: Model Priority And Diagnostics

**Files:**
- Modify: `core/model_runtime.py`
- Modify: `core/diagnostics.py`
- Modify: `tests/test_model_runtime.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Cover:
- 7B instruct preferred over 3B
- diagnostics report selected answer model and reranker state
- diagnostics remain valid when reranker is absent

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n souldrive python -m unittest tests.test_model_runtime tests.test_diagnostics -v`

- [ ] **Step 3: Write minimal implementation**

Expose:
- selected answer model name
- selected reranker model name or disabled reason
- existing GPU status unchanged

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n souldrive python -m unittest tests.test_model_runtime tests.test_diagnostics -v`

## Task 7: UI Capability Status

**Files:**
- Modify: `souldrive-ui/src/App.tsx`

- [ ] **Step 1: Write the failing build expectation**

Use the existing TypeScript build as the verification gate for the added diagnostics fields.

- [ ] **Step 2: Run build to verify it fails after type additions**

Run: `npm run build`
Working directory: `souldrive-ui`

- [ ] **Step 3: Write minimal implementation**

Show:
- current answer model
- current reranker model
- CPU/GPU acceleration status
- a lightweight enhancement hint when only the default model set is present

- [ ] **Step 4: Run build to verify it passes**

Run: `npm run build`
Working directory: `souldrive-ui`

## Task 8: Regression Verification

**Files:**
- Verify only

- [ ] **Step 1: Run targeted backend regression**

Run:
`conda run -n souldrive python -m unittest tests.test_answer_quality tests.test_query_expansion tests.test_reranker tests.test_retrieval tests.test_rag_prompting tests.test_model_runtime tests.test_diagnostics tests.test_mcp_server_security tests.test_mcp_server_indexing tests.test_knowledge_base_paths -v`

- [ ] **Step 2: Run full backend suite**

Run:
`conda run -n souldrive python -m unittest discover -s tests -v`

- [ ] **Step 3: Run frontend build**

Run:
`npm run build`
Working directory: `souldrive-ui`

- [ ] **Step 4: Optional packaged runtime check**

Run:
`.\scripts\package-sidecar.ps1`
