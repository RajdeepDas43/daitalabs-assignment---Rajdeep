# RAG Techniques Deep-Dive: Analysis & Benchmark Report

## 1. Executive Summary

We benchmarked **11 different retrieval strategies** across the full RAG pipeline, spanning chunking, embedding, retrieval, and reranking techniques. All strategies were tested against the same 100-query evaluation set (75 answerable, 25 unanswerable).

**Key finding:** The V1 baseline (BM25 + TF-IDF with Reciprocal Rank Fusion) achieves **perfect scores** — 100% Recall@5, 100% Answer Accuracy, 0% Hallucination Rate — and is the optimal choice for this dataset. More advanced techniques either match or slightly underperform it, while adding latency and complexity.

This report explains *why*, and identifies *when* each advanced technique would become the better choice.

---

## 2. Benchmark Results

| # | Strategy | Recall@5 | Accuracy | Halluc | Time | Pass |
|---|----------|----------|----------|--------|------|------|
| 1 | BM25 Only | 100.0% | 100.0% | 0.0% | 0.02s | ALL |
| 2 | TF-IDF Only | 100.0% | 100.0% | 0.0% | 0.02s | ALL |
| **3** | **BM25+TF-IDF RRF (V1 baseline)** | **100.0%** | **100.0%** | **0.0%** | **0.04s** | **ALL** |
| 4 | LSA Dense Only (50-dim SVD) | 100.0% | 98.7% | 0.0% | 0.06s | ALL |
| 5 | BM25 + Query Expansion | 100.0% | 97.3% | 0.0% | 0.03s | ALL |
| 6 | Full Hybrid (BM25+TF-IDF+LSA) | 100.0% | 100.0% | 0.0% | 0.11s | ALL |
| 7 | Contextual Chunks + BM25+TF-IDF | 100.0% | 100.0% | 0.0% | 0.04s | ALL |
| 8 | Contextual + Full Hybrid | 100.0% | 100.0% | 0.0% | 0.10s | ALL |
| 9 | Parent-Child (BM25 on children) | 100.0% | 100.0% | 0.0% | 0.01s | ALL |
| 10 | Proposition-Based Retrieval | 100.0% | 100.0% | 0.0% | 0.07s | ALL |
| 11 | V1 Hybrid + Simulated Reranker | 100.0% | 100.0% | 0.0% | 0.13s | ALL |

---

## 3. Technique-by-Technique Analysis

### 3.1 Chunking Strategies

| Strategy | Description | Impact on This Dataset | When It Shines |
|----------|-------------|----------------------|----------------|
| **Document-level** (V1) | One chunk per email with metadata header | Perfect — emails are short (~163 tokens), no splitting needed | Small documents, complete context needed |
| **Contextual** (Anthropic) | Prepend context summary to each chunk | No improvement here; helpful for context-poor chunks | Long documents where isolated chunks lose meaning. Anthropic reports 49% fewer retrieval failures |
| **Parent-Child** | Small metadata chunks for retrieval, full email for generation | Fastest (0.01s) — focused retrieval on metadata | Mixed long/short documents; when retrieval precision on specific fields matters |
| **Proposition-Based** | Decompose into atomic facts (696 props from 100 emails) | Perfect accuracy but 7x index size | Fact-heavy corpora; when queries target specific claims |
| **Semantic Chunking** | Split by embedding similarity boundaries | Not applicable — emails are already semantic units | Long heterogeneous documents (reports, manuals, legal contracts) |
| **Recursive Splitting** | Hierarchical paragraph→sentence→word splitting | Overkill for short emails | Long-form text with clear structural hierarchy |

**Recommendation:** For emails, document-level chunking is optimal. For longer documents (>1000 tokens), parent-child or contextual chunking provides the best accuracy-efficiency tradeoff.

### 3.2 Embedding & Retrieval Methods

| Method | How It Works | Recall@5 | Accuracy | Latency | Best For |
|--------|-------------|----------|----------|---------|----------|
| **BM25** | Probabilistic term matching with length normalization | 100% | 100% | 0.02s | Keyword-heavy queries, proper nouns, exact matching |
| **TF-IDF** | Term frequency × inverse document frequency | 100% | 100% | 0.02s | Statistical term importance; baseline |
| **LSA (SVD)** | Dense embeddings via truncated SVD on TF-IDF | 100% | 98.7% | 0.06s | Synonym resolution, semantic similarity |
| **BM25+TF-IDF RRF** | Reciprocal Rank Fusion of both | 100% | 100% | 0.04s | Robust hybrid; covers both keyword and statistical signals |
| **Full Hybrid (3-way)** | BM25+TF-IDF+LSA via RRF | 100% | 100% | 0.11s | Maximum robustness across query types |
| **Query Expansion** | Simulated SPLADE-like synonym expansion | 100% | 97.3% | 0.03s | Vocabulary mismatch; domain-specific terminology |

**Why dense embeddings underperform here:** The queries are primarily proper noun lookups ("Who is Felix Jordan?", "What domain does Sean Cox work at?"). BM25 matches these *exactly*. Dense embeddings compress "Felix" and "Jordan" into shared semantic dimensions, losing the exact-match precision. This is the fundamental dense-vs-sparse tradeoff.

**Why query expansion hurts slightly:** Adding synonyms like "sent→wrote→emailed" can dilute the signal when the original query already uses the exact right terms. Expansion introduces noise that shifts rankings for 2 of 75 queries.

### 3.3 Advanced Retrieval Strategies

| Strategy | Description | Impact Here | When Essential |
|----------|-------------|-------------|----------------|
| **Reciprocal Rank Fusion** | Merge ranked lists by reciprocal rank position | Core of V1; simple and robust | Always, when combining multiple retrieval methods |
| **Retrieve-then-Rerank** | Broad retrieval (top-20) → precision reranking (top-5) | No improvement (baseline recall already perfect) | When initial retrieval has good recall but poor precision |
| **Multi-Query Retrieval** | Generate query variants, retrieve with each, fuse | Not needed (queries are unambiguous) | Ambiguous or complex queries; diverse user phrasings |
| **HyDE** | Generate hypothetical answer, embed it, retrieve similar docs | Not needed (no semantic gap) | Vague queries ("tell me about budget stuff") |
| **Self-RAG** | LLM decides when/whether to retrieve | Overkill for structured queries | When some queries need retrieval and others don't |
| **CRAG** | Evaluate retrieval quality, fallback to web search if poor | Not applicable (closed corpus) | Open-domain QA with knowledge base gaps |
| **Graph RAG** | Entity-relationship graph traversal + retrieval | Not needed (no multi-hop reasoning) | Cross-document reasoning, relationship discovery |
| **RAPTOR** | Recursive tree of summaries at multiple abstraction levels | Not needed (flat corpus) | Long documents needing multi-level abstraction |

### 3.4 Generation & Hallucination Control

| Approach | Description | Our Implementation | Effectiveness |
|----------|-------------|-------------------|---------------|
| **Template-based extraction** | Pattern match query type → extract from structured fields | V1 & V2 | 100% accuracy, 0% hallucination — deterministic |
| **Faithfulness prompting** | Instruct LLM to only use retrieved context | Not needed (no LLM generation) | Reduces hallucination ~30-50% in LLM-based systems |
| **Citation-grounded** | Force LLM to cite sources for each claim | Not needed | Essential for user-facing LLM answers |
| **Map-reduce** | Summarize each doc independently, then combine | Not applicable (short docs) | Long documents exceeding context window |
| **Detail-pattern detection** | Regex patterns detect unanswerable specificity questions | V1 & V2 | 100% — catches all 25 unanswerable queries |

**Design insight:** Template-based generation is *superior* to LLM generation for this task because every query has a deterministic answer extractable from structured email metadata. Using an LLM would introduce unnecessary hallucination risk and latency.

---

## 4. Scalability Analysis

| Technique | Index Build | Query Latency | Memory | Scaling Notes |
|-----------|-------------|---------------|--------|---------------|
| BM25 | O(N×L) | O(V×D_avg) | O(N×V) | Inverted index; scales to millions with sharding |
| TF-IDF | O(N×L) | O(V×N) | O(N×V) | Sparse matrix; needs scipy.sparse at scale |
| LSA (SVD) | O(N×V×k) | O(k) | O(N×k) | Dense; k≪V; very fast queries once built |
| Neural Dense | O(N×model) | O(N×d) or ANN | O(N×d) | Needs HNSW/FAISS for sub-linear query time |
| RRF Fusion | 2-3× base | Sum of parts | Sum | Linear overhead per method |
| Contextual | +LLM/chunk | Same as base | +30% text | One-time indexing cost |
| Propositions | +LLM/chunk | O(P×V), P≫N | O(P×V) | 5-7× more chunks; higher recall potential |
| Cross-Encoder | N/A | O(k×model) | +model | Only top-k candidates; ~100ms for k=100 |
| HNSW (ANN) | O(N×log N) | O(log N) | O(N×d) | Sub-millisecond at 1B vectors |

**Scale recommendations:**

- **100–10K docs:** BM25+TF-IDF RRF (current approach) — fast, simple, effective
- **10K–100K docs:** Add FAISS HNSW for dense retrieval + cross-encoder reranking
- **100K–1M docs:** Qdrant/Weaviate vector DB, SPLADE for learned sparse, Cohere rerank
- **1M+ docs:** Milvus/Pinecone, hierarchical retrieval (RAPTOR), sharding, approximate methods

---

## 5. When Each Technique Becomes Essential

| Scenario | Techniques That Win | Why |
|----------|-------------------|-----|
| Paraphrased queries ("Who emailed about money stuff?") | Dense embeddings, SPLADE | Semantic bridging across vocabulary gaps |
| Cross-document reasoning ("Which team has the most issues?") | Graph RAG, multi-hop retrieval | Explicit relationship modeling |
| 100K+ documents | HNSW vector index, cross-encoder reranking | Sub-linear search, precision at scale |
| Domain-specific vocabulary | SPLADE, instruction-tuned embeddings | Learned term expansion, task-specific encoding |
| Multi-hop questions ("Who manages the person who sent...") | Iterative retrieval, RAPTOR | Multi-step reasoning with retrieval |
| Mixed document types (PDFs, tables, code) | Proposition chunking, contextual retrieval | Normalize heterogeneous content into atomic facts |
| User-facing answers requiring citations | Citation-grounded LLM generation | Traceability and verifiability |
| Noisy/incomplete documents | Contextual chunking, robust parsing | Context recovery for isolated chunks |

---

## 6. Conclusion

For the current email RAG task, the **V1 baseline (BM25 + TF-IDF with RRF)** is the optimal architecture. It achieves perfect scores while being the simplest and fastest approach. The benchmarks prove that more complex techniques — dense embeddings, query expansion, reranking, contextual chunking, proposition-based retrieval — add latency and complexity without improving results.

This is the correct engineering outcome: **match the technique to the problem**. Sparse retrieval dominates for keyword-matching over small corpora. The advanced techniques surveyed here would become essential as the problem scales in corpus size, query complexity, or vocabulary diversity.

The V2 benchmark script (`rag_pipeline_v2.py`) provides a reusable framework for re-evaluating these tradeoffs as the dataset or query patterns change.
