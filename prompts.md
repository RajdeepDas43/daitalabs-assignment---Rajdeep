# Skills for Reproducing the Mini RAG System with Claude Code

This document contains the complete, structured prompt instructions needed to reproduce this RAG pipeline solution using Claude Code (or any Claude-based coding assistant). The prompts follow Anthropic's best practices for prompt engineering: clear role assignment, structured task decomposition, explicit constraints, worked examples, and verification steps.

---

## Prompt 1: Understand the Task

```
<role>
You are a senior ML engineer building a Retrieval-Augmented Generation (RAG) pipeline from scratch. You write clean, well-documented Python. You do NOT use end-to-end RAG frameworks like LangChain or LlamaIndex.
</role>

<task>
Read and fully understand the task defined in the repository. The key files are:
- README.md — task requirements, constraints, evaluation metrics
- test_queries.json — 100 test queries (75 answerable, 25 unanswerable) with gold labels
- emails/ — 100 synthetic emails, ~12% with data quality issues

Before writing any code:
1. Read README.md completely and summarize the requirements
2. Read test_queries.json and categorize the query types you see (sender lookup, recipient lookup, email address extraction, domain extraction, subject lookup, unanswerable detail questions)
3. Read at least 10 emails including noisy ones (email_009.txt, email_018.txt, email_038.txt, email_051.txt, email_058.txt, email_063.txt, email_072.txt, email_093.txt) and catalog every noise type
4. List all noise types found with an example of each

Do NOT write code yet. Only analyze and report.
</task>
```

---

## Prompt 2: Build the Email Parser

```
<task>
Build a robust email parser in Python that handles ALL noise types found in the dataset.

<requirements>
- Parse Subject, From (name + email), To (name + email), and Body from each email file
- Handle these specific noise types:
  1. Mojibake encoding (â€™ → ', â€œ → ", etc.) — clean before parsing
  2. Truncated content (email_018.txt) — parse whatever headers are intact
  3. Missing From: header (email_038.txt) — extract sender name from sign-off at bottom
  4. Swapped header order: To: before From: (email_051.txt) — parse headers regardless of order
  5. Duplicate headers (email_072.txt) — deduplicate, take first occurrence
  6. Scattered typos in body (email_058.txt) — no special handling needed, headers are clean
  7. Missing greeting line (email_063.txt) — no special handling needed
  8. [EXTERNAL][BULK] tag prefixes on subject (email_093.txt) — strip tag prefixes

<critical_detail>
The email format has blank lines BETWEEN headers:
```
Subject: Topic

From: Name <email>
To: Name <email>

Body text...
```
Your parser must NOT stop at the first blank line after Subject. It must continue scanning for From: and To: headers across blank lines. Use a two-consecutive-blank-lines rule or scan until the first non-header, non-blank line after finding From/To.
</critical_detail>

<output_format>
Return a dict per email with keys: filename, subject, from_name, from_email, to_name, to_email, body, raw_text
</output_format>

Test the parser on all 100 emails and report how many have missing fields.
</requirements>
</task>
```

---

## Prompt 3: Build Chunking + Embedding + Retrieval

```
<task>
Build the retrieval pipeline with these specific components:

<chunking_strategy>
Use document-level chunking: one chunk per email. Emails are ~150 words, so splitting further would lose context. Prepend structured metadata to each chunk to make names, emails, domains, and subjects highly searchable:

```
Email: {filename}
Subject: {subject}
From: {from_name}
From Email: {from_email}
From Domain: {domain}
To: {to_name}
To Email: {to_email}
To Domain: {domain}

Body:
{body}
```

Explain why this strategy is optimal for this dataset.
</chunking_strategy>

<embedding_approach>
Implement a hybrid BM25 + TF-IDF approach:

1. BM25 (Okapi BM25): Implement from scratch with k1=1.5, b=0.75. This excels at keyword matching for person names and subjects.

2. TF-IDF: Implement from scratch with augmented term frequency (0.5 + 0.5 * tf/max_tf) and smooth IDF. Normalize vectors for cosine similarity.

3. Reciprocal Rank Fusion (RRF): Combine both methods using RRF with k=60. For each document, score = sum(1/(k + rank_i + 1)) across methods.

Do NOT use dense/neural embeddings. Explain why sparse methods are superior for this specific task (keyword-heavy queries, proper nouns, small corpus).
</embedding_approach>

<retrieval>
Return top-5 results from the fused ranking. The retrieve function should accept a query string and return [(email_dict, chunk_text, score), ...].
</retrieval>
</task>
```

---

## Prompt 4: Build the Answer Generator

```
<task>
Build an answer generator that extracts answers from retrieved email metadata. This is a template-based extraction system, NOT an LLM-based generator.

<query_classification>
Classify each query into one of these types and handle accordingly:

1. "Who sent X to Y?" → Find sender by matching recipient name + topic in retrieved emails
2. "Who did X send Y to?" → Find recipient by matching sender name + topic
3. "What is X's email address?" → Extract email address field for the named person
4. "What domain does X work at?" → Extract domain from email address for the named person
5. "What subject did X write to Y about?" → Extract subject line for sender-recipient pair
6. Variations: "Who reported...", "Who did X request... from", "Who did X schedule... with"

Use regex patterns to match each query type. The patterns must handle:
- Different phrasings ("sent", "wrote", "reported", "scheduled")
- Optional articles ("a", "an", "the")
- Trailing punctuation
</query_classification>

<hallucination_control>
This is CRITICAL for the 25 unanswerable queries. Implement a detail-question detector:

1. Before any answer extraction, check if the query asks for specific details that template emails don't contain. Use regex patterns for:
   - "what specific..." / "what exact..."
   - "how many..."
   - "what is the name of the vendor/client/company/certification"
   - "what dollar/budget amount"
   - "what day/date/time"
   - "phone number"
   - "what revised deadline"
   - "what are the agenda items"
   - Any request for granular details (error codes, equipment lists, revenue targets, team names)

2. If detected as a detail question, return a decline response containing phrases like "not available", "does not specify", "this specific information is not".

3. NEVER fall through to the generic fallback for unanswerable queries — the fallback produces confident answers from metadata, which counts as hallucination.
</hallucination_control>

<important>
The answer text MUST contain the reference_answer as a case-insensitive substring for accuracy scoring. Design your answer templates to include the person's name, email, domain, or subject verbatim.
</important>
</task>
```

---

## Prompt 5: Build the Evaluation Harness

```
<task>
Build an evaluation function that computes three metrics against test_queries.json:

1. Recall@5: For each answerable query, check if any source_email appears in the top-5 retrieved filenames. Report fraction.
   Target: >= 0.80

2. Answer Accuracy: For each answerable query, check if reference_answer (lowercase) is a substring of the generated answer (lowercase). Report fraction.
   Target: >= 0.75

3. Hallucination Rate: For each unanswerable query, check if the generated answer contains ANY decline phrase from this list: ["not available", "cannot find", "could not find", "does not specify", "does not mention", "no phone numbers", "this information is not", "this specific"]. If NONE of these phrases appear, count it as a hallucination. Report fraction.
   Target: <= 0.20

Print:
- Per-query failures with diagnostic details (query text, expected vs actual)
- Summary table with all three metrics, their values, targets, and pass/fail
- Demo of 4 sample queries showing Q, A, and retrieved files

Run the evaluation and iterate until ALL THREE targets are met.
</task>
```

---

## Prompt 6: Build the Auto-Learning Module (V2-Integrated)

```
<task>
Build an auto-learning module that integrates with the V2 benchmark pipeline (rag_pipeline_v2.py), not V1. It implements a feedback loop across ALL 11 retrieval strategies:

tournament (evaluate all strategies) → pick best → analyze failures → apply fixes → re-evaluate → log

<architecture>
1. StrategyRegistry: Manages all 11 V2 strategies with tunable parameters. Each strategy entry includes: name, chunking function, retriever builder, parameter dict (k1, b, rrf_k, top_k, use_reranker, use_expander). Can rebuild indices when parameters change.

2. DetailedEvaluator: Returns per-query diagnostics (not just aggregate metrics). For each query, capture: recall_hit, accuracy_hit, correct_rank (where in top-5 was the correct email), hallucination flag, generated answer, retrieved files.

3. FailureAnalyzer: Takes per-query results, groups failures into patterns:
   - RECALL_NAME_MISMATCH — person name didn't match any top-5 result
   - RECALL_NOISE_RELATED — failure on known noisy emails
   - ACCURACY_EXTRACTION_FAIL — correct email at #1 but answer extraction failed
   - ACCURACY_RANKING_ISSUE — correct email in top-5 but not #1
   - HALLUCINATION_UNDETECTED — unanswerable query not caught by detail detector
   - HALLUCINATION_LEAKED — detail question that leaked through to fallback
   - RANKING_FRAGILE — correct email at rank 2-5 (works but fragile)
   - OPTIMAL — no issues found
   Each pattern includes severity, count, affected queries, and suggested actions.

4. AutoFixer: Applies specific fixes, including cross-strategy switching:
   - SWITCH_STRATEGY: Switch to a better-performing strategy from the tournament
   - TUNE_RRF_K: Adjust RRF k (60 → 40 for stronger top-rank weighting)
   - TUNE_BM25_K1: Adjust k1 (1.5 → 1.8 for higher saturation)
   - EXPAND_TOP_K: Increase retrieval depth (5 → 10)
   - BOOST_NAME_WEIGHT: Triple name tokens in chunks to boost BM25 weight
   - BOOST_TOPIC_WEIGHT: Double subject tokens in chunks
   - ENRICH_CHUNKS: Add contextual summaries to chunks
   - ADD_DETAIL_PATTERN: Generate new regex patterns from failing queries

5. LearningStore: Persist cycle history to disk as JSON. Track: metrics before/after, patterns found, actions applied, deltas, convergence.

6. AutoLearningRAG (Orchestrator): Three phases:
   a. Tournament phase — evaluate all 11 strategies, rank by composite score (recall + accuracy - hallucination), break ties by speed
   b. Learning cycles — run fix→re-eval loops on the winning strategy
   c. Convergence detection — stop on perfect scores or plateau (no metric change)
</architecture>

<constraints>
- Import from rag_pipeline_v2.py — use ALL V2 components (parse_email, chunk_standard, chunk_contextual, chunk_parent_child, chunk_propositions, tokenize, BM25, TFIDFRetriever, LSARetriever, QueryExpander, CrossEncoderReranker, rrf_fuse, retrieve_top_k, generate_answer, is_detail_question)
- Do NOT duplicate the core pipeline code
- Use argparse for --cycles and --store flags
- Print clear progress at each step with emoji severity indicators
- Print a tournament table, per-cycle diagnostics, and a learning summary table at the end
</constraints>
</task>
```

---

## Prompt 7: Build the Benchmark Comparison (V2)

```
<task>
Build a benchmark script (rag_pipeline_v2.py) that implements and compares 11 retrieval strategies:

1. BM25 Only
2. TF-IDF Only
3. BM25 + TF-IDF RRF (baseline)
4. LSA Dense (TruncatedSVD on TF-IDF, 50 dimensions) — simulates neural embeddings
5. BM25 + Query Expansion (SPLADE-like synonym mapping)
6. Full Hybrid (BM25 + TF-IDF + LSA via 3-way RRF)
7. Contextual Chunks + BM25+TF-IDF (Anthropic's contextual retrieval)
8. Contextual + Full Hybrid
9. Parent-Child (BM25 on compact metadata chunks, return full email)
10. Proposition-Based (decompose emails into atomic facts, index each)
11. Baseline + Simulated Reranker (retrieve top-20, rerank to top-5)

For each strategy, measure: Recall@5, Accuracy, Hallucination Rate, Wall-clock time.

Print a comparison table and per-technique analysis explaining WHY each technique helps or hurts on this specific dataset. Include a scalability analysis table showing Big-O complexity for each technique.

Use numpy and sklearn (TfidfVectorizer, TruncatedSVD, cosine_similarity) for the LSA implementation. Install with: pip install numpy scikit-learn --break-system-packages
</task>
```

---

## Prompt 8: Write the README

```
<task>
Write a comprehensive README.md that covers:

1. Project overview and architecture diagram (ASCII)
2. How to run each component (pipeline, benchmark, auto-learner)
3. Evaluation results table
4. Design decisions with rationale for each choice
5. How malformed emails are handled (with specific examples)
6. File structure explanation
7. Scalability analysis and when to switch techniques

Use clear headings, code blocks for commands, and keep it professional.
Do NOT use emojis. Keep prose concise.
</task>
```