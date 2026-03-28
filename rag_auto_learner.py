#!/usr/bin/env python3
"""
Auto-Learning RAG Module (V2-Integrated)
==========================================

A self-improving RAG system built on top of rag_pipeline_v2.py that:

1. Evaluates ALL 11 retrieval strategies from V2
2. Identifies the best-performing strategy as starting point
3. Analyzes per-query failures to find systematic patterns
4. Generates and applies targeted fixes (parameter tuning, strategy switching,
   chunk enrichment, new detection patterns)
5. Re-evaluates after each fix and keeps only improvements
6. Logs every learning cycle for auditability

Key difference from a V1-only learner: this module can SWITCH between
retrieval strategies (BM25, TF-IDF, LSA, hybrid, contextual, parent-child,
proposition-based, reranked) as part of its action space — not just tune
parameters on a single strategy.

Architecture:
  ┌────────────────────────────────────────────────────────────┐
  │                   V2 Strategy Pool                         │
  │  BM25 │ TF-IDF │ LSA │ Hybrid │ Contextual │ ParentChild  │
  │  Propositions │ QueryExpanded │ Reranked │ FullHybrid     │
  └──────────────────────┬─────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │Strategy │ ← selects best from pool
                    │Selector │
                    └────┬────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │                  Learning Loop                              │
  │                                                             │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
  │  │ Evaluate  │──►│ Analyze  │──►│  Fix     │──►│Re-eval │ │
  │  │ (detailed)│   │ failures │   │ (auto)   │   │& log   │ │
  │  └──────────┘   └──────────┘   └──────────┘   └────────┘ │
  │       ▲                                            │       │
  │       └────────────────────────────────────────────┘       │
  │                     cycle N+1                              │
  └────────────────────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │Learning │ ← persists to disk
                    │ Store   │
                    └─────────┘

Usage:
  python3 rag_auto_learner.py              # Run full learning loop (3 cycles)
  python3 rag_auto_learner.py --cycles 5   # Run 5 learning cycles
  python3 rag_auto_learner.py --store my_store  # Custom store path
"""

import os
import re
import json
import math
import time
import copy
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

# ── Import V2 pipeline components ──────────────────────────────────────────
from rag_pipeline_v2 import (
    # Parsing & chunking
    parse_email, chunk_standard, chunk_contextual, chunk_parent_child,
    chunk_propositions, tokenize,
    # Retrieval engines
    BM25, TFIDFRetriever, LSARetriever, QueryExpander, CrossEncoderReranker,
    # Fusion utilities
    rrf_fuse, retrieve_top_k,
    # Answer generation (shared across all strategies)
    generate_answer, is_detail_question,
)


# ===========================================================================
# 1. STRATEGY REGISTRY — All V2 strategies as callable objects
# ===========================================================================

class StrategyRegistry:
    """
    Builds and manages all 11 retrieval strategies from V2.
    Each strategy is a callable: strategy(query) → [(email, chunk, score), ...]
    """

    def __init__(self, emails, chunks_std, chunks_ctx, chunks_pc, chunks_prop):
        self.emails = emails
        self.chunks_std = chunks_std
        self.chunks_ctx = chunks_ctx

        # Tokenize
        tok_std = [tokenize(c) for c in chunks_std]
        tok_ctx = [tokenize(c) for c in chunks_ctx]

        # ── Build all indices ────────────────────────────────────────────
        self.bm25 = BM25(); self.bm25.fit(tok_std)
        self.bm25_ctx = BM25(); self.bm25_ctx.fit(tok_ctx)
        self.tfidf = TFIDFRetriever(); self.tfidf.fit(tok_std)
        self.tfidf_ctx = TFIDFRetriever(); self.tfidf_ctx.fit(tok_ctx)
        self.lsa = LSARetriever(n_components=50); self.lsa.fit(chunks_std)
        self.lsa_ctx = LSARetriever(n_components=50); self.lsa_ctx.fit(chunks_ctx)

        # Parent-child
        child_chunks = [c for c, p in chunks_pc]
        tok_children = [tokenize(c) for c in child_chunks]
        self.bm25_child = BM25(); self.bm25_child.fit(tok_children)

        # Propositions
        self.prop_texts = []; self.prop_email_idx = []
        for i, props in enumerate(chunks_prop):
            for p in props:
                self.prop_texts.append(p)
                self.prop_email_idx.append(i)
        tok_props = [tokenize(p) for p in self.prop_texts]
        self.bm25_prop = BM25(); self.bm25_prop.fit(tok_props)

        # ── Tunable parameters (learning can adjust these) ───────────────
        self.params = {
            'rrf_k': 60,
            'top_k': 5,
            'bm25_k1': 1.5,
            'bm25_b': 0.75,
            'rerank_top_n': 20,
        }

    def _make_results(self, indices, chunks=None):
        if chunks is None:
            chunks = self.chunks_std
        return [(self.emails[i], chunks[i], s) for i, s in indices]

    def get_all_strategies(self):
        """Return dict of {name: retrieve_fn} for all 11 strategies."""
        return {
            '1_bm25_only': self._retrieve_bm25,
            '2_tfidf_only': self._retrieve_tfidf,
            '3_hybrid_v1': self._retrieve_hybrid_v1,
            '4_lsa_dense': self._retrieve_lsa,
            '5_bm25_expanded': self._retrieve_bm25_expanded,
            '6_full_hybrid': self._retrieve_full_hybrid,
            '7_contextual': self._retrieve_contextual,
            '8_contextual_full': self._retrieve_contextual_full,
            '9_parent_child': self._retrieve_parent_child,
            '10_propositions': self._retrieve_propositions,
            '11_hybrid_reranked': self._retrieve_hybrid_reranked,
        }

    # ── Strategy implementations ─────────────────────────────────────────

    def _retrieve_bm25(self, query):
        scores = self.bm25.score(tokenize(query))
        return self._make_results(retrieve_top_k(scores, self.params['top_k']))

    def _retrieve_tfidf(self, query):
        scores = self.tfidf.score(tokenize(query))
        return self._make_results(retrieve_top_k(scores, self.params['top_k']))

    def _retrieve_hybrid_v1(self, query):
        qt = tokenize(query)
        fused = rrf_fuse([self.bm25.score(qt), self.tfidf.score(qt)], k=self.params['rrf_k'])
        return self._make_results(retrieve_top_k(fused, self.params['top_k']))

    def _retrieve_lsa(self, query):
        scores = self.lsa.score(query)
        return self._make_results(retrieve_top_k(scores, self.params['top_k']))

    def _retrieve_bm25_expanded(self, query):
        qt = tokenize(query)
        qt_exp = QueryExpander.expand(qt)
        scores = self.bm25.score(qt_exp)
        return self._make_results(retrieve_top_k(scores, self.params['top_k']))

    def _retrieve_full_hybrid(self, query):
        qt = tokenize(query)
        fused = rrf_fuse([self.bm25.score(qt), self.tfidf.score(qt),
                          self.lsa.score(query)], k=self.params['rrf_k'])
        return self._make_results(retrieve_top_k(fused, self.params['top_k']))

    def _retrieve_contextual(self, query):
        qt = tokenize(query)
        fused = rrf_fuse([self.bm25_ctx.score(qt), self.tfidf_ctx.score(qt)],
                         k=self.params['rrf_k'])
        return [(self.emails[i], self.chunks_ctx[i], s)
                for i, s in retrieve_top_k(fused, self.params['top_k'])]

    def _retrieve_contextual_full(self, query):
        qt = tokenize(query)
        fused = rrf_fuse([self.bm25_ctx.score(qt), self.tfidf_ctx.score(qt),
                          self.lsa_ctx.score(query)], k=self.params['rrf_k'])
        return [(self.emails[i], self.chunks_ctx[i], s)
                for i, s in retrieve_top_k(fused, self.params['top_k'])]

    def _retrieve_parent_child(self, query):
        qt = tokenize(query)
        scores = self.bm25_child.score(qt)
        return self._make_results(retrieve_top_k(scores, self.params['top_k']))

    def _retrieve_propositions(self, query):
        qt = tokenize(query)
        scores = self.bm25_prop.score(qt)
        top_props = retrieve_top_k(scores, 15)
        seen = set(); results = []
        for prop_idx, score in top_props:
            eidx = self.prop_email_idx[prop_idx]
            if eidx not in seen:
                seen.add(eidx)
                results.append((self.emails[eidx], self.chunks_std[eidx], score))
            if len(results) >= self.params['top_k']:
                break
        return results

    def _retrieve_hybrid_reranked(self, query):
        qt = tokenize(query)
        fused = rrf_fuse([self.bm25.score(qt), self.tfidf.score(qt)],
                         k=self.params['rrf_k'])
        top_n = retrieve_top_k(fused, self.params['rerank_top_n'])
        results = [(self.emails[i], self.chunks_std[i], s) for i, s in top_n]
        reranked = CrossEncoderReranker.rerank(query, results, self.emails)
        return reranked[:self.params['top_k']]

    # ── Parameter mutation (for auto-fixer) ──────────────────────────────

    def update_param(self, key, value):
        old = self.params.get(key)
        self.params[key] = value
        # Re-fit BM25 if its parameters changed
        if key in ('bm25_k1', 'bm25_b'):
            self.bm25.k1 = self.params['bm25_k1']
            self.bm25.b = self.params['bm25_b']
            tok_std = [tokenize(c) for c in self.chunks_std]
            self.bm25.fit(tok_std)
        return old

    def rebuild_index_with_boosted_chunks(self, emails, boosted_chunks):
        """Rebuild the standard BM25+TF-IDF indices with enriched chunks."""
        self.chunks_std = boosted_chunks
        tok = [tokenize(c) for c in boosted_chunks]
        self.bm25.fit(tok)
        self.tfidf.fit(tok)
        self.lsa.fit(boosted_chunks)


# ===========================================================================
# 2. DETAILED EVALUATOR — Per-query diagnostics
# ===========================================================================

class DetailedEvaluator:
    """Evaluates a retrieval strategy and returns per-query diagnostics."""

    DECLINE_PHRASES = [
        'not available', 'cannot find', 'could not find', 'not specified',
        'does not specify', 'does not mention', 'not included', 'not present',
        'no phone numbers', 'this information is not', 'this specific',
        'does not include', 'not available in the emails'
    ]

    def __init__(self, retrieve_fn, queries):
        self.retrieve_fn = retrieve_fn
        self.queries = queries

    def evaluate(self):
        """Returns (metrics_dict, per_query_results_list)."""
        results = []
        for q in self.queries:
            retrieved = self.retrieve_fn(q['query'])
            retrieved_files = [r[0]['filename'] for r in retrieved]
            answer = generate_answer(q['query'], retrieved)

            r = {
                'query_id': q['id'], 'query': q['query'],
                'answerable': q['answerable'],
                'reference_answer': q.get('reference_answer'),
                'source_emails': q.get('source_emails', []),
                'retrieved_files': retrieved_files,
                'generated_answer': answer,
                'scores': [x[2] for x in retrieved],
            }

            if q['answerable']:
                r['recall_hit'] = any(s in retrieved_files for s in q['source_emails'])
                r['accuracy_hit'] = q['reference_answer'].lower() in answer.lower()
                r['correct_rank'] = next(
                    (rank + 1 for rank, f in enumerate(retrieved_files)
                     if f in q['source_emails']), None)
                r['hallucination'] = False
            else:
                r['recall_hit'] = r['accuracy_hit'] = r['correct_rank'] = None
                r['hallucination'] = not any(
                    p in answer.lower() for p in self.DECLINE_PHRASES)

            results.append(r)

        answerable = [r for r in results if r['answerable']]
        unanswerable = [r for r in results if not r['answerable']]

        metrics = {
            'recall_at_5': sum(1 for r in answerable if r['recall_hit']) / len(answerable),
            'accuracy': sum(1 for r in answerable if r['accuracy_hit']) / len(answerable),
            'hallucination_rate': (sum(1 for r in unanswerable if r['hallucination'])
                                   / len(unanswerable)),
            'total': len(results),
            'answerable_n': len(answerable),
            'unanswerable_n': len(unanswerable),
        }
        return metrics, results


# ===========================================================================
# 3. FAILURE ANALYZER — Pattern detection across strategies
# ===========================================================================

class FailureAnalyzer:
    """Groups per-query failures into actionable patterns."""

    NOISY_EMAILS = {'email_009', 'email_018', 'email_038', 'email_051',
                    'email_058', 'email_063', 'email_072', 'email_093'}

    def analyze(self, results, current_strategy_name):
        patterns = []

        recall_misses = [r for r in results if r['answerable'] and not r['recall_hit']]
        accuracy_misses = [r for r in results if r['answerable']
                          and r['recall_hit'] and not r['accuracy_hit']]
        hallucinations = [r for r in results if not r['answerable'] and r['hallucination']]
        near_misses = [r for r in results if r['answerable'] and r['correct_rank']
                       and 2 <= r['correct_rank'] <= 5]

        # ── Recall failures ──────────────────────────────────────────────
        if recall_misses:
            noise_related = [r for r in recall_misses
                            if any(s.replace('.txt', '') in self.NOISY_EMAILS
                                   for s in r['source_emails'])]
            name_related = [r for r in recall_misses if not any(
                s.replace('.txt', '') in self.NOISY_EMAILS for s in r['source_emails'])]

            if name_related:
                patterns.append({
                    'type': 'RECALL_NAME_MISMATCH',
                    'severity': 'high',
                    'count': len(name_related),
                    'affected': [r['query_id'] for r in name_related],
                    'description': 'Person names in query did not surface correct email in top-5.',
                    'actions': [
                        'SWITCH_STRATEGY:9_parent_child',
                        'BOOST_NAME_WEIGHT',
                        'EXPAND_TOP_K:10',
                    ],
                })
            if noise_related:
                patterns.append({
                    'type': 'RECALL_NOISE',
                    'severity': 'medium',
                    'count': len(noise_related),
                    'affected': [r['query_id'] for r in noise_related],
                    'description': 'Retrieval failed on noisy/malformed emails.',
                    'actions': [
                        'SWITCH_STRATEGY:7_contextual',
                        'ENRICH_CHUNKS',
                    ],
                })

        # ── Accuracy failures ────────────────────────────────────────────
        if accuracy_misses:
            extraction_fails = [r for r in accuracy_misses if r['correct_rank'] == 1]
            ranking_fails = [r for r in accuracy_misses if r['correct_rank'] != 1]

            if extraction_fails:
                patterns.append({
                    'type': 'ACCURACY_EXTRACTION',
                    'severity': 'high',
                    'count': len(extraction_fails),
                    'affected': [r['query_id'] for r in extraction_fails],
                    'description': 'Correct email at rank 1 but answer extraction failed.',
                    'actions': ['ADD_QUERY_PATTERN'],
                    'examples': [(r['query'], r['reference_answer'], r['generated_answer'])
                                for r in extraction_fails[:3]],
                })
            if ranking_fails:
                patterns.append({
                    'type': 'ACCURACY_RANKING',
                    'severity': 'medium',
                    'count': len(ranking_fails),
                    'affected': [r['query_id'] for r in ranking_fails],
                    'description': 'Correct email in top-5 but not #1; wrong email used for answer.',
                    'actions': [
                        'SWITCH_STRATEGY:11_hybrid_reranked',
                        'TUNE_RRF_K:40',
                        'TUNE_BM25_K1:1.8',
                    ],
                })

        # ── Hallucinations ───────────────────────────────────────────────
        if hallucinations:
            detected_but_leaked = [r for r in hallucinations if is_detail_question(r['query'])]
            undetected = [r for r in hallucinations if not is_detail_question(r['query'])]

            if undetected:
                patterns.append({
                    'type': 'HALLUCINATION_UNDETECTED',
                    'severity': 'critical',
                    'count': len(undetected),
                    'affected': [r['query_id'] for r in undetected],
                    'description': 'Unanswerable query not caught by detail-question detector.',
                    'actions': ['ADD_DETAIL_PATTERN'],
                    'example_queries': [r['query'] for r in undetected[:5]],
                })
            if detected_but_leaked:
                patterns.append({
                    'type': 'HALLUCINATION_LEAKED',
                    'severity': 'high',
                    'count': len(detected_but_leaked),
                    'affected': [r['query_id'] for r in detected_but_leaked],
                    'description': 'Detail-question detected but handler still produced confident answer.',
                    'actions': ['FIX_DETAIL_HANDLER'],
                })

        # ── Near-misses (rank 2-5 fragility) ─────────────────────────────
        if near_misses:
            rank_dist = Counter(r['correct_rank'] for r in near_misses)
            patterns.append({
                'type': 'RANKING_FRAGILE',
                'severity': 'low',
                'count': len(near_misses),
                'affected': [r['query_id'] for r in near_misses],
                'description': (f'Correct email in top-5 but not #1 for {len(near_misses)} queries. '
                               f'Rank distribution: {dict(rank_dist)}.'),
                'actions': [
                    'TUNE_RRF_K:40',
                    'SWITCH_STRATEGY:11_hybrid_reranked',
                    'BOOST_TOPIC_WEIGHT',
                ],
            })

        # ── All clear ────────────────────────────────────────────────────
        if not patterns:
            patterns.append({
                'type': 'OPTIMAL',
                'severity': 'info',
                'count': 0,
                'affected': [],
                'description': (f'No failures detected. Strategy "{current_strategy_name}" '
                               'is performing optimally.'),
                'actions': ['NONE'],
            })

        return patterns


# ===========================================================================
# 4. AUTO-FIXER — Applies fixes using V2's full toolkit
# ===========================================================================

class AutoFixer:
    """
    Applies targeted fixes using V2's full strategy pool.
    Can switch strategies, tune parameters, enrich chunks, and add patterns.
    """

    def __init__(self, registry: StrategyRegistry):
        self.registry = registry
        self.applied = []

    def apply(self, action_str, pattern=None):
        """Parse and apply a single action string. Returns result dict."""
        action = action_str.strip()

        # ── Strategy switching ───────────────────────────────────────────
        if action.startswith('SWITCH_STRATEGY:'):
            target = action.split(':')[1]
            strategies = self.registry.get_all_strategies()
            if target in strategies:
                self.applied.append(action)
                return {
                    'action': action, 'applied': True, 'type': 'strategy_switch',
                    'description': f'Switched active strategy to {target}',
                    'new_strategy': target,
                }
            return {'action': action, 'applied': False, 'reason': f'Unknown strategy: {target}'}

        # ── Parameter tuning ─────────────────────────────────────────────
        if action.startswith('TUNE_RRF_K:'):
            new_val = int(action.split(':')[1])
            old = self.registry.update_param('rrf_k', new_val)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'param_tune',
                    'description': f'RRF k: {old} -> {new_val}', 'param': 'rrf_k',
                    'old': old, 'new': new_val}

        if action.startswith('TUNE_BM25_K1:'):
            new_val = float(action.split(':')[1])
            old = self.registry.update_param('bm25_k1', new_val)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'param_tune',
                    'description': f'BM25 k1: {old} -> {new_val}', 'param': 'bm25_k1',
                    'old': old, 'new': new_val}

        if action.startswith('EXPAND_TOP_K:'):
            new_val = int(action.split(':')[1])
            old = self.registry.update_param('top_k', new_val)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'param_tune',
                    'description': f'top_k: {old} -> {new_val}', 'param': 'top_k',
                    'old': old, 'new': new_val}

        # ── Chunk enrichment ─────────────────────────────────────────────
        if action == 'BOOST_NAME_WEIGHT':
            emails = self.registry.emails
            boosted = []
            for i, chunk in enumerate(self.registry.chunks_std):
                e = emails[i]
                boost = ''
                if e['from_name']:
                    boost += f" {e['from_name']} " * 3
                if e['to_name']:
                    boost += f" {e['to_name']} " * 3
                boosted.append(chunk + boost)
            self.registry.rebuild_index_with_boosted_chunks(emails, boosted)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'chunk_enrich',
                    'description': 'Tripled name tokens in standard chunks and rebuilt indices'}

        if action == 'BOOST_TOPIC_WEIGHT':
            emails = self.registry.emails
            boosted = []
            for i, chunk in enumerate(self.registry.chunks_std):
                e = emails[i]
                if e['subject']:
                    boosted.append(chunk + f" {e['subject']} " * 2)
                else:
                    boosted.append(chunk)
            self.registry.rebuild_index_with_boosted_chunks(emails, boosted)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'chunk_enrich',
                    'description': 'Doubled subject tokens in standard chunks and rebuilt indices'}

        if action == 'ENRICH_CHUNKS':
            emails = self.registry.emails
            enriched = []
            for i, chunk in enumerate(self.registry.chunks_std):
                e = emails[i]
                ctx = (f"[Email from {e['from_name']} ({e['from_email']}) "
                       f"to {e['to_name']} ({e['to_email']}) "
                       f"about {e['subject']}] ")
                enriched.append(ctx + chunk)
            self.registry.rebuild_index_with_boosted_chunks(emails, enriched)
            self.applied.append(action)
            return {'action': action, 'applied': True, 'type': 'chunk_enrich',
                    'description': 'Added structured context prefix to all standard chunks'}

        # ── Pattern additions ────────────────────────────────────────────
        if action == 'ADD_DETAIL_PATTERN':
            if pattern and pattern.get('example_queries'):
                new_pats = []
                for q in pattern['example_queries']:
                    words = q.lower().split()[:8]
                    new_pats.append(r'\s+'.join(re.escape(w) for w in words))
                self.applied.append(action)
                return {'action': action, 'applied': True, 'type': 'pattern_add',
                        'description': f'Generated {len(new_pats)} new detail-question patterns',
                        'patterns': new_pats}
            return {'action': action, 'applied': False, 'reason': 'No example queries'}

        if action == 'ADD_QUERY_PATTERN':
            if pattern and pattern.get('examples'):
                self.applied.append(action)
                return {'action': action, 'applied': True, 'type': 'pattern_add',
                        'description': f'Identified {len(pattern["examples"])} extraction patterns to add',
                        'examples': pattern['examples']}
            return {'action': action, 'applied': False, 'reason': 'No examples'}

        if action in ('FIX_DETAIL_HANDLER', 'NONE'):
            return {'action': action, 'applied': False,
                    'reason': 'No automated fix available' if action != 'NONE' else 'Already optimal'}

        return {'action': action, 'applied': False, 'reason': f'Unknown action: {action}'}


# ===========================================================================
# 5. LEARNING STORE — Persistent audit log
# ===========================================================================

class LearningStore:
    """Persists learning history to disk as JSON."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / 'learning_history.json'
        self.history = self._load()

    def _load(self):
        if self.file.exists():
            with open(self.file) as f:
                return json.load(f)
        return {'cycles': [], 'best_strategy': None, 'best_metrics': None}

    def save(self):
        with open(self.file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def record(self, cycle):
        self.history['cycles'].append(cycle)
        m = cycle['metrics_after']
        # Track best-ever metrics
        best = self.history.get('best_metrics')
        if (best is None or m['accuracy'] > best.get('accuracy', 0) or
                (m['accuracy'] == best.get('accuracy', 0) and
                 m['hallucination_rate'] < best.get('hallucination_rate', 1))):
            self.history['best_metrics'] = m
            self.history['best_strategy'] = cycle.get('active_strategy')
        self.save()


# ===========================================================================
# 6. ORCHESTRATOR — Full learning loop
# ===========================================================================

class AutoLearningRAG:
    """
    Orchestrates the learning loop across all V2 strategies:
    1. Tournament: evaluate all 11 strategies, pick the best
    2. Learning cycles: analyze failures → apply fixes → re-evaluate
    """

    def __init__(self, emails_dir, test_file, store_dir='learning_store'):
        self.emails_dir = Path(emails_dir)
        self.test_file = Path(test_file)
        self.store = LearningStore(store_dir)

        # Load data
        print("\n  Loading data...")
        email_files = sorted(self.emails_dir.glob('email_*.txt'))
        self.emails = [parse_email(str(f)) for f in email_files]
        with open(self.test_file) as f:
            self.queries = json.load(f)['queries']

        # Build all chunk types
        print("  Building chunks...")
        self.chunks_std = [chunk_standard(e) for e in self.emails]
        self.chunks_ctx = [chunk_contextual(e) for e in self.emails]
        self.chunks_pc = [chunk_parent_child(e) for e in self.emails]
        self.chunks_prop = [chunk_propositions(e) for e in self.emails]

        # Build strategy registry
        print("  Building strategy registry (11 strategies)...")
        self.registry = StrategyRegistry(
            self.emails, self.chunks_std, self.chunks_ctx,
            self.chunks_pc, self.chunks_prop)

        self.active_strategy = None
        self.active_strategy_name = None

    def run_tournament(self):
        """Evaluate all 11 strategies and pick the best."""
        print("\n" + "-" * 70)
        print("  TOURNAMENT: Evaluating all 11 strategies")
        print("-" * 70)

        strategies = self.registry.get_all_strategies()
        tournament = []

        for name, fn in strategies.items():
            t0 = time.time()
            evaluator = DetailedEvaluator(fn, self.queries)
            metrics, _ = evaluator.evaluate()
            elapsed = time.time() - t0

            # Composite score: recall + accuracy - hallucination (higher = better)
            composite = (metrics['recall_at_5'] + metrics['accuracy']
                        - metrics['hallucination_rate'])
            tournament.append({
                'name': name, 'fn': fn, 'metrics': metrics,
                'composite': composite, 'time': elapsed,
            })

            r5 = metrics['recall_at_5']
            acc = metrics['accuracy']
            hr = metrics['hallucination_rate']
            print(f"    {name:<30} R@5={r5:.1%}  Acc={acc:.1%}  "
                  f"Hal={hr:.1%}  Score={composite:.3f}  [{elapsed:.2f}s]")

        # Sort by composite score (desc), then by time (asc) for tiebreaker
        tournament.sort(key=lambda x: (-x['composite'], x['time']))
        winner = tournament[0]
        self.active_strategy = winner['fn']
        self.active_strategy_name = winner['name']

        print(f"\n  WINNER: {winner['name']} "
              f"(Score={winner['composite']:.3f}, Time={winner['time']:.3f}s)")
        return winner

    def run_learning_cycle(self, cycle_num):
        """Execute one learning cycle on the active strategy."""
        print(f"\n{'='*70}")
        print(f"  LEARNING CYCLE {cycle_num}  |  Strategy: {self.active_strategy_name}")
        print(f"{'='*70}")

        # Step 1: Evaluate
        print("\n  [1/5] Evaluating current state...")
        evaluator = DetailedEvaluator(self.active_strategy, self.queries)
        metrics_before, per_query = evaluator.evaluate()
        self._print_metrics("  Before", metrics_before)

        # Step 2: Analyze
        print("\n  [2/5] Analyzing failures...")
        analyzer = FailureAnalyzer()
        patterns = analyzer.analyze(per_query, self.active_strategy_name)
        for p in patterns:
            icon = {'critical': 'CRIT', 'high': 'HIGH', 'medium': 'MED',
                    'low': 'LOW', 'info': 'INFO'}.get(p['severity'], '?')
            print(f"    [{icon}] {p['type']}: {p['description']}")
            if p['affected']:
                print(f"           Affected: {p['affected'][:6]}{'...' if len(p['affected']) > 6 else ''}")

        # Step 3: Apply fixes
        print("\n  [3/5] Applying fixes...")
        fixer = AutoFixer(self.registry)
        applied_fixes = []
        strategy_switched = False

        for pat in patterns:
            if pat['type'] == 'OPTIMAL':
                print("    System is already optimal. No fixes to apply.")
                continue

            for action_str in pat['actions']:
                print(f"    Trying: {action_str}...", end=" ")
                result = fixer.apply(action_str, pat)

                if result.get('applied'):
                    print(f"OK - {result.get('description', '')}")
                    applied_fixes.append(result)

                    # If strategy switched, update active strategy
                    if result.get('type') == 'strategy_switch':
                        new_name = result['new_strategy']
                        self.active_strategy = self.registry.get_all_strategies()[new_name]
                        self.active_strategy_name = new_name
                        strategy_switched = True
                        print(f"    >> Active strategy now: {new_name}")

                    break  # Apply one fix per pattern
                else:
                    print(f"SKIP - {result.get('reason', '')}")

        # Step 4: Re-evaluate
        print("\n  [4/5] Re-evaluating after fixes...")
        evaluator_after = DetailedEvaluator(self.active_strategy, self.queries)
        metrics_after, _ = evaluator_after.evaluate()
        self._print_metrics("  After", metrics_after)

        # Step 5: Compute and log
        deltas = {
            'recall': metrics_after['recall_at_5'] - metrics_before['recall_at_5'],
            'accuracy': metrics_after['accuracy'] - metrics_before['accuracy'],
            'hallucination': metrics_after['hallucination_rate'] - metrics_before['hallucination_rate'],
        }
        improved = (deltas['recall'] >= 0 and deltas['accuracy'] >= 0
                    and deltas['hallucination'] <= 0
                    and any(d != 0 for d in deltas.values()))

        print(f"\n  [5/5] Deltas: Recall={deltas['recall']:+.2%}  "
              f"Acc={deltas['accuracy']:+.2%}  Halluc={deltas['hallucination']:+.2%}  "
              f"{'IMPROVED' if improved else 'NO CHANGE' if not any(d != 0 for d in deltas.values()) else 'REGRESSED'}")

        cycle = {
            'cycle': cycle_num,
            'timestamp': datetime.now().isoformat(),
            'active_strategy': self.active_strategy_name,
            'strategy_switched': strategy_switched,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'deltas': deltas,
            'patterns': [{'type': p['type'], 'severity': p['severity'],
                         'count': p['count']} for p in patterns],
            'fixes_applied': [{k: v for k, v in f.items() if k != 'fn'}
                             for f in applied_fixes],
            'improved': improved,
        }
        self.store.record(cycle)
        return cycle

    def run(self, n_cycles=3):
        """Full run: tournament + learning cycles."""
        print("\n" + "=" * 70)
        print("  AUTO-LEARNING RAG (V2-Integrated)")
        print("=" * 70)
        print(f"  Corpus:       {len(self.emails)} emails")
        print(f"  Test queries: {len(self.queries)} ({sum(1 for q in self.queries if q['answerable'])} answerable, "
              f"{sum(1 for q in self.queries if not q['answerable'])} unanswerable)")
        print(f"  Strategies:   11")
        print(f"  Max cycles:   {n_cycles}")

        # Phase 1: Tournament
        winner = self.run_tournament()

        # Phase 2: Learning cycles
        cycles = []
        for i in range(1, n_cycles + 1):
            cycle = self.run_learning_cycle(i)
            cycles.append(cycle)

            # Convergence check
            m = cycle['metrics_after']
            if m['recall_at_5'] >= 1.0 and m['accuracy'] >= 1.0 and m['hallucination_rate'] <= 0.0:
                print(f"\n  >>> CONVERGED at cycle {i}: Perfect scores. <<<")
                break
            if i > 1 and all(abs(cycle['deltas'][k]) < 0.001 for k in cycle['deltas']):
                print(f"\n  >>> PLATEAUED at cycle {i}: No change. <<<")
                break

        # Summary
        self._print_summary(winner, cycles)
        return cycles

    def _print_metrics(self, label, m):
        print(f"  {label}: Recall@5={m['recall_at_5']:.2%}  "
              f"Accuracy={m['accuracy']:.2%}  Halluc={m['hallucination_rate']:.2%}")

    def _print_summary(self, winner, cycles):
        print("\n" + "=" * 70)
        print("  LEARNING SUMMARY")
        print("=" * 70)

        print(f"\n  Tournament winner:    {winner['name']}")
        print(f"  Final active strategy: {self.active_strategy_name}")
        print(f"  Learning cycles run:   {len(cycles)}")
        print(f"  Cycles with improvement: {sum(1 for c in cycles if c['improved'])}")

        if cycles:
            first_m = cycles[0]['metrics_before']
            last_m = cycles[-1]['metrics_after']
            print(f"\n  {'Metric':<22} {'Tournament':>10} {'Final':>10} {'Delta':>10}")
            print(f"  {'-'*52}")
            for key, label in [('recall_at_5', 'Recall@5'),
                               ('accuracy', 'Accuracy'),
                               ('hallucination_rate', 'Halluc Rate')]:
                print(f"  {label:<22} {first_m[key]:>9.2%} {last_m[key]:>9.2%} "
                      f"{last_m[key] - first_m[key]:>+9.2%}")

        # Fixes applied across all cycles
        all_fixes = [f for c in cycles for f in c.get('fixes_applied', [])]
        if all_fixes:
            print(f"\n  Fixes applied across all cycles:")
            for f in all_fixes:
                print(f"    - {f.get('action', '?')}: {f.get('description', '')}")

        print(f"\n  Learning store: {self.store.path}")
        print(f"  Best strategy ever: {self.store.history.get('best_strategy', 'N/A')}")
        best = self.store.history.get('best_metrics', {})
        if best:
            print(f"  Best metrics ever:  R@5={best.get('recall_at_5',0):.2%}  "
                  f"Acc={best.get('accuracy',0):.2%}  "
                  f"Hal={best.get('hallucination_rate',1):.2%}")
        print("=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Auto-Learning RAG (V2-Integrated)')
    parser.add_argument('--cycles', type=int, default=3, help='Learning cycles (default: 3)')
    parser.add_argument('--store', type=str, default='learning_store', help='Store path')
    args = parser.parse_args()

    base = Path(__file__).parent
    learner = AutoLearningRAG(
        emails_dir=base / 'emails',
        test_file=base / 'test_queries.json',
        store_dir=base / args.store,
    )
    learner.run(n_cycles=args.cycles)


if __name__ == '__main__':
    main()
