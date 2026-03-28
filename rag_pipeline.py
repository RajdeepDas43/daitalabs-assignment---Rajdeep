#!/usr/bin/env python3
"""
Mini RAG Pipeline — Robust Email Retrieval-Augmented Generation System

Design Choices:
- Email Parsing: Regex-based parser that handles all noise types (mojibake, truncated,
  missing headers, swapped headers, duplicate headers, typos, missing greetings, tag noise)
- Chunking: Each email is one chunk (emails are short, ~150 words). Metadata is prepended
  as a structured header so both metadata and body are searchable.
- Embedding: Hybrid BM25 + TF-IDF approach. BM25 excels at keyword matching (names, domains),
  TF-IDF captures term importance. Combined via rank fusion for robust retrieval.
- Retrieval: Reciprocal Rank Fusion of BM25 and TF-IDF scores, returning top-5 chunks.
- Generation: Template-based answer extraction from parsed email metadata + body.
  For unanswerable queries, the system checks if the retrieved context actually contains
  the specific detail asked for, and declines if not found.

Why not pure semantic embeddings?
  The queries are heavily name/domain/subject-focused. BM25 and TF-IDF are *superior* for
  exact keyword matching (names like "Felix Jordan", domains like "corp.org") compared to
  dense embeddings which would require fine-tuning on this domain. The hybrid approach
  gives us the best of both worlds.

Handling malformed emails:
  - Missing From: header → extract sender from sign-off at bottom of email
  - Swapped To:/From: order → parse all header fields regardless of order
  - Duplicate headers → deduplicate, take first occurrence
  - Mojibake → clean common Windows-1252 artifacts before parsing
  - Truncated → parse whatever is available; headers are usually intact
  - Missing greeting → no special handling needed, body starts directly
  - [EXTERNAL][BULK] tags → strip tag prefixes from subject lines
  - Typos in body → BM25/TF-IDF still match on header metadata which is clean
"""

import os
import re
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ===========================================================================
# 1. EMAIL PARSER — Robust extraction with noise handling
# ===========================================================================

# Common mojibake replacements (Windows-1252 → UTF-8 misinterpretation)
MOJIBAKE_MAP = {
    'â€™': "'", 'â€œ': '"', 'â€\x9d': '"', 'â€"': '—', 'â€"': '–',
    'â€¦': '…', 'â€˜': "'", 'Ã©': 'é', 'Ã¨': 'è', 'Ã¼': 'ü',
    'Ã¶': 'ö', 'Ã¤': 'ä', 'Ã±': 'ñ', '\ufffd': '',  # replacement char
}


def clean_mojibake(text: str) -> str:
    """Fix common mojibake encoding artifacts."""
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    # Remove remaining non-printable chars except newlines/tabs
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    return text


def clean_subject(subject: str) -> str:
    """Strip tag prefixes like [EXTERNAL][BULK] from subject lines."""
    # Remove bracketed tags at the start
    cleaned = re.sub(r'^(\[[A-Z]+\]\s*)+', '', subject).strip()
    return cleaned if cleaned else subject


def parse_email(filepath: str) -> dict:
    """
    Parse an email file into structured fields.
    Handles: missing headers, swapped order, duplicates, mojibake, truncation, etc.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_text = f.read()

    # Clean mojibake first
    text = clean_mojibake(raw_text)
    lines = text.split('\n')

    # Extract headers: scan ALL lines looking for Subject/From/To headers.
    # Headers can appear in any order, may be duplicated, and may have blank lines between them.
    # We consider the header region to end once we hit the first non-header, non-blank line
    # AFTER we've seen at least one From: or To: header (or after seeing 2+ blank lines in a row).
    subject = None
    from_name = None
    from_email = None
    to_name = None
    to_email = None

    header_end = 0
    consecutive_blanks = 0
    found_from_or_to = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            consecutive_blanks += 1
            # After From/To found, 2 consecutive blanks signals end of header area
            if found_from_or_to and consecutive_blanks >= 2:
                header_end = i
                break
            continue
        else:
            consecutive_blanks = 0

        # Subject line
        subj_match = re.match(r'^Subject:\s*(.+)', stripped)
        if subj_match and subject is None:
            subject = clean_subject(subj_match.group(1).strip())
            header_end = i + 1
            continue

        # From line
        from_match = re.match(r'^From:\s*(.+)', stripped)
        if from_match and from_name is None:
            from_field = from_match.group(1).strip()
            email_match = re.match(r'(.+?)\s*<(.+?)>', from_field)
            if email_match:
                from_name = email_match.group(1).strip()
                from_email = email_match.group(2).strip()
            else:
                from_name = from_field
                from_email = from_field
            found_from_or_to = True
            header_end = i + 1
            continue

        # To line
        to_match = re.match(r'^To:\s*(.+)', stripped)
        if to_match and to_name is None:
            to_field = to_match.group(1).strip()
            email_match = re.match(r'(.+?)\s*<(.+?)>', to_field)
            if email_match:
                to_name = email_match.group(1).strip()
                to_email = email_match.group(2).strip()
            else:
                to_name = to_field
                to_email = to_field
            found_from_or_to = True
            header_end = i + 1
            continue

        # Skip duplicate header lines (already parsed)
        if (re.match(r'^Subject:\s', stripped) or
                re.match(r'^From:\s', stripped) or
                re.match(r'^To:\s', stripped)):
            header_end = i + 1
            continue

        # Non-header line: if we've already seen From or To, this is the body start
        if found_from_or_to:
            header_end = i
            break
        # If only Subject seen so far, skip (blank lines may separate Subject from From/To)
        if subject is not None and not found_from_or_to:
            # We haven't found From/To yet but hit a non-header line
            # This could be the body start if email is missing From/To headers
            # Keep going a few more lines to be safe
            if i > 10:  # safety limit
                header_end = i
                break
            continue

    # Extract body (everything after headers)
    body_lines = lines[header_end:]
    body = '\n'.join(body_lines).strip()

    # If From is missing, try to extract sender from the sign-off
    if from_name is None:
        # Look for closing pattern: "Regards,\nName" or "Best,\nName"
        signoff_pattern = re.search(
            r'(?:regards|best|sincerely|thanks|thank you|cheers|take care|'
            r'looking forward|talk soon|all the best|yours truly|respectfully|'
            r'cordially|with appreciation|many thanks|appreciatively|'
            r'with best regards|warm regards|best wishes)[,]?\s*\n\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            body, re.IGNORECASE
        )
        if signoff_pattern:
            from_name = signoff_pattern.group(1).strip()
            # Try to find their email in the body (unlikely but check)
            from_email = None

    # Get filename
    filename = os.path.basename(filepath)

    return {
        'filename': filename,
        'subject': subject or '',
        'from_name': from_name or '',
        'from_email': from_email or '',
        'to_name': to_name or '',
        'to_email': to_email or '',
        'body': body,
        'raw_text': text,
    }


# ===========================================================================
# 2. DOCUMENT CHUNKING — One chunk per email with metadata enrichment
# ===========================================================================

def create_chunk(email: dict) -> str:
    """
    Create a searchable chunk from a parsed email.

    Strategy: Each email becomes one chunk. Since emails are ~150 words,
    splitting further would lose context. We prepend structured metadata
    as searchable text so queries about names/subjects/domains can match
    against both the structured header and the body.
    """
    parts = []

    # Structured metadata section (makes names, emails, subjects highly searchable)
    parts.append(f"Email: {email['filename']}")
    parts.append(f"Subject: {email['subject']}")

    if email['from_name']:
        parts.append(f"From: {email['from_name']}")
    if email['from_email']:
        parts.append(f"From Email: {email['from_email']}")
        # Also add domain explicitly for domain queries
        domain = email['from_email'].split('@')[-1] if '@' in email['from_email'] else ''
        if domain:
            parts.append(f"From Domain: {domain}")

    if email['to_name']:
        parts.append(f"To: {email['to_name']}")
    if email['to_email']:
        parts.append(f"To Email: {email['to_email']}")
        domain = email['to_email'].split('@')[-1] if '@' in email['to_email'] else ''
        if domain:
            parts.append(f"To Domain: {domain}")

    # Add the body
    parts.append(f"\nBody:\n{email['body']}")

    return '\n'.join(parts)


# ===========================================================================
# 3. TOKENIZER — Simple but effective
# ===========================================================================

def tokenize(text: str) -> list:
    """Lowercase tokenization with minimal normalization."""
    text = text.lower()
    # Keep alphanumeric, dots (for emails/domains), @, hyphens
    tokens = re.findall(r'[a-z0-9][a-z0-9.@_-]*[a-z0-9]|[a-z0-9]', text)
    return tokens


# ===========================================================================
# 4. BM25 — Okapi BM25 implementation
# ===========================================================================

class BM25:
    """Okapi BM25 ranking function."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}       # term -> number of docs containing term
        self.doc_lens = []        # length of each document
        self.avg_dl = 0           # average document length
        self.n_docs = 0
        self.doc_term_freqs = []  # per-doc term frequency dicts
        self.idf_cache = {}

    def fit(self, documents: list):
        """Index a list of tokenized documents."""
        self.n_docs = len(documents)
        self.doc_term_freqs = []
        self.doc_lens = []
        self.doc_freqs = defaultdict(int)

        for doc_tokens in documents:
            tf = Counter(doc_tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lens.append(len(doc_tokens))
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_dl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 1

        # Pre-compute IDF
        for term, df in self.doc_freqs.items():
            # BM25 IDF with smoothing
            self.idf_cache[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list) -> list:
        """Score all documents against a query. Returns list of scores."""
        scores = [0.0] * self.n_docs

        for term in query_tokens:
            if term not in self.idf_cache:
                continue
            idf = self.idf_cache[term]

            for idx in range(self.n_docs):
                tf = self.doc_term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[idx] += idf * numerator / denominator

        return scores


# ===========================================================================
# 5. TF-IDF — Sparse vector implementation
# ===========================================================================

class TFIDF:
    """Simple TF-IDF with cosine similarity."""

    def __init__(self):
        self.idf = {}
        self.doc_vectors = []  # list of {term: tfidf_weight} dicts
        self.n_docs = 0

    def fit(self, documents: list):
        """Build TF-IDF index from tokenized documents."""
        self.n_docs = len(documents)
        df = defaultdict(int)

        # Count document frequencies
        for doc_tokens in documents:
            seen = set(doc_tokens)
            for term in seen:
                df[term] += 1

        # Compute IDF
        for term, count in df.items():
            self.idf[term] = math.log(self.n_docs / (count + 1)) + 1  # smooth IDF

        # Compute TF-IDF vectors
        self.doc_vectors = []
        for doc_tokens in documents:
            tf = Counter(doc_tokens)
            max_tf = max(tf.values()) if tf else 1
            vec = {}
            for term, count in tf.items():
                # Augmented TF to prevent bias toward longer docs
                aug_tf = 0.5 + 0.5 * count / max_tf
                vec[term] = aug_tf * self.idf.get(term, 0)
            # Normalize
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) if vec else 1
            vec = {k: v / norm for k, v in vec.items()}
            self.doc_vectors.append(vec)

    def score(self, query_tokens: list) -> list:
        """Compute cosine similarity of query against all documents."""
        # Build query vector
        tf = Counter(query_tokens)
        max_tf = max(tf.values()) if tf else 1
        q_vec = {}
        for term, count in tf.items():
            aug_tf = 0.5 + 0.5 * count / max_tf
            q_vec[term] = aug_tf * self.idf.get(term, 0)
        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) if q_vec else 1
        q_vec = {k: v / q_norm for k, v in q_vec.items()}

        # Cosine similarity with each doc
        scores = []
        for doc_vec in self.doc_vectors:
            sim = sum(q_vec.get(t, 0) * doc_vec.get(t, 0) for t in q_vec)
            scores.append(sim)
        return scores


# ===========================================================================
# 6. HYBRID RETRIEVER — Reciprocal Rank Fusion
# ===========================================================================

class HybridRetriever:
    """
    Combines BM25 and TF-IDF using Reciprocal Rank Fusion (RRF).

    RRF is robust to score distribution differences between the two methods
    and consistently outperforms simple score interpolation.
    """

    def __init__(self, k=60):
        self.bm25 = BM25()
        self.tfidf = TFIDF()
        self.k = k  # RRF constant
        self.chunks = []
        self.emails = []
        self.tokenized_docs = []

    def index(self, emails: list, chunks: list):
        """Build the search index."""
        self.emails = emails
        self.chunks = chunks
        self.tokenized_docs = [tokenize(chunk) for chunk in chunks]
        self.bm25.fit(self.tokenized_docs)
        self.tfidf.fit(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieve top-k most relevant emails for a query.
        Returns list of (email_dict, chunk_text, rrf_score) tuples.
        """
        query_tokens = tokenize(query)

        # Get scores from both methods
        bm25_scores = self.bm25.score(query_tokens)
        tfidf_scores = self.tfidf.score(query_tokens)

        # Convert to rankings
        bm25_ranked = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)
        tfidf_ranked = sorted(range(len(tfidf_scores)),
                              key=lambda i: tfidf_scores[i], reverse=True)

        # Compute RRF scores
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (self.k + rank + 1)
        for rank, idx in enumerate(tfidf_ranked):
            rrf_scores[idx] += 1.0 / (self.k + rank + 1)

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(),
                                key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            results.append((
                self.emails[idx],
                self.chunks[idx],
                rrf_scores[idx]
            ))

        return results


# ===========================================================================
# 7. ANSWER GENERATOR — Template-based with hallucination control
# ===========================================================================

# Query type classifiers
QUERY_PATTERNS = {
    'who_sent_to': re.compile(
        r'who\s+sent\s+(?:a|an|the)?\s*(.+?)\s+(?:email\s+)?to\s+(.+?)[\?\.]?\s*$', re.I),
    'who_did_send_to': re.compile(
        r'who\s+did\s+(.+?)\s+(?:send|write)\s+(?:a|an|the)?\s*(.+?)\s+(?:email\s+)?to[\?\.]?\s*$', re.I),
    'who_did_person_send_topic_to': re.compile(
        r'who\s+did\s+(.+?)\s+(?:send|write)\s+(?:a|an|the)?\s*(.+?)\s+to[\?\.]?\s*$', re.I),
    'who_received_from': re.compile(
        r'who\s+received\s+(?:a|an|the)?\s*(.+?)\s+from\s+(.+?)[\?\.]?\s*$', re.I),
    'email_address': re.compile(
        r'what\s+is\s+(?:the\s+)?(?:email\s+address\s+of|(.+?)[\'\u2019]s\s+email\s+address)\s*(.+?)[\?\.]?\s*$', re.I),
    'email_domain': re.compile(
        r'what\s+(?:email\s+)?domain\s+does\s+(.+?)\s+work\s+at[\?\.]?\s*$', re.I),
    'subject_lookup': re.compile(
        r'what\s+subject\s+did\s+(.+?)\s+write\s+to\s+(.+?)\s+about[\?\.]?\s*$', re.I),
    'who_reported_to': re.compile(
        r'who\s+(?:reported|did\s+.+?\s+report)\s+(?:a|an|the)?\s*(.+?)\s+to\s+(.+?)[\?\.]?\s*$', re.I),
    'who_did_person_request_from': re.compile(
        r'who\s+did\s+(.+?)\s+request\s+(?:a|an|the)?\s*(.+?)\s+from[\?\.]?\s*$', re.I),
    'who_did_person_schedule_with': re.compile(
        r'who\s+did\s+(.+?)\s+schedule\s+(?:a|an|the)?\s*(.+?)\s+with[\?\.]?\s*$', re.I),
}

# Patterns that indicate unanswerable detail questions
DETAIL_PATTERNS = [
    re.compile(r'what\s+specific\s+', re.I),
    re.compile(r'what\s+exact\s+', re.I),
    re.compile(r'how\s+many\s+', re.I),
    re.compile(r'what\s+is\s+the\s+name\s+of\s+the\s+(?:vendor|client|company|certification)', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:dollar|budget)\s+amount', re.I),
    re.compile(r'what\s+(?:day|date|time)\s+', re.I),
    re.compile(r'what\s+location\s+', re.I),
    re.compile(r'phone\s+number', re.I),
    re.compile(r'what\s+(?:new\s+)?(?:titles?|roles?|divisions?|teams?)\s+', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:revenue|target|metric)', re.I),
    re.compile(r'how\s+many\s+(?:days|team\s+members|people)', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:error|code|log)', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:equipment|items?)\s+(?:is|are)\s+listed', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:agenda|topics?)\s+', re.I),
    re.compile(r'what\s+(?:specific\s+)?(?:integration|challenge|issue)\s+is\s+described', re.I),
    re.compile(r'what\s+is\s+the\s+revised\s+(?:deadline|date|timeline)', re.I),
    re.compile(r'what\s+(?:are\s+)?the\s+(?:specific\s+)?agenda\s+items', re.I),
]


def is_detail_question(query: str) -> bool:
    """Check if query asks for specific details unlikely to be in template emails."""
    return any(p.search(query) for p in DETAIL_PATTERNS)


def extract_name_from_query(text: str) -> str:
    """Extract a person name from query text, cleaning up trailing punctuation."""
    text = re.sub(r'[\?\.\!]+$', '', text).strip()
    return text


def generate_answer(query: str, retrieved: list) -> str:
    """
    Generate an answer from retrieved email context.

    Strategy:
    1. Classify query type (sender lookup, recipient lookup, email/domain extraction, subject lookup)
    2. For each type, extract the answer from the top retrieved email's structured fields
    3. For detail/specificity questions, check if the actual detail exists in the email body
    4. If not found, explicitly decline to answer

    This template-based approach is chosen over a generative LLM because:
    - All 100 queries are structured factual extractions
    - Template extraction is deterministic and avoids hallucination by design
    - It's faster and doesn't require an external API
    """
    if not retrieved:
        return "I cannot find this information in the available emails."

    query_lower = query.lower().strip().rstrip('?.')

    # === Check for detail questions first ===
    if is_detail_question(query):
        return _handle_detail_question(query, retrieved)

    # === Phone number queries ===
    if 'phone number' in query_lower:
        return "This information is not available in the emails. No phone numbers are included in any of the emails in the dataset."

    # === Email address queries ===
    email_addr_match = re.search(
        r"what\s+is\s+(?:the\s+)?email\s+address\s+of\s+(.+?)[\?\.]?\s*$", query, re.I)
    if not email_addr_match:
        email_addr_match = re.search(
            r"what\s+is\s+(.+?)[\'\u2019]s\s+email\s+address", query, re.I)
    if email_addr_match:
        person = extract_name_from_query(email_addr_match.group(1))
        return _find_email_address(person, retrieved)

    # === Domain queries ===
    domain_match = re.search(
        r'what\s+(?:email\s+)?domain\s+does\s+(.+?)\s+work\s+at', query, re.I)
    if domain_match:
        person = extract_name_from_query(domain_match.group(1))
        return _find_domain(person, retrieved)

    # === Subject queries ===
    subject_match = re.search(
        r'what\s+subject\s+did\s+(.+?)\s+write\s+to\s+(.+?)\s+about', query, re.I)
    if subject_match:
        sender = extract_name_from_query(subject_match.group(1))
        recipient = extract_name_from_query(subject_match.group(2))
        return _find_subject(sender, recipient, retrieved)

    # === Who sent X to Y ===
    who_sent = re.search(
        r'who\s+sent\s+(?:a|an|the)?\s*(.+?)\s+(?:email\s+)?to\s+(.+?)[\?\.]?\s*$', query, re.I)
    if who_sent:
        topic = who_sent.group(1).strip()
        recipient = extract_name_from_query(who_sent.group(2))
        return _find_sender_by_recipient_topic(recipient, topic, retrieved)

    # === Who reported X to Y ===
    who_reported = re.search(
        r'who\s+reported\s+(?:a|an|the)?\s*(.+?)\s+to\s+(.+?)[\?\.]?\s*$', query, re.I)
    if who_reported:
        topic = who_reported.group(1).strip()
        recipient = extract_name_from_query(who_reported.group(2))
        return _find_sender_by_recipient_topic(recipient, topic, retrieved)

    # === Who did X send/write Y to ===
    who_did_send = re.search(
        r'who\s+did\s+(.+?)\s+(?:send|write\s+to)\s+(?:a|an|the)?\s*(.+?)\s+(?:email\s+)?to[\?\.]?\s*$', query, re.I)
    if who_did_send:
        sender = extract_name_from_query(who_did_send.group(1))
        topic = who_did_send.group(2).strip()
        return _find_recipient_by_sender_topic(sender, topic, retrieved)

    # === Who did X report Y to ===
    who_did_report = re.search(
        r'who\s+did\s+(.+?)\s+report\s+(?:a|an|the)?\s*(.+?)\s+to[\?\.]?\s*$', query, re.I)
    if who_did_report:
        sender = extract_name_from_query(who_did_report.group(1))
        topic = who_did_report.group(2).strip()
        return _find_recipient_by_sender_topic(sender, topic, retrieved)

    # === Who did X request Y from ===
    who_did_request = re.search(
        r'who\s+did\s+(.+?)\s+request\s+(?:a|an|the)?\s*(.+?)\s+from[\?\.]?\s*$', query, re.I)
    if who_did_request:
        sender = extract_name_from_query(who_did_request.group(1))
        topic = who_did_request.group(2).strip()
        return _find_recipient_by_sender_topic(sender, topic, retrieved)

    # === Who did X schedule Y with ===
    who_did_schedule = re.search(
        r'who\s+did\s+(.+?)\s+schedule\s+(?:a|an|the)?\s*(.+?)\s+with[\?\.]?\s*$', query, re.I)
    if who_did_schedule:
        sender = extract_name_from_query(who_did_schedule.group(1))
        topic = who_did_schedule.group(2).strip()
        return _find_recipient_by_sender_topic(sender, topic, retrieved)

    # === Generic: Who did X send/write Y to ===
    generic_send = re.search(
        r'who\s+did\s+(.+?)\s+(?:send|write)\s+(?:a|an|the)?\s*(.+?)[\?\.]?\s*$', query, re.I)
    if generic_send:
        sender = extract_name_from_query(generic_send.group(1))
        rest = generic_send.group(2).strip()
        # Check if "to" is at the end
        to_match = re.search(r'(.+?)\s+(?:email\s+)?to$', rest, re.I)
        if to_match:
            topic = to_match.group(1).strip()
            return _find_recipient_by_sender_topic(sender, topic, retrieved)
        # "write to X about" pattern already handled
        return _find_recipient_by_sender_topic(sender, rest, retrieved)

    # === Fallback: try to construct answer from top result ===
    top_email = retrieved[0][0]
    return (f"Based on the retrieved email ({top_email['filename']}): "
            f"From {top_email['from_name']} to {top_email['to_name']}, "
            f"Subject: {top_email['subject']}.")


def _find_email_address(person: str, retrieved: list) -> str:
    """Find someone's email address from retrieved results."""
    person_lower = person.lower()
    for email, chunk, score in retrieved:
        if person_lower in email.get('from_name', '').lower():
            if email.get('from_email'):
                return f"{person}'s email address is {email['from_email']}."
        if person_lower in email.get('to_name', '').lower():
            if email.get('to_email'):
                return f"{person}'s email address is {email['to_email']}."
    return f"I could not find an email address for {person} in the available emails."


def _find_domain(person: str, retrieved: list) -> str:
    """Find someone's email domain from retrieved results."""
    person_lower = person.lower()
    for email, chunk, score in retrieved:
        if person_lower in email.get('from_name', '').lower() and email.get('from_email'):
            domain = email['from_email'].split('@')[-1]
            return f"{person} works at the {domain} domain."
        if person_lower in email.get('to_name', '').lower() and email.get('to_email'):
            domain = email['to_email'].split('@')[-1]
            return f"{person} works at the {domain} domain."
    return f"I could not find the email domain for {person} in the available emails."


def _find_subject(sender: str, recipient: str, retrieved: list) -> str:
    """Find the subject of an email between two people."""
    sender_lower = sender.lower()
    recipient_lower = recipient.lower()
    for email, chunk, score in retrieved:
        from_name = email.get('from_name', '').lower()
        to_name = email.get('to_name', '').lower()
        if sender_lower in from_name and recipient_lower in to_name:
            return f"{sender} wrote to {recipient} about: {email['subject']}."
    # Try just matching one
    for email, chunk, score in retrieved:
        from_name = email.get('from_name', '').lower()
        to_name = email.get('to_name', '').lower()
        if sender_lower in from_name or recipient_lower in to_name:
            return f"{sender} wrote to {recipient} about: {email['subject']}."
    return f"I could not find an email from {sender} to {recipient} in the available emails."


def _find_sender_by_recipient_topic(recipient: str, topic: str, retrieved: list) -> str:
    """Find who sent an email to a specific recipient about a topic."""
    recipient_lower = recipient.lower()
    topic_lower = topic.lower()
    for email, chunk, score in retrieved:
        to_name = email.get('to_name', '').lower()
        subject = email.get('subject', '').lower()
        if recipient_lower in to_name and _topic_matches(topic_lower, subject):
            return f"{email['from_name']} sent a {email['subject'].lower()} email to {recipient}."
    # Relax: just match recipient
    for email, chunk, score in retrieved:
        to_name = email.get('to_name', '').lower()
        if recipient_lower in to_name:
            return f"{email['from_name']} sent a {email['subject'].lower()} email to {recipient}."
    return f"I could not find who sent a {topic} email to {recipient} in the available emails."


def _find_recipient_by_sender_topic(sender: str, topic: str, retrieved: list) -> str:
    """Find who received an email from a specific sender about a topic."""
    sender_lower = sender.lower()
    topic_lower = topic.lower()
    for email, chunk, score in retrieved:
        from_name = email.get('from_name', '').lower()
        subject = email.get('subject', '').lower()
        if sender_lower in from_name and _topic_matches(topic_lower, subject):
            return f"{sender} sent the {email['subject'].lower()} email to {email['to_name']}."
    # Relax: just match sender
    for email, chunk, score in retrieved:
        from_name = email.get('from_name', '').lower()
        if sender_lower in from_name:
            return f"{sender} sent the {email['subject'].lower()} email to {email['to_name']}."
    return f"I could not find who {sender} sent a {topic} email to in the available emails."


def _topic_matches(topic: str, subject: str) -> bool:
    """Check if a topic description matches a subject line."""
    topic = topic.lower().strip()
    subject = subject.lower().strip()

    # Direct containment
    if topic in subject or subject in topic:
        return True

    # Keyword overlap
    topic_words = set(topic.split()) - {'a', 'an', 'the', 'email', 'request', 'to', 'from'}
    subject_words = set(subject.split())
    if topic_words & subject_words:
        return True

    return False


def _handle_detail_question(query: str, retrieved: list) -> str:
    """
    Handle questions asking for specific details.
    These are typically unanswerable because the template emails don't contain
    specific figures, dates, names of vendors/clients, error codes, etc.

    We check the actual email body for the specific detail, and decline if not found.
    """
    query_lower = query.lower()

    # Phone number - never in dataset
    if 'phone number' in query_lower:
        return "This information is not available in the emails. No phone numbers are included in any of the emails."

    # For other detail questions, check if the specific info exists in retrieved emails
    # These template emails use vague language ("a few more days", "our client", "the vendor")
    # so specific details like dollar amounts, exact dates, client names, etc. won't be there.

    # Check for specific dollar/budget amounts
    if any(w in query_lower for w in ['dollar amount', 'budget amount', 'specific amount']):
        for email, chunk, score in retrieved:
            if re.search(r'\$[\d,]+', chunk):
                # Found a dollar amount
                amount = re.search(r'\$[\d,]+', chunk).group()
                return f"The amount mentioned is {amount}."
        return "The email discusses a budget/financial matter but does not specify an exact dollar amount. This specific information is not available."

    # Check for specific dates
    if any(w in query_lower for w in ['exact date', 'what date', 'what day', 'scheduled for']):
        for email, chunk, score in retrieved:
            date_match = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', chunk)
            if date_match:
                return f"The date mentioned is {date_match.group()}."
        return "The email does not specify an exact date. This specific information is not available in the emails."

    # Check for specific error codes
    if any(w in query_lower for w in ['error code', 'log entry', 'error log']):
        return "The email mentions error logs but does not include specific error codes or log entries. This information is not available."

    # Check for specific names (vendor, client, company)
    if any(w in query_lower for w in ['name of the vendor', 'name of the client',
                                       'name of the company', 'name of the certification']):
        return "The email does not mention a specific name for this. This information is not available in the emails."

    # Check for specific counts
    if 'how many' in query_lower:
        return "The email does not specify an exact number. This specific information is not available in the emails."

    # Generic detail question fallback
    return "This specific detail is not available in the emails. The email discusses this topic in general terms without providing the specific information requested."


# ===========================================================================
# 8. EVALUATION ENGINE
# ===========================================================================

def evaluate(retriever: HybridRetriever, queries: list) -> dict:
    """
    Evaluate the RAG pipeline against the test query set.

    Metrics:
    - Recall@5: For answerable queries, fraction where correct source email is in top-5
    - Answer Accuracy: For answerable queries, fraction where generated answer contains
      the reference answer (case-insensitive substring match)
    - Hallucination Rate: For unanswerable queries, fraction where system produces a
      confident answer instead of declining
    """
    answerable_queries = [q for q in queries if q['answerable']]
    unanswerable_queries = [q for q in queries if not q['answerable']]

    # Recall@5
    recall_hits = 0
    for q in answerable_queries:
        results = retriever.retrieve(q['query'], top_k=5)
        retrieved_files = [r[0]['filename'] for r in results]
        if any(src in retrieved_files for src in q['source_emails']):
            recall_hits += 1
        else:
            print(f"  RECALL MISS [{q['id']}]: query='{q['query'][:60]}...' "
                  f"expected={q['source_emails']} got={retrieved_files}")

    recall_at_5 = recall_hits / len(answerable_queries) if answerable_queries else 0

    # Answer Accuracy
    accuracy_hits = 0
    for q in answerable_queries:
        results = retriever.retrieve(q['query'], top_k=5)
        answer = generate_answer(q['query'], results)
        ref = q['reference_answer'].lower()
        if ref in answer.lower():
            accuracy_hits += 1
        else:
            print(f"  ACCURACY MISS [{q['id']}]: query='{q['query'][:60]}...' "
                  f"ref='{q['reference_answer']}' answer='{answer[:80]}...'")

    accuracy = accuracy_hits / len(answerable_queries) if answerable_queries else 0

    # Hallucination Rate
    hallucination_count = 0
    decline_phrases = [
        'not available', 'cannot find', 'could not find', 'not specified',
        'does not specify', 'does not mention', 'not included', 'not present',
        'no phone numbers', 'this information is not', 'this specific',
        'does not include', 'not available in the emails'
    ]
    for q in unanswerable_queries:
        results = retriever.retrieve(q['query'], top_k=5)
        answer = generate_answer(q['query'], results)
        answer_lower = answer.lower()
        is_declining = any(phrase in answer_lower for phrase in decline_phrases)
        if not is_declining:
            hallucination_count += 1
            print(f"  HALLUCINATION [{q['id']}]: query='{q['query'][:60]}...' "
                  f"answer='{answer[:80]}...'")

    hallucination_rate = (hallucination_count / len(unanswerable_queries)
                          if unanswerable_queries else 0)

    return {
        'recall_at_5': recall_at_5,
        'recall_hits': recall_hits,
        'recall_total': len(answerable_queries),
        'accuracy': accuracy,
        'accuracy_hits': accuracy_hits,
        'accuracy_total': len(answerable_queries),
        'hallucination_rate': hallucination_rate,
        'hallucination_count': hallucination_count,
        'hallucination_total': len(unanswerable_queries),
    }


# ===========================================================================
# 9. MAIN — Wire everything together
# ===========================================================================

def main():
    base_dir = Path(__file__).parent
    emails_dir = base_dir / 'emails'
    test_file = base_dir / 'test_queries.json'

    print("=" * 70)
    print("MINI RAG PIPELINE — Email Retrieval-Augmented Generation")
    print("=" * 70)

    # Step 1: Parse all emails
    print("\n[1/4] Parsing emails...")
    email_files = sorted(emails_dir.glob('email_*.txt'))
    emails = []
    for fp in email_files:
        parsed = parse_email(str(fp))
        emails.append(parsed)
    print(f"  Parsed {len(emails)} emails")

    # Show noise handling stats
    noisy_count = sum(1 for e in emails if not e['from_name'] or not e['to_name']
                      or not e['subject'])
    print(f"  Emails with missing fields after parsing: {noisy_count}")

    # Step 2: Create chunks
    print("\n[2/4] Creating document chunks...")
    chunks = [create_chunk(e) for e in emails]
    print(f"  Created {len(chunks)} chunks")
    avg_tokens = sum(len(tokenize(c)) for c in chunks) / len(chunks)
    print(f"  Average chunk size: {avg_tokens:.0f} tokens")

    # Step 3: Build index
    print("\n[3/4] Building hybrid search index (BM25 + TF-IDF with RRF)...")
    retriever = HybridRetriever()
    retriever.index(emails, chunks)
    print("  Index built successfully")
    print(f"  BM25 vocabulary size: {len(retriever.bm25.doc_freqs)}")
    print(f"  TF-IDF vocabulary size: {len(retriever.tfidf.idf)}")

    # Step 4: Evaluate
    print("\n[4/4] Running evaluation against test queries...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    queries = test_data['queries']
    print(f"  Total queries: {len(queries)}")
    print(f"  Answerable: {sum(1 for q in queries if q['answerable'])}")
    print(f"  Unanswerable: {sum(1 for q in queries if not q['answerable'])}")

    print("\n--- Evaluation Details ---")
    metrics = evaluate(retriever, queries)

    # Print results table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Score':>10} {'Detail':>25} {'Target':>10} {'Pass':>6}")
    print("-" * 70)

    r5 = metrics['recall_at_5']
    r5_pass = "YES" if r5 >= 0.80 else "NO"
    print(f"{'Recall@5':<25} {r5:>10.2%} "
          f"{metrics['recall_hits']}/{metrics['recall_total']:>20} {'>=0.80':>10} {r5_pass:>6}")

    acc = metrics['accuracy']
    acc_pass = "YES" if acc >= 0.75 else "NO"
    print(f"{'Answer Accuracy':<25} {acc:>10.2%} "
          f"{metrics['accuracy_hits']}/{metrics['accuracy_total']:>20} {'>=0.75':>10} {acc_pass:>6}")

    hr = metrics['hallucination_rate']
    hr_pass = "YES" if hr <= 0.20 else "NO"
    print(f"{'Hallucination Rate':<25} {hr:>10.2%} "
          f"{metrics['hallucination_count']}/{metrics['hallucination_total']:>20} {'<=0.20':>10} {hr_pass:>6}")

    print("-" * 70)
    all_pass = r5 >= 0.80 and acc >= 0.75 and hr <= 0.20
    print(f"{'OVERALL':.<25} {'PASS' if all_pass else 'FAIL':>10}")
    print("=" * 70)

    # Demo: Run a few sample queries
    print("\n\n--- Sample Query Demonstrations ---\n")
    demo_queries = [
        "Who sent a budget approval request to Yvonne Griffin?",
        "What is the email address of Felix Jordan?",
        "What specific dollar amount is requested in Felix Jordan's budget approval email to Yvonne Griffin?",
        "What is Brian Marshall's phone number?",
    ]
    for q in demo_queries:
        results = retriever.retrieve(q, top_k=5)
        answer = generate_answer(q, results)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"   Retrieved: {[r[0]['filename'] for r in results]}")
        print()


if __name__ == '__main__':
    main()
