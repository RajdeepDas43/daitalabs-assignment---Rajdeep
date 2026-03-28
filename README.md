# Mini RAG System

```
Time: 90 Minutes
```

## Problem

Build a Retrieval-Augmented Generation (RAG) pipeline that can process documents, retrieve relevant information, and generate answers to questions.

## Dataset

A dataset of 100 synthetic emails is provided in the `emails/` directory. Each email contains:
- Subject line
- Sender and receiver information (from a pool of 200 unique people)
- Body content (100+ words) with diverse topics including:
  - Project updates
  - Meeting requests
  - Budget approvals
  - Technical issues
  - Client feedback
  - Team announcements
  - Deadline extensions
  - Training opportunities
  - Vendor proposals
  - Performance reviews

**Note:** Approximately 12% of emails contain real-world data quality issues such as garbled encoding artifacts, truncated content, missing header fields, non-standard field ordering, duplicate headers, appended footer noise, or blank subject lines. Your pipeline must handle these gracefully.

Use this dataset to test your RAG system's ability to:
- Chunk and process email documents, including malformed ones
- Retrieve relevant information based on queries
- Generate accurate answers using the retrieved context
- Correctly decline to answer when information is not present

## Requirements

Your RAG system should implement:

1. **Document Chunking**
   - Split documents into appropriate chunks
   - Handle different document types and sizes, including malformed emails
   - Explain your chunking strategy

2. **Embedding**
   - Generate embeddings for document chunks
   - Choose an appropriate embedding model
   - Store embeddings efficiently

3. **Retrieval**
   - Implement similarity search to find relevant chunks
   - Handle query embedding and matching
   - Return top-k most relevant results

4. **Generation**
   - Use retrieved context to generate answers
   - Design effective prompts
   - Integrate with a language model
   - When an answer cannot be found in the retrieved context, the system must say so explicitly rather than fabricating an answer

## Evaluation

A gold-labeled test set is provided in `test_queries.json`. It contains **100 queries: 75 answerable and 25 unanswerable**.

Run your system against all 100 queries and report the following three metrics:

| Metric | Definition | Target |
|---|---|---|
| **Recall@5** | Fraction of answerable queries where the correct source email appears in the top-5 retrieved chunks | ≥ 0.80 |
| **Answer Accuracy** | Fraction of answerable queries where the generated answer contains the reference answer (case-insensitive substring match) | ≥ 0.75 |
| **Hallucination Rate** | Fraction of unanswerable queries where the system produces a confident answer instead of declining | ≤ 0.20 |

Include a table in your submission showing the score for each metric and a brief description of how you measured it.

## Constraints

- Do not use end-to-end RAG frameworks (e.g., LangChain, LlamaIndex)
- Build core components yourself or use individual libraries
- Document your design choices and tradeoffs
- Explain how your pipeline handles malformed or incomplete emails

## Submission

- Create a public git repository containing your submission and share the repository link
- Do not fork this repository or create pull requests
- Include a `prompts.md` file as described below

---

## ⚠️ Prompt Dump Requirement — Read Carefully

Submissions without a valid `prompts.md` are **automatically rejected** without review.

### What to include

`prompts.md` must be a **raw, unedited dump of your full AI conversation history** — every message you sent to an AI assistant and every reply it gave, for the entire duration of building this solution.

This means:

- **Both sides of the conversation.** Every user message you typed AND the full AI response to it. Not summaries. Not paraphrases. The actual text.
- **Everything, not just the "good" parts.** Include prompts where the AI gave a wrong answer, where you had to correct it, where something broke and you pasted an error, where you tried an approach and then abandoned it. Dead ends are expected and required.
- **Unformatted and unedited.** Do not rewrite, tidy, reorder, or reorganize the conversation. Copy-paste it as-is from your AI tool (ChatGPT, Claude, Cursor, Copilot, etc.). If you used multiple tools or sessions, include all of them.
- **Code and errors included.** If you pasted code, a stack trace, a file diff, or terminal output into the chat, that must appear in the dump too.

### What NOT to do

The following will be treated as a fabricated submission and rejected:

- **Using an AI to write or generate your prompts.** Your prompts must be words you typed yourself. If you asked an AI to "write me a good prompt for X" and then used that output as your next message, that is not a genuine prompt — it is AI-generated content being laundered as human intent. This defeats the entire purpose of the requirement and is grounds for **immediate rejection**.
- Submitting only your prompts without the AI's replies
- Rewriting or "cleaning up" the conversation after the fact
- Summarizing what you asked into tidy numbered steps
- Writing a narrative document that describes your process instead of dumping the raw chat
- Omitting failed attempts or corrections

### How to export

- **ChatGPT**: use the share link or copy the full conversation from the browser
- **Claude**: copy the full conversation from the browser
- **Cursor / Copilot**: copy each exchange from the chat panel as you go
- If your tool has no export, paste the conversation manually as you work — do not reconstruct it from memory at the end

### Format

No specific format is required. A flat chronological dump is fine. You may add a single line before each session like `--- Session 1 ---` if you used multiple tools, but do not restructure or annotate the content beyond that.
