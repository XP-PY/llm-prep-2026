# RAG Systems

## 1. What problem RAG is actually solving

RAG (Retrieval-Augmented Generation) is not mainly about "making the model smarter."

Its real purpose is:

> retrieve external evidence first, then generate an answer grounded in that evidence.

RAG is especially useful for four kinds of problems:

* **knowledge gaps**
  the base model does not know the fact, or remembers it inaccurately
* **outdated knowledge**
  the relevant fact appeared after the model's training cutoff
* **domain knowledge weakness**
  the model is too general and not reliable enough on specialized documents or terminology
* **evidence-required answering**
  the user wants not only an answer, but also sources, citations, and traceability

A practical decision rule is:

* if the problem is "the model does not know the fact," think **RAG**
* if the problem is "the model does not behave the right way," think **SFT / post-training**
* if the problem is "the task needs multi-step decisions," think **agent**

RAG is not the same thing as an agent, but agents often use RAG as a core capability.

---

## 2. RAG is more than a retriever

Many people reduce RAG to:

* retrieve something
* stuff documents into the context
* generate an answer

That is only **Naive RAG**.

In practice, it is more useful to think of RAG in three layers.

### 2.1 Naive RAG

The minimal form is:

* build an index offline
* run `retrieve -> generate` online

Its advantage is simplicity.  
Its weakness is that quality is unstable and failure analysis is hard.

### 2.2 Advanced RAG

This layer adds optimization steps before and after retrieval:

* query rewriting
* metadata filtering
* hybrid retrieval
* reranking
* compression
* context expansion

The key idea is:

> keep the same basic retrieval-and-generation pattern, but make every stage more reliable.

### 2.3 Modular RAG

This layer treats RAG as a set of composable modules:

* multi-route retrieval
* routing across different data sources
* query-type routing
* structured retrieval
* agent-style iterative correction

The key idea is:

> RAG is no longer a single linear pipeline, but a system of composable retrieval and generation modules.

---

## 3. Start from the task, not from the framework

The first design question should not be "LangChain or LlamaIndex?"

It should be:

### 3.1 What kind of task is this?

Common task types include:

* **fact lookup**
  for example: "In what year was this paper published?"
* **document QA**
  for example: "How does this report define a given concept?"
* **multi-document synthesis**
  for example: "Compare the differences between two papers"
* **insufficient-evidence refusal**
  for example: "If the document does not say it, explicitly say so"
* **workflow support**
  for example: "Search, compare, then summarize a research direction"

### 3.2 What does a correct output look like?

At minimum, decide:

* whether citations are required
* whether "insufficient evidence" is allowed
* whether the output should be a short answer or a structured response
* whether the task is single-document grounding or multi-document comparison

### 3.3 What is the main risk?

Different tasks fail in different ways:

* FAQ systems often fail by not retrieving the right evidence
* legal and medical systems are highly vulnerable to evidence misuse
* research assistants often fail by mis-comparing sources, mis-citing papers, or overgeneralizing

RAG design is fundamentally about:

> task objective -> data objects -> retrieval strategy -> generation constraints -> evaluation loop

---

## 4. The offline layer of a full RAG system

If the offline layer is weak, the online layer usually cannot recover.

### 4.1 Corpus design

Do not think of a corpus as "just a pile of text."

A good corpus must answer:

* what a document object looks like
* whether the source is trustworthy
* how metadata is recorded
* how frequently the corpus is updated

A minimal document object should usually contain:

* `doc_id`
* `title`
* `source_type`
* `source_url` or another source identifier
* `published_at`
* `content`
* `metadata`
* `trust_level`

For a research assistant, it is useful to distinguish at least:

* **primary sources**
  original papers, official documentation, original reports
* **secondary sources**
  surveys, project pages, technical blogs
* **tertiary sources**
  notes, forum summaries, secondary writeups

This matters because source quality affects retrieval, evidence selection, and answer confidence.

### 4.2 Why chunking matters so much

The retrieval unit in RAG is not usually "the whole document."
It is the **chunk**.

Chunk design directly affects:

* retrieval recall
* retrieval precision
* context noise seen by the LLM
* final answer quality

Chunks are not better just because they are larger. Large chunks create three common problems:

* **embedding compression loss**
  a long block is compressed into a single vector, so multiple themes get diluted
* **lost in the middle**
  important details are easier to bury inside long contexts
* **topic dilution**
  a chunk that mixes unrelated topics becomes harder to retrieve precisely

### 4.3 Common chunking strategies

#### Fixed-size chunking

Advantages:

* simple
* fast
* easy to implement

Weakness:

* can break semantic boundaries

Best for:

* quick prototypes
* raw text without clear structure

#### Recursive chunking

This uses a hierarchy of separators:

* paragraphs
* sentences
* words
* characters

Advantages:

* better semantic preservation than fixed-size chunking
* more stable on long documents

Best for:

* most general-purpose text workloads

#### Semantic chunking

This splits on semantic shifts instead of only length thresholds.

Advantages:

* each chunk is more internally coherent

Weaknesses:

* higher cost
* harder to tune

Best for:

* long documents with clear topic changes

#### Structure-aware chunking

This chunks according to document structure:

* Markdown headers
* HTML sections
* PDF sections
* tables, lists, or code blocks

This is often the best choice for structured technical documents.

It is especially useful for papers, manuals, and research reports.

### 4.4 Practical chunking design advice

Do not store only `chunk_text`.

For production-style systems, also keep:

* `chunk_id`
* `parent_doc_id`
* `section_title`
* `source_page`
* `chunk_order`
* `start_offset / end_offset`

This enables:

* citation generation
* parent-child retrieval
* section-aware answer synthesis
* better failure analysis

A very practical principle is:

> **retrieve with small chunks, generate with larger context**

That means:

* use small chunks for retrieval, to improve precision
* use parent chunks or context windows for generation, to improve completeness

This pattern is especially useful for papers, technical documentation, and long research texts.

### 4.5 What to look at when choosing an embedding model

Embedding quality is often the upper bound of retrieval quality.

At minimum, evaluate:

* **language coverage**
  is the data Chinese, English, or multilingual?
* **domain fit**
  general-purpose embedding or domain-specific embedding?
* **max tokens**
  this constrains chunk size
* **vector dimension**
  this affects storage and retrieval cost
* **speed and deployment cost**
  local model or API?

Do not rely only on public benchmarks.

A safer workflow is:

* shortlist two or three candidates
* evaluate them on your own retrieval cases
* choose based on your own task data

### 4.6 How to choose a vector store or index

For a prototype or local demo:

* `FAISS`
* `Chroma`

are often enough.

For a larger production system, especially if you need:

* high concurrency
* distributed deployment
* complex metadata filtering
* management of multiple indexes

then consider:

* `Milvus`
* `Qdrant`
* `Weaviate`
* `Pinecone`

The core value of a vector database is not merely "it stores vectors."

It is:

* fast ANN search
* metadata filtering
* large-scale index management
* persistence and scalability

---

## 5. How to decompose the online RAG pipeline

A minimal but well-structured RAG pipeline usually has at least these stages:

1. question understanding
2. query construction
3. retrieval
4. reranking or filtering
5. evidence selection
6. answer generation
7. citation or refusal handling

### 5.1 Query construction

The user's raw question is often not the best retrieval query.

Common problems include:

* it is too short
* it is vague
* it is ambiguous
* it contains multiple sub-questions
* the user's wording differs from the document language

So query construction is not optional. It is a real module.

Common methods include:

* **simple rewriting**
  rewrite the question into retrieval-friendly wording
* **multi-query**
  generate several alternative formulations and retrieve with all of them
* **step-back prompting**
  move to a higher-level question before returning to the specific question
* **HyDE**
  generate a hypothetical answer document first, then retrieve using that representation
* **query routing**
  decide which retriever or data source should handle the query

### 5.2 Dense, sparse, and hybrid retrieval

#### Dense retrieval

Advantages:

* semantic understanding
* better generalization

Weaknesses:

* may be weaker on exact terminology, abbreviations, identifiers, and strict lexical matches

#### Sparse retrieval

Examples include BM25.

Advantages:

* strong exact keyword matching
* good for technical terms and identifiers

Weaknesses:

* poor semantic generalization
* more exposed to vocabulary mismatch

#### Hybrid retrieval

This combines dense and sparse retrieval.

It is often the more robust default for production RAG, especially in:

* specialized domains
* paper search
* code retrieval
* settings with many terms, model names, or identifiers

Common fusion methods:

* **RRF**
  rank-based fusion; simple and robust
* **weighted linear fusion**
  normalize dense and sparse scores, then combine them

If you are still early in the project, a practical baseline is:

> start with `dense + BM25 + RRF`

This is often more reliable than pure dense retrieval.

### 5.3 Metadata filtering

In some settings, filtering before retrieval matters more than better embeddings.

For a research assistant, examples include:

* only papers after a certain year
* only a given topic tag
* only primary sources
* only a specific conference or journal

This can significantly reduce noise.

### 5.4 Reranking

Initial retrieval is for **candidate recall**, not necessarily for final ranking quality.

Common rerank methods:

* **RRF**
  often used as result fusion
* **Cross-Encoder**
  accurate but more expensive
* **LLM-based reranker**
  semantically strong but usually expensive
* **ColBERT**
  a balance between accuracy and efficiency

A good system often uses:

* initial retrieval for high recall
* reranking for high precision

### 5.5 Compression

Not every retrieved chunk should be passed in full to the LLM.

Compression aims to:

* remove irrelevant noise
* keep only information directly useful for the question
* reduce token cost

Common approaches:

* dropping weakly relevant documents
* extracting only relevant sentences
* keeping only passages aligned with the query

Many RAG systems do not fail because they retrieve nothing.
They fail because:

> they retrieve enough, but pass too much noisy context into generation.

### 5.6 Context expansion and parent-child retrieval

This is a very important design pattern.

Typical setup:

* retrieve using child chunks
* generate using parent chunks or context windows

Why this matters:

* child chunks improve retrieval precision
* parent chunks improve answer completeness

This is especially useful for research assistants, because many questions require:

* the local paragraph
* nearby context
* the relationship between method, experiment, and limitation sections

---

## 6. Generation is not just "stuff documents into a prompt"

The generation stage should be deliberately designed.

### 6.1 Prompt constraints

At minimum, the model should be instructed to:

* answer only from the provided evidence
* explicitly say when evidence is insufficient
* cite sources where possible
* avoid adding unsupported details

### 6.2 Output format

If the system will later be evaluated, parsed, or integrated into an agent workflow, structured output is strongly preferred.

Typical fields include:

* `answer`
* `citations`
* `confidence`
* `status`

This is much easier to work with than unconstrained free-form text.

### 6.3 Response-mode routing

Not every question should use the same prompt template.

Typical response modes include:

* short factual answer
* answer with citations
* structured summary
* multi-document comparison
* insufficient-evidence refusal

A stronger RAG system usually routes question types to different generation templates.

### 6.4 Uncertainty handling

A good RAG system does not only "answer a lot."
It should also know how to:

* say it does not know
* say the evidence is insufficient
* distinguish between "the document does not say this" and "the system failed to retrieve it"

Many real failures happen because:

> the evidence is not strong enough, but the system answers with too much certainty anyway.

---

## 7. What RAG evaluation should actually measure

Do not judge a RAG system only by whether the final answer "looks good."

A useful framework is the RAG triad.

### 7.1 Context relevance

This asks:

> are the retrieved contexts relevant to the question?

This is mainly about the retriever.

### 7.2 Faithfulness or groundedness

This asks:

> is the final answer actually supported by the retrieved context?

This is mainly about whether the generator stayed grounded.

### 7.3 Answer relevance

This asks:

> did the final answer truly answer the user's question?

An answer can be faithful but still incomplete or off-target.

### 7.4 Retrieval metrics

If you have labeled data, useful retrieval metrics include:

* Precision@k
* Recall@k
* F1
* MRR
* MAP

These help answer:

* does the retriever bring back enough relevant candidates?
* does the ranker place critical evidence near the top?

### 7.5 Response evaluation

Common methods include:

* **rule-based evaluation**
  useful for structured scenarios
* **rubric-based evaluation**
  useful for search or RAG assistants
* **LLM-as-judge**
  useful for large-scale, lower-cost evaluation

For a research assistant, at minimum evaluate:

* retrieval relevance
* evidence consistency
* citation quality
* answer completeness
* uncertainty handling

---

## 8. Common RAG failure modes

Do not reduce all RAG failures to "hallucination."

A more useful failure taxonomy is:

### 8.1 Query formulation failure

The question was not transformed into a retrieval-friendly query.

### 8.2 Retrieval failure

The query is acceptable, but the retriever failed to bring back the key document.

### 8.3 Evidence selection failure

The right document was in the candidate set, but the wrong passage or source was selected.

### 8.4 Answer synthesis failure

The evidence is correct, but the final answer misreads it, exaggerates it, or stitches it together incorrectly.

### 8.5 Uncertainty-handling failure

The system should have said "insufficient evidence," but answered anyway.

### 8.6 Corpus failure

The knowledge base itself is incomplete, outdated, noisy, or poorly organized by source quality.

A very common mistake is:

> blaming every failure on the model size or model quality.

In practice, many RAG failures are caused by:

* corpus design
* chunking
* query rewriting
* retrieval or reranking

not by the LLM alone.

---

## 9. Design guidance for a research assistant

If the goal is a research agent or research assistant, these principles are especially useful.

### 9.1 Do not store only raw chunk text

At minimum, keep:

* `doc_id`
* `chunk_id`
* `parent_doc_id`
* `title`
* `section_title`
* `source_type`
* `published_at`
* `topic_tags`
* `trust_level`
* `page_content`

### 9.2 Default to hybrid retrieval

Research scenarios are full of:

* method names
* dataset names
* acronyms
* metric names
* specialized terminology

Pure dense retrieval often misses exact technical cues.

### 9.3 Preserve parent-child relationships

Many paper-level answers cannot be supported by a tiny chunk alone.

For research assistants, a strong pattern is:

* retrieve child chunks
* expand to parent context
* cite both the parent document and the section

### 9.4 Design for insufficient evidence from the start

A research assistant is not just a search summarizer.
It must be able to say:

* this paper does not say that
* the current evidence is insufficient for that conclusion
* these sources conflict with each other

### 9.5 Freeze benchmark and evaluation early

Before aggressively changing retrievers or models, stabilize:

* gold cases
* document schema
* runtime trace schema
* evaluation rubric

Only then can you tell whether retrieval changes actually improve the system.

---

## 10. A minimal but solid RAG blueprint

If you were building a first serious version today, a practical order is:

### Phase 1: minimal closed loop

* local corpus
* recursive or structure-aware chunking
* one embedding model
* one vector store
* basic retrieval
* basic generation prompt
* gold cases plus evaluation

### Phase 2: stabilize retrieval

* metadata filtering
* hybrid retrieval
* query rewriting
* reranking

### Phase 3: stabilize generation

* citations
* structured output
* uncertainty and refusal
* evidence-aware prompts

### Phase 4: integrate with an agent loop

* multi-step retrieval
* retrieval trace
* retry and reflection
* routing across tools or data sources

---

## 11. A few decision rules worth remembering

* The point of RAG is not "attach a vector database," but "reliably bring external evidence into the answer."
* Chunking is not a minor preprocessing detail; it directly affects retrieval quality.
* Embedding quality often sets the retrieval ceiling; reranking improves final candidate ordering.
* Hybrid retrieval is often more robust than pure dense retrieval, especially in terminology-heavy domains.
* Many RAG failures are not "retrieval found nothing," but "the wrong evidence was used" or "the answer ignored the evidence."
* A good RAG system should know how to cite, refuse, and express uncertainty, not only how to answer.
* For a research assistant, `source quality + parent-child retrieval + evaluation rubric` usually matter from the very beginning.
