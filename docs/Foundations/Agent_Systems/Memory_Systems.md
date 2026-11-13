# Memory Systems for Agents

## 1. Why agents need memory

An agent without memory behaves as if it has to restart the task from scratch at every step.

This creates several direct problems:

* **short-session forgetting**
  the system quickly loses format constraints, user preferences, and recent context
* **poor multi-step execution**
  it does not know what it has already searched, verified, or failed on
* **broken long-term collaboration**
  it cannot retain user preferences, project progress, validated conclusions, or unresolved questions
* **no learning from history**
  successful and failed past episodes cannot improve future behavior

So the goal of memory is not "store all chat history."

It is:

> retain the right information at the right time scale so the system can keep working coherently.

---

## 2. Distinguish memory from RAG first

These two concepts are often mixed together.

### 2.1 RAG is mainly about external knowledge access

RAG answers questions like:

* what facts does the model not know right now?
* which external documents can provide evidence?
* how do we generate an answer grounded in that evidence?

Its focus is:

* external knowledge bases
* retrieval
* evidence selection
* grounded answering

### 2.2 Memory is mainly about internal state and experience

Memory answers questions like:

* where is the current task now?
* what has the system already done?
* what stable user preferences exist?
* what project-level facts and progress have been established?

Its focus is:

* state continuity
* multi-step task progression
* cross-session collaboration
* historical experience

A short distinction is:

* **RAG**: get knowledge from outside the system
* **memory**: preserve the system's own state, experience, and long-term information

In many real systems, both are combined:

* RAG provides external evidence
* memory stores what was already searched, what evidence proved reliable, and what the user consistently cares about

---

## 3. The basic memory lifecycle

A mature memory system is not just "store" and "retrieve."

It usually includes at least five stages:

1. **Encoding**
   Convert incoming information into a form the system can store.
2. **Storage**
   Place it in the right memory layer and the right storage backend.
3. **Retrieval**
   Bring back relevant memory when needed based on the current query, task state, or context.
4. **Consolidation**
   Promote short-lived local information into more stable long-term memory.
5. **Forgetting**
   remove information that is unimportant, stale, duplicated, or no longer valid.

The key principle is:

> not every piece of information should immediately become long-term memory, and not every memory should live forever.

---

## 4. Four useful memory categories

For practical agent systems, the following four-layer view is very useful.

## 4.1 Working memory

Working memory is the short-term buffer for the current session and current task.

Typical contents:

* the current user question
* current output constraints
* recent retrieval results
* current task state
* tentative conclusions

Typical properties:

* short lifetime
* limited capacity
* high access frequency
* optimized more for speed than persistence

It solves:

> how the system avoids forgetting within the current session and task.

Engineering-wise, working memory is often best implemented with:

* in-memory storage
* small queues or buffers
* TTL-based expiration
* lightweight hybrid retrieval such as keywords plus simple vectors

## 4.2 Episodic memory

Episodic memory stores specific events and experiences.

Typical contents:

* what happened in a particular conversation
* a successful or failed tool call
* a full task execution episode
* an experiment, discussion, or decision record

Typical properties:

* strong temporal order
* rich contextual fields
* supports replaying what happened

It solves:

> how the system recalls a concrete experience instead of only storing abstract conclusions.

For a research agent, episodic memory is especially useful because much research work is event-like:

* which papers were read
* which comparison produced a conclusion
* which experiment failed
* which query produced no result

## 4.3 Semantic memory

Semantic memory stores abstract knowledge, stable concepts, and long-term rules.

Typical contents:

* the user's long-term research area
* commonly used datasets, conferences, and method families
* stable project facts
* long-term rules the assistant should follow
* conclusions validated repeatedly over time

Typical properties:

* more stable
* more abstract
* not tied to one specific event

It solves:

> how the system builds a long-term knowledge structure instead of replaying disconnected events forever.

## 4.4 Perceptual memory

Perceptual memory handles multimodal inputs such as:

* images
* audio
* PDF screenshots
* charts
* code screenshots

It usually stores not only the file itself, but also:

* modality type
* summaries
* embeddings
* source and timestamp
* relations to other task objects

It solves:

> how the system incorporates non-text content into long-term task collaboration.

For your current research agent, this is not the first layer to build, but it becomes valuable later for figures, tables, and screenshots from papers.

---

## 5. How different memory types should be designed

Do not make all four memory types identical and differentiate them only with a `memory_type` field.

A better approach is:

* unify the high-level object model
* differentiate behavior by layer

In other words, you can have a shared `MemoryItem`, but each memory type should use different policies.

### 5.1 Working memory

Main priorities:

* speed
* small size
* easy expiration

Good fit:

* in-memory lists or queues
* lightweight indexing
* session-level isolation

### 5.2 Episodic memory

Main priorities:

* temporal order
* event completeness
* replayability

Good fit:

* structured document storage plus vector retrieval
* explicit `session_id` and `task_id`
* time-based filtering

### 5.3 Semantic memory

Main priorities:

* abstract knowledge
* stable relations
* durable persistence

Good fit:

* vector stores
* graph databases
* structured knowledge tables

### 5.4 Perceptual memory

Main priorities:

* multimodal representations
* modality-specific indexes
* cross-modal retrieval

Good fit:

* separate storage for different modalities
* shared metadata conventions
* optional cross-modal alignment

---

## 6. What a memory item should minimally contain

A minimal but useful `MemoryItem` should usually contain:

* `memory_id`
* `memory_type`
* `content`
* `importance`
* `timestamp`
* `session_id`
* `user_id`
* `task_id`
* `source`
* `metadata`

If you want a more production-oriented design, also add:

* `status`
  for example: verified, tentative, stale
* `ttl`
  expiration time
* `embedding_ref`
  pointer to vector representation
* `parent_memory_id`
  useful for consolidation and provenance tracking

The most important principle is:

> memory should not be only a string; it should be an object with status, provenance, and lifecycle.

---

## 7. Memory writes should be selective

This is one of the first major failure points in memory design.

Without a write policy, memory quickly becomes polluted by:

* temporary noise
* wrong guesses
* duplicated content
* stale project status
* user remarks that have no long-term value

So define at least three kinds of rules.

### 7.1 Write rules

What is worth storing?

Examples of good candidates:

* stable user preferences
* high-value task state
* validated conclusions
* clearly important interaction events

Examples of things that should not automatically become long-term memory:

* one-off guesses
* unverified model inferences
* duplicated retrieval outputs

### 7.2 Update rules

When should an existing memory be replaced or revised?

For example:

* the user's preference changed
* the project stage was updated
* a conclusion was overturned by new evidence

### 7.3 Trust rules

How reliable is a memory item?

At minimum, it is useful to distinguish:

* `verified`
* `tentative`
* `derived`
* `stale`

This is especially important for a research agent, because "a model-generated inference" and "a conclusion supported by paper evidence" should not be treated as equally reliable.

---

## 8. Memory retrieval is not just string matching

The goal of memory retrieval is not "return all vaguely relevant text."

It is:

> under the current task, return the most useful, most trustworthy, and most action-relevant memories.

A stronger retrieval score often combines:

* **semantic similarity**
* **keyword overlap**
* **recency**
* **importance**
* **memory type**
* **trust status**

### 8.1 Working-memory retrieval

Working memory often benefits from:

* TF-IDF or keyword search
* lightweight embeddings
* recency and importance weighting

because it is smaller, newer, and strongly tied to the current task.

### 8.2 Episodic-memory retrieval

Episodic memory often needs:

* semantic retrieval
* time filtering
* session filtering
* event-type filtering

It is not only about semantic similarity. It is also about:

* whether the episode belongs to the same phase
* whether it belongs to the same task
* whether it happened recently

### 8.3 Semantic-memory retrieval

Semantic memory is often better served by:

* vector retrieval
* relation retrieval
* graph reasoning

If you only use vector search, you may miss concept-level relations.

For example, in a research system:

* "hyperspectral target detection"
* "anomaly detection"
* "targetness prior"

may require conceptual links rather than simple lexical overlap.

---

## 9. Forgetting is not a bug, it is a feature

Many people implicitly assume:

* once memory is stored, it should always remain available

That usually leads to failure.

A system with no forgetting mechanism often becomes:

* noisier over time
* slower over time
* more internally inconsistent over time

### 9.1 Common forgetting strategies

#### Importance-based forgetting

Remove low-importance memories.

Useful for:

* temporary noise
* low-value records

#### Time-based forgetting

Remove memories older than a threshold.

Useful for working memory and some episodic memory.

#### Capacity-based forgetting

When memory exceeds a capacity limit, drop the least important or least useful items.

Useful for working memory and cache-like layers.

### 9.2 Forgetting does not have to mean hard deletion

Sometimes the better operation is:

* archive
* demote
* mark as stale

rather than permanent deletion.

For a research agent, for example:

* an old project state can become `stale`
* a conclusion invalidated by new evidence can become `superseded`

This preserves traceability without polluting current retrieval.

---

## 10. Consolidation is the most valuable part

A mature memory system should not only "remember many things."

It should also know how to:

> turn short-term information into durable knowledge.

That is what consolidation does.

Typical transitions include:

* `working -> episodic`
  important short-term task state becomes a durable event record
* `episodic -> semantic`
  repeated concrete episodes become long-term knowledge or rules

For example:

* if multiple conversations indicate that the user prefers structured answers
  - that should not remain only as separate events
  - it should become a stable long-term preference

* if repeated paper comparisons suggest that a method works better for few-shot hyperspectral classification
  - those repeated episodes may eventually become a semantic research note

A practical interpretation is:

* temporary task information -> working memory
* important event -> episodic memory
* stable pattern or rule -> semantic memory

---

## 11. How to choose storage backends

Not every memory type should go into the same database.

### 11.1 Working memory

Recommended:

* in-memory storage
* small local caches

because the main priorities are:

* speed
* expiration
* low overhead

### 11.2 Episodic memory

Recommended:

* document stores, SQLite, or JSONL
* combined with vector search

because episodic memory needs both structured fields and semantic lookup.

### 11.3 Semantic memory

Recommended:

* vector stores
* graph databases
* structured knowledge stores

depending on whether you need:

* semantic recall only
* or also relation reasoning and graph-style traversal

### 11.4 Perceptual memory

Recommended:

* modality-separated storage
* file objects plus metadata plus modality-specific embeddings

Do not force all modalities into an identical storage shape if the retrieval behavior differs.

---

## 12. Design guidance for a research agent

Because your current project is a research agent, I recommend building memory in this order.

### 12.1 First: task memory

The first thing to build is not a fancy long-term user profile.
It is:

* the current question
* current retrieval queries
* which documents have already been read
* which evidence has been confirmed
* current candidate conclusions
* failure reasons

This directly affects whether the system can complete multi-step research tasks.

### 12.2 Second: project progress memory

For research work, the highest-value long-horizon memory is often:

* the current project phase
* completed reading
* compared methods
* preliminary conclusions
* next planned steps

This is usually more important than style personalization.

### 12.3 Third: semantic memory

Once you have accumulated enough:

* reading traces
* method comparisons
* benchmark notes
* recurring concepts

you can gradually consolidate them into:

* method families
* dataset knowledge
* long-term terminology notes
* a stable research knowledge layer

### 12.4 Do not start with a heavy knowledge graph too early

Many people reach for Neo4j or graph extraction too early.

If your benchmark, schema, and evaluation are not yet stable, that often creates unnecessary complexity.

A safer order is:

1. task memory
2. project notes
3. semantic store
4. graph augmentation

---

## 13. Common failure modes of memory systems

### 13.1 Storing too much

The system stores everything, so retrieval later becomes mostly noise.

### 13.2 Storing wrong information

The system writes hallucinations, incorrect guesses, or unsupported conclusions into long-term memory.

### 13.3 No update mechanism

Old preferences, old conclusions, and old project states remain active and pollute current tasks.

### 13.4 No layering

Short-term context, long-term knowledge, and project state are mixed into one undifferentiated pool.

### 13.5 No alignment with trace

If memory and runtime trace are disconnected, failure analysis and later data generation become much harder.

The most important warning is:

> a bad memory system does not make an agent smarter; it makes the agent repeat its mistakes more consistently.

---

## 14. A minimal practical memory blueprint

If you had to build a useful first version today, a practical order is:

### Phase 1

* session-level working memory
* explicit task state object
* simple importance field
* TTL expiration

### Phase 2

* project progress records
* episodic archiving
* basic semantic retrieval

### Phase 3

* long-term user preferences
* semantic-memory consolidation
* `working -> episodic -> semantic` promotion

### Phase 4

* multimodal perceptual memory
* graph augmentation
* explicit memory-quality evaluation

---

## 15. A few decision rules worth remembering

* The goal of memory is not "store more," but "retain the right information at the right time scale."
* Memory is not the same thing as chat history; chat history is only one input source.
* Working memory supports current task progression; long-term memory supports long-term collaboration.
* Episodic memory stores events; semantic memory stores patterns and stable knowledge.
* Forgetting is a necessary ability, not a defect.
* Consolidation matters more than storage volume, because it determines whether experience becomes knowledge.
* For a research agent, task memory and project progress memory are usually more important than early user-persona engineering.
