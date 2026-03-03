# Hallucination Mitigation at Inference Time

## 1) Why LLMs hallucinate (a practical mental model)

### 1.1 Two major causes

1. **Knowledge gap / missing evidence**

   * The model lacks the needed fact in its parametric memory, or it’s outdated.
   * It still must output *something*, so it “fills in.”

2. **Epistemic uncertainty disguised as confidence**

   * Even when it’s unsure, the next-token distribution can still produce plausible-sounding text.
   * Decoding (greedy/beam) can amplify the most “confident-sounding” completion.

### 1.2 What we can control at inference time

Without retraining, we can only control:

* **Decoding policy** (how we choose tokens)
* **Sampling and agreement checks** (how stable the answer is across runs)
* **External verification** (search/retrieval/tools)
* **Self-verification prompting** (make the model inspect and correct itself)

---

## 2) Core idea: Inference-time mitigation

### Principle: Reduce factual errors without retraining

**Inference-Time Mitigation** = intervene during generation to:

* detect uncertainty or inconsistency,
* verify against evidence,
* regenerate or revise only the risky parts.

This is usually **cheaper than training**, and can be deployed quickly.

---

## 3) Four “workhorse” methods (what they do + when to use)

### 3.1 SelfCheckGPT (Manakul et al., 2023) — “agreement = confidence”

**Idea:** If the model is truthful, it tends to produce **consistent** facts across multiple stochastic samples.
**Procedure:**

1. Sample $N$ responses with temperature $T>0$
2. Split the main answer into sentences/claims
3. For each claim, measure its similarity/entailment to claims in other samples
4. Low agreement → likely hallucination → flag or rewrite

**Best for:** long-form answers, summaries, medical/legal explanations, anything where you can afford extra compute.

---

### 3.2 Entropy-based decoding — “uncertainty-aware token selection”

**Idea:** High entropy of the next-token distribution implies the model is uncertain.
Instead of blindly taking the top token, you **penalize** uncertain steps or trigger verification.

Let $p(\cdot \mid x_{<t})$ be next-token distribution. Entropy:
$$
H_t = -\sum_{v \in V} p(v)\log p(v)
$$

**Use it in two ways:**

* **Soft control:** increase conservatism when $H_t$ is high (lower temperature, smaller top-p)
* **Hard trigger:** if $H_t$ exceeds threshold, switch to a verification step (RAG/tool call)

**Best for:** low-overhead deployments where you can’t sample 10 times.

---

### 3.3 Retrieval-augmented verification — “trust evidence, not vibes”

**Idea:** For factual questions, the model must ground its answer in retrieved text.
Common pattern:

1. Draft answer
2. Extract claims (or key entities)
3. Retrieve evidence via search / vector DB
4. Verify: do retrieved passages support the claims?
5. If not, revise answer with citations or abstain

**Best for:** knowledge-heavy assistants, up-to-date facts, domain QA.

---

### 3.4 Chain-of-Verification (CoVe) — “ask-check-rewrite”

**Idea:** Force the model to become its own fact-checker:

1. Generate an initial response
2. Generate verification questions (what must be true for this answer to be correct?)
3. Answer those questions (possibly with retrieval/tools)
4. Rewrite the final response using only verified content

**Best for:** structured tasks, safety-critical outputs, compliance-heavy domains.

---

## 4) Comparison table (more engineering-realistic)

| Method                     |            Extra Cost | What it catches best           | Needs Sampling? | Needs Tools/Retrieval? | Best Use Case              |
| -------------------------- | --------------------: | ------------------------------ | --------------- | ---------------------- | -------------------------- |
| Standard greedy / low-temp |                  1.0× | none (baseline)                | No              | No                     | fastest responses          |
| Entropy-aware decoding     |            ~1.05–1.2× | local uncertainty spikes       | No              | Optional               | cheap “risk-aware” serving |
| CoVe                       |                 ~2–5× | missing steps, shaky reasoning | Optional        | Optional/Recommended   | structured verification    |
| SelfCheckGPT               | ~3–10× (depends on N) | unstable facts in long answers | Yes             | No                     | long-form fact-checking    |
| Retrieval-aug verification |              variable | outdated/parametric gaps       | No              | Yes                    | factual QA with citations  |

> Note: The “+20–30% factuality” style numbers are **dataset- and setup-dependent**. Treat gains as “often meaningful” rather than guaranteed.

---

## 5) Detailed derivation (SelfCheckGPT-style consistency)

### 5.1 Setup

* Prompt: $P$
* Sample $N$ responses: $R_1, R_2, \dots, R_N$
* Choose one as **reference answer** $R_{\text{ref}}$ (often the first sample, or a greedy decode)

Split $R_{\text{ref}}$ into claims/sentences:
$$
R_{\text{ref}} = {c_1, c_2, \dots, c_M}
$$

### 5.2 Consistency score for one claim

For claim $c_j$, compute similarity against the other samples:
$$
\text{Score}(c_j) = \frac{1}{N-1} \sum_{i \neq \text{ref}} \text{Sim}(c_j, R_i)
$$

Where **Sim** can be:

* sentence embedding cosine similarity (fast)
* BERTScore (stronger but heavier)
* NLI entailment probability (best quality but heavier)

### 5.3 Thresholding

Flag as risky if:
$$
\text{Score}(c_j) < \tau
$$
Typical $\tau$ ranges:

* embedding cosine: ~0.70–0.85
* BERTScore F1: ~0.60–0.80
  (You should calibrate $\tau$ on a dev set.)

### 5.4 What to do with flagged claims

Options (in increasing cost):

1. **Annotate**: “This part may be uncertain…”
2. **Regenerate only the flagged sentences**
3. **Trigger retrieval verification** only for flagged claims
4. **Abstain** if evidence cannot be found

---

## 6) Practical pitfalls (what breaks in real systems)

### Pitfall A: Sampling overhead explodes latency

* Sampling $N=10$ is expensive.
  **Fix:** use $N=3$ to $5$ + only do it when risk is high (entropy trigger).

### Pitfall B: False positives on creative tasks

* Stories, metaphors, brainstorming will appear “inconsistent” by design.
  **Fix:** only enable strict checks in factual modes (QA, summarization, citations-required).

### Pitfall C: Agreement does not guarantee truth

* The model can consistently repeat the same wrong “myth.”
  **Fix:** for high-stakes facts, add retrieval-based verification.

### Pitfall D: Claim extraction is messy

* Splitting into sentences is easy but misses multi-sentence claims.
  **Fix:** start sentence-level (cheap) → later add claim extraction via a smaller “claim parser” prompt.

---

## 7) Step-by-step code (drop-in notebook friendly)

### Cell 1: Sample multiple responses (HF-style)

```python
import torch

@torch.inference_mode()
def sample_responses(tokenizer, model, prompt, n=5, temperature=0.7, top_p=0.9, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outs = []
    for _ in range(n):
        gen = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        outs.append(text)
    return outs
```

### Cell 2: Split into sentences / claims (baseline)

```python
import re

def split_sentences(text: str):
    # simple heuristic; replace with spaCy later if you want
    sents = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [s for s in sents if s]
```

### Cell 3: Fast consistency scoring with sentence embeddings

This is the “good enough” baseline: **cheap + stable**.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

def consistency_scores(reference_answer: str, other_answers: list[str]):
    ref_sents = split_sentences(reference_answer)
    other_texts = other_answers

    other_embs = embedder.encode(other_texts, normalize_embeddings=True)
    scores = []
    for s in ref_sents:
        s_emb = embedder.encode([s], normalize_embeddings=True)[0]
        sims = (other_embs @ s_emb)  # cosine since normalized
        scores.append(float(np.mean(sims)))
    return ref_sents, scores
```

### Cell 4: Flag & selective rewrite (cheap intervention)

```python
def flag_claims(sentences, scores, tau=0.78):
    flagged = []
    for s, sc in zip(sentences, scores):
        if sc < tau:
            flagged.append((s, sc))
    return flagged

def build_rewrite_prompt(original_answer: str, flagged_claims: list[tuple[str, float]]):
    claim_text = "\n".join([f"- {c} (score={sc:.2f})" for c, sc in flagged_claims])
    return f"""You wrote the answer below.

Answer:
{original_answer}

The following sentences may be unsupported or uncertain:
{claim_text}

Rewrite the answer to be maximally factual:
- Remove or soften any unsupported claims
- If you are unsure, say you are unsure
- Prefer verifiable statements
Return the rewritten answer only."""
```

### Cell 5: Benchmark idea (don’t overcomplicate)

Start with one dataset:

* TruthfulQA (factual QA)
* HaluEval (hallucination eval)
  Log:
* baseline factuality score
* mitigation factuality score
* latency multiplier
