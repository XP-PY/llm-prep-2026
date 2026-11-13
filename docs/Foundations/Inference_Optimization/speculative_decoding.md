<!-- # Speculative Decoding

## Principle: Breaking Autoregressive Bottlenecks
Vanilla autoregressive decoding generates one token at a time → latency linear in output length.
**Speculative Decoding** parallelizes drafting: Use a fast draft model (or extra heads) to propose multiple tokens, verify in batch with the target model → accept prefix until mismatch.
Multi-angle:
- **Theoretical:** Expected speedup ≈ γ + 1, where γ = average accepted tokens (3–6 typical).
- **Real-World Cases:** Medusa in vLLM → 2–2.5x faster on Grok/Mistral APIs; Lookahead in DeepMind papers → 3x on Gemini prototypes.
- **Pitfalls:** Draft model mismatch → low γ; tree explosion in advanced variants.
- **Future Implications:** Combine with distillation or multimodal (e.g., speculate image tokens); verify latest arXiv for EAGLE/MLA hybrids.

### Comparison Table
|Method|Speedup (typical)|Extra Params/Model|Acceptance Rate γ|Quality Impact|Complexity|Used In|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Greedy Sampling|1x|None|N/A|Baseline|Low|Standard HF generate|
|Medusa (heads)|2–2.5x|~1–5%|4–6|None|Low|vLLM,production APIs|
|Lookahead (n-gram)|2.5–3.5x|None|5–8|Slight ↑|Medium|Recent SOTA papers|
|Full Speculative (small draft model)|2–4x|Separate model|6–10|None|High|Academic (SpecInfer)|

## Detailed Derivation (Medusa Focus — Simplest Production Variant)
Add k Medusa heads on top of base model (linear projections from final hidden states).
Draft phase: Predict k future tokens in parallel (tree of size branching factor b^k).
Verify phase: One forward pass on base model over the drafted sequence.
Accept longest correct prefix (length γ).
Repeat until max length.
**Expected Speedup:**
$$\text{Speedup} = \frac{\gamma + 1}{1 + \frac{k}{\gamma + 1}} \quad \text{(k = draft steps, simplified)}$$
Real formula (Leviathan et al.):
$$\text{Speedup} = \frac{1 + \gamma}{1 + p_{\text{reject}}}$$
p_reject ≈ probability of mismatch per step. -->

# Speculative Decoding

## 1) Why decoding is slow

**Autoregressive decoding** normally generates **one token per step**:

* Step 1: run the big model → get token 1
* Step 2: run the big model again → get token 2
* …

So if you generate (L) tokens, you do about **(L) expensive model steps**.
That’s the “autoregressive bottleneck”.

---

## 2) Core idea: “Draft many, verify once”

**Speculative Decoding** speeds this up by splitting generation into two roles:

* **Draft (cheap):** propose the next (k) tokens quickly
* **Verify (expensive but batched):** run the real target model once to check the whole chunk
* **Accept:** keep the longest prefix that matches what the target model would have chosen

### The loop (one cycle)

1. Start with current prefix (x).
2. Draft model proposes a chunk:
$$
   \hat{y}_1, \hat{y}_2, \dots, \hat{y}_k
$$
3. Target model verifies in one forward pass on ($x + \hat{y}_{1:k}$).
4. Accept the longest prefix of the draft that agrees with the target:

   * accepted length $\gamma$
5. Append those $\gamma$ tokens to output and repeat.

**Key metric:**
$\gamma$ = *average number of drafted tokens accepted per cycle.*

If $\gamma$ is large, you get speedups because **one target-model pass produces multiple final tokens**.

---

## 3) Why speedup happens (intuition)

Baseline greedy (big model only):

* roughly **1 big-model step per output token**

Speculative decoding:

* roughly **1 big-model step per $\gamma$ output tokens**
* plus some cheap drafting overhead

So big-model work per token goes down by about a factor of ($\gamma$.

A common rule of thumb:

* If $\gamma \approx 3 \text{to} 6$, you can often see **~2× to 4×** speedups in practice (depending on overheads).

---

## 4) Common variants (what changes between methods)

### Comparison Table (interpreting each column)

* **Speedup:** what people often see in practice
* **Extra Params/Model:** what you need to add
* **Acceptance rate $\gamma$:** larger is better
* **Quality impact:** usually none if the target model is the “judge”
* **Complexity:** engineering difficulty

|             Method             | Main idea                                             | Speedup (typical) |  Extra Params/Model  | Typical $\gamma$ |      Quality Impact     | Complexity | Used In            |
| :---: | :---: | :---------------: | :------------------: | :--------------: | :---------------------: | :--------: | :---: |
|         Greedy Sampling        | big model generates 1 token/step                      |         1×        |         None         |        N/A       |         Baseline        |     Low    | HF `generate()`    |
|         Medusa (heads)         | add $k$ extra “future-token” heads to the same model  |       2–2.5×      |     ~1–5% params     |        4–6       |           None          |     Low    | vLLM / production  |
|       Lookahead (n-gram)       | draft tokens using simple heuristics / cache patterns |      2.5–3.5×     |         None         |        5–8       | sometimes slight change |   Medium   | research systems   |
| Full Speculative (draft model) | small model drafts, big model verifies                |        2–4×       | separate small model |       6–10       |           None          |    High    | academic / systems |

**Important:** all these aim to reduce **how often you run the big model**.

---

## 5) Medusa-focused explanation (simplest production variant)

### What Medusa adds

* Keep the **same backbone model**.
* Add **$k$ lightweight heads** (usually linear layers) on top of final hidden states.
* Head (i) tries to predict token (t+i) (future tokens).

### Medusa decoding cycle

1. **Draft (cheap):** the $k$ heads propose $k$ future tokens in parallel.
2. **Verify (expensive):** run the backbone once over the proposed chunk.
3. **Accept:** keep the longest prefix that matches the backbone’s greedy choice.
4. Repeat.

---

## 6) Speedup formulas (what they mean, not just math)

### A practical mental model

* Each cycle costs about:

  * **1 target-model verification step** (expensive)
  * draft overhead (cheap)
* Each cycle produces about **$\gamma$** accepted tokens (sometimes plus a guaranteed progress token)

So speedup increases when:

* $\gamma$ increases (draft aligns well with target)
* overhead decreases (draft is very cheap, verification is efficient)

### Two common simplified expressions

**Heuristic / simplified view:**
$$
\text{Speedup} \approx \text{(tokens gained per verify)} \sim \gamma (\text{or } \gamma+1 \text{ depending on variant})
$$
Meaning: *one verify pass yields multiple output tokens.*

**Another abstract approximation:**
$$
\text{Speedup} = \frac{1+\gamma}{1+p_{\text{reject}}}
$$

* $p_{\text{reject}}$: chance that draft mismatches early → wasted drafted tokens → lower speedup
  This says: *more acceptance helps; more rejection hurts.*

*(Exact formulas differ by paper/system because they count costs differently: draft compute, KV-cache ops, batching, etc.)*

---

## 7) Common pitfalls (why speedups sometimes disappoint)

* **Draft mismatch → small $\gamma$:** if the draft is wrong early, you accept few tokens.
* **Overheads dominate:** memory bandwidth, KV-cache movement, kernel launch overhead.
* **Tree explosion (advanced variants):** too many branches → kills efficiency.

---

## 8) Future directions (what people try next)

* **Distill a better draft model** (raise $\gamma$)
* **Multimodal speculation** (e.g., speculate image/audio tokens)
* **Hybrid approaches** that mix speculation with other inference tricks (keep an eye on newer arXiv work)

---

If you want, I can also rewrite this into a **one-page “cheat sheet”** with a diagram-like ASCII flow and a tiny numeric example ($\gamma=5, k=8$) to make the speedup feel concrete.


## Code Implementation
```Python
# ========= Shared interfaces (toy) =========
# Assume `model` returns logits for next token given a prefix.
# token ids are ints; argmax is "greedy".
def next_token_greedy(model, prefix_tokens):
    logits = model(prefix_tokens)              # shape: [vocab]
    return int(logits.argmax())

def forward_logits_for_positions(model, tokens):
    """
    Target model forward on the whole sequence (prefix + draft),
    returning per-position logits for "next token at each position".
    For position i, logits[i] predicts tokens[i+1].
    (This is how verifier checks many tokens in one pass.)
    """
    return model(tokens, return_all_logits=True)  # shape: [len(tokens), vocab]

# ========= 1) Greedy Sampling (baseline) =========
def greedy_generate(target_model, prompt, max_new_tokens):
    out = list(prompt)
    for _ in range(max_new_tokens):
        t = next_token_greedy(target_model, out)
        out.append(t)
    return out

# ========= 2) Full Speculative Decoding (small draft model + big target model) =========
def speculative_generate(target_model, draft_model, prompt, max_new_tokens, draft_k=8):
    out = list(prompt)

    while len(out) < len(prompt) + max_new_tokens:
        # --- Draft phase (fast): propose k tokens ---
        draft = []
        tmp = out
        for _ in range(draft_k):
            d = next_token_greedy(draft_model, tmp + draft)
            draft.append(d)

        # --- Verify phase (slow): one target forward over (out + draft) ---
        seq = out + draft
        logits_all = forward_logits_for_positions(target_model, seq)

        # --- Accept longest prefix that matches target greedy choices ---
        accepted = 0
        for j in range(len(draft)):
            # position of logits that predicts token at out_len + j
            pos = len(out) - 1 + j
            target_choice = int(logits_all[pos].argmax())
            if target_choice == draft[j]:
                accepted += 1
            else:
                break

        # output accepted tokens
        out.extend(draft[:accepted])

        # if mismatch happened early, append the target token at mismatch position
        # (common variant to ensure progress)
        if accepted < len(draft) and len(out) < len(prompt) + max_new_tokens:
            pos = len(out) - 1
            t = int(logits_all[pos].argmax())
            out.append(t)

    return out

# ========= 3) Medusa (multi-head) sketch =========
class MedusaHeads:
    """
    Toy: k heads predict future tokens from the same final hidden state.
    In practice: heads are linear projections trained to predict t+1, t+2, ..., t+k.
    """
    def __init__(self, k):
        self.k = k

    def predict_k_future_tokens(self, base_model, prefix_tokens):
        h = base_model(prefix_tokens, return_last_hidden=True)  # last hidden state
        # heads produce k distributions, each for a future step
        # logits_list[i] predicts token at step i+1
        logits_list = base_model.medusa_heads(h)                # list of [vocab]
        draft = [int(logits.argmax()) for logits in logits_list]
        return draft

def medusa_generate(target_model_with_medusa, prompt, max_new_tokens, k=6):
    out = list(prompt)
    medusa = MedusaHeads(k)

    while len(out) < len(prompt) + max_new_tokens:
        # --- Draft via heads (cheap) ---
        draft = medusa.predict_k_future_tokens(target_model_with_medusa, out)

        # --- Verify via the SAME target backbone (one forward) ---
        seq = out + draft
        logits_all = forward_logits_for_positions(target_model_with_medusa, seq)

        # --- Accept longest prefix matching target ---
        accepted = 0
        for j in range(len(draft)):
            pos = len(out) - 1 + j
            target_choice = int(logits_all[pos].argmax())
            if target_choice == draft[j]:
                accepted += 1
            else:
                break

        out.extend(draft[:accepted])

        # ensure progress if mismatch
        if accepted < len(draft) and len(out) < len(prompt) + max_new_tokens:
            pos = len(out) - 1
            out.append(int(logits_all[pos].argmax()))

    return out

# ========= 4) Lookahead (n-gram) sketch =========
def build_ngram_index(tokens, n=4):
    """
    Maps context tuple (n-1 tokens) -> list of next tokens observed after it.
    This is a naive "lookahead cache" built from already generated text.
    """
    idx = {}
    for i in range(len(tokens) - n + 1):
        ctx = tuple(tokens[i:i+n-1])
        nxt = tokens[i+n-1]
        idx.setdefault(ctx, []).append(nxt)
    return idx

def propose_from_ngram(out, ngram_index, n=4, k=8):
    draft = []
    tmp = list(out)
    for _ in range(k):
        ctx = tuple(tmp[-(n-1):]) if len(tmp) >= n-1 else None
        if ctx is None or ctx not in ngram_index:
            break
        # simplest: pick the most frequent next token observed
        candidates = ngram_index[ctx]
        next_tok = max(set(candidates), key=candidates.count)
        draft.append(next_tok)
        tmp.append(next_tok)
    return draft

def lookahead_ngram_generate(target_model, prompt, max_new_tokens, n=4, k=8):
    out = list(prompt)
    ngram_index = build_ngram_index(out, n=n)

    while len(out) < len(prompt) + max_new_tokens:
        # --- Draft via n-gram cache (very fast, heuristic) ---
        draft = propose_from_ngram(out, ngram_index, n=n, k=k)
        if not draft:
            # fallback: normal greedy step
            out.append(next_token_greedy(target_model, out))
            ngram_index = build_ngram_index(out, n=n)
            continue

        # --- Verify with target model ---
        seq = out + draft
        logits_all = forward_logits_for_positions(target_model, seq)

        accepted = 0
        for j in range(len(draft)):
            pos = len(out) - 1 + j
            target_choice = int(logits_all[pos].argmax())
            if target_choice == draft[j]:
                accepted += 1
            else:
                break

        out.extend(draft[:accepted])

        # progress on mismatch
        if accepted < len(draft) and len(out) < len(prompt) + max_new_tokens:
            pos = len(out) - 1
            out.append(int(logits_all[pos].argmax()))

        # update cache from new text
        ngram_index = build_ngram_index(out, n=n)

    return out
```