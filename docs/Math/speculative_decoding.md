# Speculative Decoding

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
p_reject ≈ probability of mismatch per step.

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