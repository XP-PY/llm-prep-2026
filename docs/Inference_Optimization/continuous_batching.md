# Continuous Batching & PagedAttention

## 1) The serving problem we’re trying to solve

In real inference serving (APIs), requests are:

* **Different prompt lengths**
* **Different generation lengths**
* Arrive at **different times**

A GPU is fastest when it stays busy doing large, dense matrix work. But naive batching struggles with variable-length generation.

---

## 2) Why “standard batching” wastes GPU

### Standard static batching (common HF pipeline style)

You form a batch of $B$ requests, then generate token-by-token:

* At each step, the GPU runs on **all $B$** sequences.
* If sequences have different lengths, you must **pad** to align shapes.
* If one request finishes early, it often still “occupies a slot” until the batch is done, or you rebuild the batch (expensive).

Two big inefficiencies:

1. **Padding waste**
   Short sequences still compute on padded positions → wasted FLOPs.

2. **Head-of-line blocking**
   The batch’s progress is limited by the slowest / longest request.
   Some sequences finish early → GPU slots become partially idle.

---

## 3) Continuous Batching (the throughput fix)

**Goal:** keep the GPU batch full **at every decoding step**.

### Key idea

Instead of keeping a fixed batch until everyone finishes, we do:

* When a sequence finishes, **immediately replace it** with a new waiting request.
* Do this **mid-generation**, step after step.

So the batch is *continuous*:

* Some sequences are on step 200 of decoding,
* Some are newly added and on step 1,
* But the GPU always sees a “full” micro-batch to process.

### What you get

* Greatly reduced GPU idle time
* Much less head-of-line blocking
* Much higher throughput in tokens/sec (often multiple × in real serving)

### What it costs

* You need a **scheduler** that:

  * selects which requests to run each step,
  * handles adding/removing sequences frequently,
  * balances fairness vs throughput.

---

## 4) KV cache: the memory bottleneck behind the scenes

During decoding, Transformers store **KV cache** so they don’t recompute attention over the entire prefix each step.

KV cache size grows with total generated tokens:
$$
\text{KV size} \propto \text{layers} \times \text{(total tokens)} \times \text{heads} \times \text{head dim} \times 2
$$
(the $2$ is for **K and V**)

In serving, KV cache is often the main memory limiter.

---

## 5) Why KV cache makes variable-length serving hard

### Naive (contiguous) allocation problem

If each sequence gets a contiguous KV buffer, then:

* Sequences finish at different times → you free holes
* New sequences arrive with different lengths → hard to fit
* You get **fragmentation** and **wasted memory**
* You also can’t flexibly “pack” many sequences efficiently

This becomes painful when you’re constantly swapping sequences in/out (continuous batching).

---

## 6) PagedAttention (the memory + flexibility fix)

**PagedAttention** treats KV cache like **virtual memory paging**:

### Core idea

1. Split KV cache into fixed-size **blocks** (“pages”), e.g. **16–32 tokens per block**.
2. Each sequence’s KV cache becomes a list of blocks:

   * logical block 0 → physical block A
   * logical block 1 → physical block F
   * …
3. Maintain a **mapping table** (logical → physical).
4. Allocate/free blocks dynamically as tokens grow or sequences end.

### Why this helps

* **No need for contiguous memory**
* Less fragmentation → better memory utilization
* Easier to support **many variable-length** sequences at once
* Enables features like **prefix caching / sharing** (reuse KV blocks for common prompts)

Think of it as:

* contiguous KV = “one big array per request” (hard to manage)
* paged KV = “a linked list of fixed pages” (easy to allocate/deallocate)

---

## 7) How they work together

* **Continuous batching** keeps compute busy (throughput)
* **PagedAttention** keeps memory efficient and flexible (KV cache)

Together, they make high-throughput LLM serving practical under real-world traffic.

---

## 8) Comparison table (clean)

| Method                               |                       Throughput vs static batch |             Memory efficiency |                    Latency impact | Complexity | Used in                        |
| :------------------------------------: | :-----------------------------------------------: | :----------------------------: | :--------------------------------: | :---------: | :------------------------------: |
| Static batching (typical HF)         |                                               1× | Low (padding + fragmentation) | Higher (blocked by long requests) |        Low | Basic pipelines                |
| Continuous batching                  |                      ~5–10× (workload-dependent) |                        Higher |     Lower average (less blocking) |     Medium | Production serving             |
| Continuous batching + PagedAttention | Similar throughput, **more stable at high load** |                     Very high |                       Low average |     Medium | Modern high-throughput engines |

*(Exact multipliers depend heavily on model size, traffic pattern, max context, and GPU.)*

---

## 9) PagedAttention “derivation” in plain language

### KV cache growth

Every new generated token requires storing K/V for each layer.

### What paging changes

Instead of reserving a big contiguous KV array for each sequence, you grow it block-by-block:

* If block size is 16 tokens:

  * token 1–16 use block 0
  * token 17–32 use block 1
  * etc.

When a sequence ends:

* you free its blocks immediately
* those blocks can be reused for new sequences right away

### Block size trade-off (why “16–32 typical” shows up)

* **Smaller blocks** → less wasted space at the tail, but more mapping overhead
* **Larger blocks** → less overhead, but more tail waste

So serving engines pick a middle ground.

---

## 10) Simple formulas (what they *mean*)

### Memory savings intuition

If static batching forces padding, then wasted memory/compute rises with padding.

A rough view:
$$
\text{Memory Savings} \approx 1 - \frac{\text{wasted tokens (padding/holes)}}{\text{total tokens kept in KV}}
$$

### Throughput with continuous batching

Throughput is basically:
$$
\text{Throughput} \approx \frac{\text{total tokens processed}}{\text{wall time}}
$$
Continuous batching tries to maximize this by minimizing GPU idle time—i.e., keep the denominator small for the same workload.

---

## 11) Pitfalls / gotchas

* **Scheduler overhead:** if you reshuffle too aggressively, CPU-side scheduling or kernel launch overhead can rise.
* **Paging overhead:** mapping and block management must be efficient; bad choices can hurt performance.
* **Fairness vs throughput:** always filling the batch can starve long/slow requests unless you schedule carefully.

## Code Implementation
```Python
# =========================
# PSEUDOCODE: Continuous Batching + PagedAttention
# (not runnable; for understanding)
# =========================

# -------------------------
# Concepts
# -------------------------
# - Each request is a Sequence (prompt + generated tokens).
# - Scheduler runs a loop: each iteration generates 1 token for many active sequences.
# - Continuous batching: when a sequence finishes, immediately replace it with a waiting one.
# - PagedAttention: KV cache is stored in fixed-size blocks (pages). Each sequence owns a list
#   of physical blocks; a block table maps logical block index -> physical block id.

BLOCK_TOKENS = 16          # typical 16~32
MAX_ACTIVE = 64            # max sequences in one step batch (depends on GPU/mem)
EOS_ID = 2                 # end-of-sequence token id

# -------------------------
# Memory manager for KV cache blocks
# -------------------------
class BlockManager:
    def __init__(self, num_blocks_total):
        self.free_blocks = list(range(num_blocks_total))  # pool of physical block ids

    def alloc_block(self):
        assert self.free_blocks, "OOM: no KV blocks available"
        return self.free_blocks.pop()

    def free_block(self, block_id):
        self.free_blocks.append(block_id)

class PagedKV:
    """
    Stores per-sequence KV in blocks.
    For each sequence:
      - block_table[logical_block_idx] = physical_block_id
      - token_len: total tokens already in this sequence (prompt + generated)
    """
    def __init__(self, block_mgr: BlockManager):
        self.block_mgr = block_mgr
        self.block_table = []      # list[int] of physical block ids
        self.token_len = 0

    def ensure_capacity_for_next_token(self):
        """Allocate a new KV block if next token crosses a block boundary."""
        next_pos = self.token_len  # next token will be written at this index
        if next_pos % BLOCK_TOKENS == 0:
            new_block = self.block_mgr.alloc_block()
            self.block_table.append(new_block)

    def append_one_token_slot(self):
        """Reserve KV slot for 1 new token."""
        self.ensure_capacity_for_next_token()
        self.token_len += 1

    def release_all(self):
        """Free all blocks when sequence finishes."""
        for blk in self.block_table:
            self.block_mgr.free_block(blk)
        self.block_table.clear()

# -------------------------
# Sequence / Request
# -------------------------
class Sequence:
    def __init__(self, request_id, prompt_tokens):
        self.id = request_id
        self.tokens = list(prompt_tokens)
        self.finished = False

        # Each sequence gets its own paged KV allocator
        self.kv = None

    def init_kv(self, kv: PagedKV):
        self.kv = kv
        # Allocate KV blocks for the prompt tokens (already known tokens)
        for _ in range(len(self.tokens)):
            self.kv.append_one_token_slot()

    def needs_more(self, max_new_tokens):
        # simple stopping rule: EOS or max length
        return (not self.finished) and (len(self.tokens) < max_new_tokens)

# -------------------------
# Model interface (abstract)
# -------------------------
def model_step(active_seqs):
    """
    One decoding step for many active sequences.
    With PagedAttention, the attention kernel reads KV by using each seq's block_table mapping.
    This returns next_token per sequence.
    """
    # In reality:
    # - build a "batch plan" with token positions, block tables, and offsets
    # - run fused kernels: attention reads KV from paged blocks
    # Here we just pretend it returns token ids.
    next_tokens = {}
    for s in active_seqs:
        next_tokens[s.id] = fake_next_token(s)  # placeholder
    return next_tokens

def fake_next_token(seq):
    # placeholder: not meaningful
    return 1

# -------------------------
# Scheduler: Continuous batching loop
# -------------------------
class ContinuousBatchScheduler:
    def __init__(self, block_mgr: BlockManager, max_active=MAX_ACTIVE):
        self.block_mgr = block_mgr
        self.max_active = max_active
        self.waiting_queue = []   # incoming requests waiting to run
        self.active = []          # currently running sequences

    def submit(self, seq: Sequence):
        self.waiting_queue.append(seq)

    def _try_admit_new_seqs(self):
        """
        Fill empty slots in active batch with new requests.
        Continuous batching: do this every step, not only at "batch boundaries".
        """
        while len(self.active) < self.max_active and self.waiting_queue:
            seq = self.waiting_queue.pop(0)

            # Attach PagedKV for the sequence
            paged_kv = PagedKV(self.block_mgr)
            seq.init_kv(paged_kv)

            self.active.append(seq)

    def _evict_finished(self):
        """
        Remove finished sequences and free their KV blocks immediately.
        This is the key to keeping memory reusable and batch always full.
        """
        still_active = []
        for seq in self.active:
            if seq.finished:
                seq.kv.release_all()  # PagedAttention: free all pages
            else:
                still_active.append(seq)
        self.active = still_active

    def run(self, max_total_steps, max_new_tokens):
        """
        Main serving loop.
        Each iteration generates 1 new token for each active sequence (if not finished).
        """
        for _step in range(max_total_steps):

            # 1) Evict finished (free KV pages)
            self._evict_finished()

            # 2) Admit new requests to keep GPU busy (continuous batching)
            self._try_admit_new_seqs()

            if not self.active:
                break  # no work

            # 3) One GPU decoding step for current active batch
            next_tokens = model_step(self.active)

            # 4) Append tokens + grow KV by 1 slot per seq
            for seq in self.active:
                if seq.finished:
                    continue

                t = next_tokens[seq.id]
                seq.tokens.append(t)

                # PagedAttention: allocate new KV slot for the appended token
                seq.kv.append_one_token_slot()

                # stopping condition
                if t == EOS_ID or len(seq.tokens) >= max_new_tokens:
                    seq.finished = True

        # final eviction cleanup
        self._evict_finished()


# =========================
# How to read this pseudocode
# =========================
# - Continuous batching is in:
#     _evict_finished() + _try_admit_new_seqs() called EVERY step.
#   This is what replaces finished sequences immediately.
#
# - PagedAttention is in:
#     PagedKV.append_one_token_slot()
#   which allocates KV memory by fixed-size blocks.
#   Attention kernels would use seq.kv.block_table to locate KV pages.
```