# [SmolVLM: Redefining Small and Efficient Multimodal Models](https://arxiv.org/abs/2504.05299)

> **Brief:** **Small vision-language model:** SigLIP + SmolLM2, pixel-shuffle visual token compression, image splitting, video SFT, and edge/WebGPU-friendly inference.

## Convenient Links

* [Paper (arXiv)](https://arxiv.org/abs/2504.05299)
* [Hugging Face Blog](https://huggingface.co/blog/smolvlm2)
* [SmolVLM2 Model Collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever)
* [SmolVLM WebGPU Demo](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM2)
* [SmolVLA note in this repo](../Robotics/Policies/SmolVLA.md)
* [SigLIP note in this repo](./SigLIP.md)
* [Gemma 3 note in this repo](./Gemma_3.md)
* [DeepSeek-VL note in this repo](./DeepSeek_VL.md)

## 1. One-Sentence Summary

SmolVLM is a family of compact Hugging Face vision-language models from **256M to 2.2B parameters** that combines **SigLIP vision encoders**, **SmolLM2 language backbones**, aggressive visual token compression, image splitting, and video-aware instruction tuning to deliver strong image and video understanding with very low GPU memory usage.

## 2. Why SmolVLM Matters

Large VLMs are capable, but they are often impractical for local or edge deployment:

* many billions of parameters
* high visual-token counts
* large KV-cache and activation memory
* expensive video inference
* poor fit for phones, browsers, and small GPUs

SmolVLM asks a more deployment-oriented question:

> How much multimodal capability can we keep if the model must run under tight memory and latency budgets?

The paper matters because it does not only shrink a large VLM. It studies which design choices actually matter at small scale:

* balanced vision encoder vs. language model size
* longer context windows for visual tokens
* pixel shuffle for token compression
* image splitting for high-resolution inputs
* learned positional tokens for split images
* prompt and media boundary formatting
* careful data mixtures for image, text, and video

The central lesson is that small VLMs need their own recipe. Simply copying large-model design choices wastes memory and can hurt performance.

## 3. Model Family

The paper introduces three main variants:

| Model          | Vision encoder        | Language backbone | Main target |
| :------------- | :-------------------- | :---------------- | :---------- |
| SmolVLM-256M   | `93M` SigLIP-B/16     | SmolLM2-135M      | sub-1GB edge inference |
| SmolVLM-500M   | `93M` SigLIP-B/16     | SmolLM2-360M      | stronger edge / laptop use |
| SmolVLM-2.2B   | `400M` SigLIP-SO400M  | SmolLM2-1.7B      | best quality while still memory-efficient |

The smallest model runs single-image inference with less than `1 GB` of GPU memory, while the largest model reaches much stronger benchmark performance at about `4.9 GB` for batch size 1.

## 4. High-Level Architecture

The architecture is:

```text
image or video frames
-> optional image splitting / frame sampling
-> SigLIP vision encoder
-> pixel shuffle token compression
-> linear / MLP projection
-> SmolLM2 token sequence
-> text output
```

Text and visual tokens are concatenated into one sequence and processed by the language model with self-attention.

This is the same broad VLM pattern used by many modern multimodal LLMs:

```text
vision tokens + text tokens -> decoder-only language model
```

The difference is that SmolVLM is tuned around memory efficiency at every step.

## 5. Vision-Language Token Flow

For an image input, the flow is:

```text
image
-> split into sub-images when needed
-> encode each image region with SigLIP
-> apply pixel shuffle to reduce spatial tokens
-> project to SmolLM2 embedding space
-> interleave with text tokens
-> generate answer tokens
```

For video, the model samples frames and treats them as a sequence of visual inputs.

The paper explicitly avoids frame averaging in the final design because averaging multiple frames degraded video performance. Keeping separate frame representations gives the model better temporal evidence, even though it costs more tokens.

## 6. Encoder-LM Balance

A key finding is that small VLMs need a balanced allocation between the vision tower and the language tower.

The paper tests:

* SmolLM2-135M
* SmolLM2-360M
* SmolLM2-1.7B
* SigLIP-B/16 at about `93M`
* SigLIP-SO400M at about `428M`

The result is not simply "bigger vision encoder is better."

For the smallest language model, the large vision encoder is inefficient. The language model does not have enough capacity to use the extra visual features well. At intermediate scale, the large encoder helps but adds many parameters. At the largest scale, the stronger encoder becomes more reasonable because it is a smaller fraction of total model size.

Practical rule:

> In compact VLMs, the vision encoder should be scaled in proportion to the language model, not copied from a larger architecture.

## 7. Context Length

A single `512 x 512` image encoded by SigLIP-B/16 can produce `1024` visual tokens before compression.

That is already half of a `2k` language-model context window before including:

* split image regions
* multiple images
* video frames
* the user prompt
* the generated answer

To make multimodal inputs practical, SmolVLM extends context length:

| Model scale | Context setting |
| :---------- | :-------------- |
| 256M / 500M variants | up to about `8k` tokens |
| 2.2B variant | `16k` tokens |

The paper extends context by increasing the RoPE base from `10k` to `273k` and fine-tuning on a long-context mixture.

Main lesson:

> Compact VLMs benefit strongly from longer context, because visual inputs are token-heavy even when the language model is small.

## 8. Pixel Shuffle Token Compression

SmolVLM uses **pixel shuffle**, also called space-to-depth, to reduce the number of visual tokens.

If the vision encoder produces a feature grid:

$$
H \times W \times C
$$

then pixel shuffle with ratio `r` rearranges local spatial neighborhoods into channels:

$$
\frac{H}{r}
\times
\frac{W}{r}
\times
(C r^2)
$$

So the number of visual tokens is reduced by:

$$
r^2
$$

For example:

| Shuffle ratio | Token reduction |
| :------------ | :-------------- |
| `r = 2`       | `4x` fewer tokens |
| `r = 4`       | `16x` fewer tokens |

Large VLMs often use `r = 2` to preserve spatial fidelity. SmolVLM finds that smaller models often benefit from the more aggressive `r = 4`, because attention overhead is a bigger bottleneck.

The tradeoff is:

```text
higher r
-> fewer visual tokens
-> lower memory and faster inference
-> less fine-grained spatial precision
```

For small models, the memory savings can outweigh the spatial loss.

## 9. Image Splitting

High-resolution images are difficult because resizing everything down can destroy important details, especially for:

* OCR
* documents
* tables
* charts
* small objects

SmolVLM uses an image-splitting strategy inspired by UReader and SPHINX:

```text
high-resolution image
-> downsized global image
-> multiple local sub-images
-> encode all regions
-> mark positions
-> pass visual tokens to LM
```

The global image preserves scene-level context. The local sub-images preserve fine details.

This is especially important for small VLMs because they cannot rely on huge language-model capacity to compensate for weak visual evidence.

## 10. Learned Positional Tokens

When images are split into sub-images, the model must know where each crop came from.

The authors initially tried raw string tokens such as:

```text
<row_1_col_2>
```

This caused unstable training in small models, especially for OCR. The paper calls this the "OCR loss plague": the training loss drops, but OCR performance does not improve.

SmolVLM instead uses learned positional tokens for split image locations.

The key finding is:

> Learned positional tokens outperform raw text position tokens for compact VLMs.

The likely reason is that tiny models have limited capacity. They should not have to learn both the syntax and the geometry of position strings from scratch.

## 11. Prompt and Media Formatting

The paper finds that compact VLMs are sensitive to prompt structure.

Three formatting choices help:

1. **System prompts**
   * examples: "You are a useful conversational assistant" or "You are a visual agent and should provide concise answers"

2. **Media intro/outro tokens**
   * mark where image or video content begins and ends
   * reduce confusion in multi-frame video inputs

3. **Completion-only supervised fine-tuning**
   * mask user prompts during SFT
   * train only on assistant completions

This is a small-model lesson: formatting that a larger model might infer implicitly should often be made explicit for compact VLMs.

## 12. Training Recipe

SmolVLM training proceeds in two broad stages:

```text
vision stage
-> image, document, chart, table, OCR, VQA, text reasoning

video stage
-> video captioning, temporal understanding, narrative comprehension
-> multi-image and text data retained
```

### 12.1 Vision Stage

The vision stage uses a mixture derived from prior Idefics-style datasets, with added MathWriting.

It includes:

* document understanding
* captioning
* visual question answering
* multi-image reasoning
* chart understanding
* table understanding
* visual reasoning
* general knowledge Q&A
* text reasoning, math, and coding

The point is to teach visual grounding without destroying language and reasoning ability.

### 12.2 Video Stage

The video fine-tuning stage keeps:

| Data type | Share |
| :-------- | ----: |
| Text      | `14%` |
| Video     | `33%` |

Video data sources include:

* LLaVA-video-178k
* Video-STAR
* Vript
* ShareGPT4Video
* Vista-400k
* MovieChat
* FineVideo

The stage also includes multi-image data from M4-Instruct and Mammoth.

## 13. Data-Mixing Lessons

The paper reports several data lessons that are especially relevant for small multimodal models.

### 13.1 Do Not Blindly Reuse LLM SFT Data

Adding text from the SmolTalk LLM-SFT blend degraded image and video scores.

The paper attributes this to reduced data diversity. For small VLMs, high-quality but narrow text data can crowd out multimodal learning signal.

### 13.2 Use Very Little Chain-of-Thought Data

Small amounts of CoT data help slightly:

```text
about 0.02% to 0.05%
```

But larger CoT proportions hurt performance, especially on image tasks.

The reason is capacity pressure. In a small VLM, too much reasoning-text supervision can compete with visual representation learning.

### 13.3 Moderate Video Duration Helps

Increasing average video duration improves both video and image benchmarks up to about:

```text
3.5 minutes
```

Beyond that, returns diminish relative to the compute cost.

## 14. Evaluation Setup

The paper evaluates SmolVLM with **VLMEvalKit** and compares against efficient open-source VLMs.

Benchmark groups include:

| Group | Benchmarks |
| :---- | :--------- |
| Single-image | OCRBench, AI2D, ChartQA, TextVQA, DocVQA, ScienceQA |
| Multi-task | MMMU, MathVista, MMStar |
| Video | Video-MME, MLVU, MVBench, WorldSense, TempCompass |

The paper argues that parameter count is not enough to judge deployment cost. For VLMs, RAM usage is often a better practical proxy because visual-token handling and architecture choices strongly affect memory.

## 15. Main Results

The headline benchmark averages are:

| Model | Avg score | Batch-1 RAM |
| :---- | --------: | ----------: |
| SmolVLM-256M | `44.0%` | `0.8 GB` |
| SmolVLM-500M | `51.0%` | `1.2 GB` |
| SmolVLM-2.2B | `59.8%` | `4.9 GB` |
| MolmoE-A1B-7B reference | - | `27.7 GB` |

At batch size 64:

| Model | RAM |
| :---- | --: |
| SmolVLM-256M | `15.0 GB` |
| SmolVLM-500M | `16.0 GB` |
| SmolVLM-2.2B | `49.9 GB` |

The most important result is not that SmolVLM wins every benchmark. It does not. The point is that it delivers competitive performance at much lower memory cost.

## 16. Benchmark Highlights

Selected SmolVLM-2.2B results:

| Benchmark | Score |
| :-------- | ----: |
| OCRBench | `72.9%` |
| AI2D | `70.0%` |
| ChartQA | `68.7%` |
| TextVQA | `73.0%` |
| DocVQA | `80.0%` |
| ScienceQA | `89.6%` |
| MMMU | `42.0%` |
| MathVista | `51.5%` |
| MMStar | `46.0%` |
| Video-MME | `52.1%` |
| MLVU | `55.2%` |
| TempCompass | `53.7%` |

Scaling from 256M to 2.2B improves nearly every metric. The paper notes that some difficult reasoning-heavy benchmarks, such as MMMU and AI2D, still benefit from much larger language backbones.

## 17. On-Device and Edge Performance

SmolVLM is explicitly designed for deployment:

* phones
* laptops
* browsers
* small GPUs
* WebGPU environments

The paper reports:

| Hardware / setting | Result |
| :----------------- | :----- |
| A100, SmolVLM-256M, batch 1 | `0.8` examples/sec |
| A100, SmolVLM-256M, batch 64 | `16.3` examples/sec |
| A100, SmolVLM-500M, batch 64 | `9.9` examples/sec |
| L4, SmolVLM-256M | peak `2.7` examples/sec at batch 8 |
| Browser WebGPU, 14-inch MacBook Pro M4 Max | up to `80` decode tokens/sec with 256M |

The authors also release ONNX exports, which matters for practical deployment outside a standard PyTorch server.

## 18. Why SmolVLM Helps SmolVLA

SmolVLA uses **SmolVLM-2** as its perception and language backbone.

The connection is natural:

```text
SmolVLM
-> compact image-language understanding
-> efficient visual tokens
-> low-latency perception backbone

SmolVLA
-> adds robot state
-> adds flow-matching action expert
-> outputs continuous robot actions
```

SmolVLM's design choices are useful for robotics because robot policies are latency-sensitive. Every extra visual token competes with control frequency and deployment cost.

## 19. Comparison to Nearby VLMs

| Model | Main design | Strength | Limitation |
| :---- | :---------- | :------- | :--------- |
| PaliGemma | SigLIP + Gemma | strong compact VLM baseline | less focused on tiny deployment |
| DeepSeek-VL | hybrid SigLIP + SAM encoder | strong high-res detail and OCR | larger memory footprint |
| Gemma 3 | multimodal Gemma family | strong general model family | vision starts at larger scales |
| Qwen2-VL | dynamic high-res visual processing | strong benchmarks | high token and memory cost |
| InternVL2 | strong compact-to-large VLM family | competitive image/video results | more RAM than SmolVLM at similar scale |
| SmolVLM | SigLIP + SmolLM2 + aggressive token efficiency | very low memory for image/video VLM | smaller models still limited on hard reasoning |

SmolVLM's main identity is not benchmark dominance. It is **performance per GB of memory**.

## 20. Limitations

The paper's results also show clear limits:

* the smallest models are still weaker on hard reasoning-heavy benchmarks
* aggressive pixel shuffle can lose fine spatial information
* video understanding is competitive but not uniformly best
* adding generic LLM-SFT data can hurt, so data mixing is fragile
* larger language backbones still matter for MMMU-style and AI2D-style tasks
* efficiency depends on implementation details such as ONNX, WebGPU, and visual preprocessing

The broader limitation is that compact VLMs cannot simply absorb every useful training trick from larger models. They need more selective architecture and data choices.

## 21. Practical Takeaways

The main engineering lessons are:

1. **Optimize memory, not only parameter count**
   * visual-token count and context length can dominate VLM deployment cost

2. **Use a balanced vision-language split**
   * a huge vision encoder can be wasted with a tiny language model

3. **Compress visual tokens aggressively for small models**
   * `r = 4` pixel shuffle can outperform more conservative compression at small scale

4. **Keep high-resolution detail through image splitting**
   * use global context plus local crops rather than one low-res resize

5. **Use learned position tokens for split images**
   * raw text position markers are brittle in tiny VLMs

6. **Be conservative with CoT and LLM-SFT data**
   * small models can be overwhelmed by text-heavy reasoning supervision

7. **Treat video as a token-budget problem**
   * do not average away frames if temporal evidence matters

## 22. Mental Model

The shortest way to remember SmolVLM is:

```text
SigLIP sees the image.
Pixel shuffle makes vision tokens cheap.
Image splitting preserves detail.
SmolLM2 does the multimodal reasoning.
Careful SFT keeps the model small but useful.
```

So SmolVLM is not just a smaller VLM.

It is better understood as:

> a memory-first VLM recipe for image and video understanding on edge devices, small GPUs, browsers, and robotics backbones.
