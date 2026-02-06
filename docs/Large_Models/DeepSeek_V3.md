
|Component|DeepSeek-V2|DeepSeek-V3|Why the Change/Improvement?|
|:---:|:---:|:---:|:---:|
|Total parameters|236B|671B|Larger scale for stronger capabilities|
|Activated per token|21B|37B|More active capacity → better reasoning|
|Attention mechanism|MLA (latent KV compression)|MLA (same, further matured)|Proven KV cache savings|
|MoE design|DeepSeekMoE with auxiliary loss|DeepSeekMoE with **auxiliary-loss-free** balancing|Removes unstable auxiliary loss term|
|Training objective|Standard next-token prediction|**Multi-token prediction** (predict multiple future tokens)|Accelerates learning & improves performance
|Context length|128K|128K (same, but more stable at scale)|—|