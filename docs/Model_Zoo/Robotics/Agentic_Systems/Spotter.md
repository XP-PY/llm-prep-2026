# [Spotter: Let the Embodied Model Lead, and the VLM Reflect for It](https://arxiv.org/abs/2609.36808)

> **Brief:** Spotter adds **parallel failure supervision and physical repair** to a frozen embodied policy. The policy keeps executing while a local screener checks recent chunks; a VLM examines suspicious windows, takes over with a short motion-primitive plan when needed, and checks whether its repair worked. This improves manipulation without putting VLM reasoning before every policy call or updating policy weights.

**Source:** Li et al., 2026; supplied PDF, arXiv:2609.36808v2 dated October 31, 2026. The method, experimental settings, and qualifications below follow this version and Appendices A–C.

## Convenient Links

- [Paper](https://arxiv.org/abs/2609.36808)
- [Official code](https://github.com/zqc3117/Spotter)
- Related systems: [Harness VLA](./Harness_VLA.md), [RoboHarness](./RoboHarness.md), [AdaHVLA](./AdaHVLA.md), [SenseNova-RoboRSI](./SenseNova_RoboRSI.md)
- Policy background: [pi0.5](../Policies/Pi_0_5.md), [Hi Robot](../Policies/Pi_Hi_Robot.md)
- Agent background: [Memory Systems](../../../Agent_Systems/Memory_Systems.md), [Tool Interfaces](../../../Agent_Systems/Tool_Registry_and_Function_Calling.md)

## 1. Why Does Detecting a Failed Grasp Not Solve It?

After missing an object, a policy can continue toward the destination with an empty gripper. Its action sequence still resembles a successful demonstration, but the physical prerequisite for transport is absent. Spotter starts by separating two recovery abilities: **produce a different action that works**, and **recognize which proposed correction will work**.

### 1.1 Retrying Is Usually Insufficient in the Tested Policies

The authors evaluate Cosmos Policy, a world action model, and pi0.5, a VLA, on RoboCasa pick-and-place tasks. An oracle detects a failed grasp, and each method receives up to three correction attempts. Success means lifting the object in any attempt, rather than completing the entire task.

| Correction method | Cosmos Policy | pi0.5 |
|---|---:|---:|
| Restore the pre-failure chunk and resample three seeds | 7.50% | 8.75% |
| Explain the failure through language, up to three rounds | Not supported for arbitrary instructions | 9.69% |
| Recovery fine-tuning, followed by the resampling protocol | 8.13% | 8.44% |

Each setting contains **320 oracle-detected failed grasps**. Resampling uses simulator restoration as a diagnostic experiment; language feedback resumes from the failed state without rewinding. Cosmos Policy accepts only its trained official instructions, so the free-form language-feedback experiment uses pi0.5.

The authors attribute these weak results to successful-demonstration training and limited failure-state coverage. Only about **3% of their ordinary rollouts** produce recovery data meeting their collection requirements. These experiments establish a limitation of the tested checkpoints and recovery procedures, not that all embodied policies are incapable of learning recovery.

### 1.2 More Samples Also Need a Reliable Judge

Using Cosmos Policy, the authors generate **eight candidates per saved state** and execute each for 32 steps. They compare successful/failed candidate pairs from the same state. Pairwise ranking accuracy is 0.5 for random ordering; ties receive half credit. Only states containing both kinds of candidate enter this metric.

The policy's value head and Consistency-Consensus distinguish candidates better at ordinary pre-grasp states than immediately after failure. GeoBoN and RCS remain near random. Even the VLM judge is weak when restricted to nearby images and imagined candidate futures: post-failure accuracy is **53% for GPT and 51% for Qwen**, compared with 57% and 53% at normal states.

The diagnostic covers **991 post-failure states / 10,974 pairs** and **432 normal states / 4,853 pairs**, with bootstrap confidence intervals. Consistency-Exploring additionally reads the candidate's actual executed outcome; it is an explicitly non-deployable diagnostic, not a valid online selector.

This motivates supplying a different source of correction: a VLM that reads **the failed attempt, earlier history, telemetry, and previous repairs**, then acts through explicit motion tools. Spotter does not merely ask the original policy to resample or follow a revised instruction.

## 2. Let the Policy Execute While Supervision Runs Behind It

![Policy-only, VLM-led, and Spotter parallel execution timelines](../../../../assets/Spotter/Spotter_parallel_supervision.png)

*Paper Figure 3. In Spotter, screening and VLM judgment overlap policy execution. A confirmed error triggers repair; routine checks do not require the policy to wait. Block widths are schematic, and the bounded-lead pause is omitted from the illustration.*

### 2.1 Separate the Fast Path from the Repair Path

For instruction $\ell$, RGB images $I_t$, and robot state $q_t$, the frozen policy generates an action chunk:

$$
A_t=\pi_{\theta}(I_t,q_t,\ell),
\qquad \theta\text{ fixed}.
$$

The surrounding system has three roles:

| Component | Receives | Responsibility |
|---|---|---|
| Embodied policy | Current images, proprioception, instruction | Execute ordinary manipulation continuously |
| Local screener $S$ | Completed chunk windows: ordered images and telemetry | Flag possible failure cheaply and with high recall |
| VLM judge $J$ | Suspicious windows, episode history, tools, lessons, repair expectations | Confirm the problem, diagnose, physically repair, and verify |

Qwen3.8-27B is the local screener. The judge is either the same Qwen model or GPT-6 Astra; **Spotter (GPT) still uses Qwen for screening**. These are runtime roles, not jointly trained neural branches or shared-attention modules.

```text
Frozen policy -> execute chunk -> immediately attempt next chunk
                         |
                  record images/state/actions
                         |
               local screener checks pending windows
                  /                         \
           routine window               alarm condition
          policy continues              VLM judges in parallel
                                         /              \
                                       okay          confirmed error
                                                     pause policy
                                                     primitive repair
                                                     return control
                                                     verify next window
```

The supervisor receives deployment observations and commanded-action history, **not object poses or the benchmark success predicate**. The evaluator alone uses privileged state for scoring and episode termination. Depth from the same cameras is available internally to the repair executor, but is not shown to the VLM.

### 2.2 Bound How Far Execution Can Outrun Its Evidence

Let $C_i$ be policy chunk $i$, $z_i$ its recorded observations/telemetry, and $j$ the latest checked chunk. The policy may start chunk $i$ only if

$$
i-j\leq K.
$$

At the lead limit, execution waits at a chunk boundary until supervision catches up. It also pauses for a confirmed error. Thus, parallel execution does not mean unlimited open-loop continuation while a slow judge reasons.

Delayed checks combine **all completed unchecked chunks**, $z_{j+1:i}$, rather than discarding older windows. This matters because a gripper can miss an object and move away before the next observation is interpreted. The lead limit bounds delay in policy chunks; it is not a fixed wall-clock latency or a guarantee against damage.

The screener computes risk $r_i$ and a flag

$$
r_i=S(z_{j+1:i}),
\qquad
f_i=\mathbf{1}\{r_i\geq\tau\;\lor\;E_i\;\lor\;F_i\},
$$

where $E_i$ denotes empty-gripper evidence and $F_i$ a failed/invalid screening call. Defaults use risk on a 0–10 scale with **$\tau=3$**; treating screening failures as flags favors recall.

In the full configuration, two consecutive flagged **checks** trigger the judge. Newly screened empty-gripper evidence can trigger it immediately if there has been no intervention in the previous two checks. The default aperture threshold is **12 mm**, reduced to **4 mm for mug tasks**, because a successful handle grasp can leave only 6–12 mm of opening. This is a task-sensitive heuristic, not a universal grasp detector.

### 2.3 Chunk Size and Screening Frequency Are Different

| Default full-context setting | Cosmos Policy | pi0.5 |
|---|---:|---:|
| Policy output / executed chunk | Executes 16 steps | Predicts 50, executes first 25 |
| Screening interval | Every two chunks, 32 steps | Every chunk, 25 steps |
| Lead limit $K$ | 4 chunks | 2 chunks |
| Maximum lead in policy steps | 64 | 50 |

Every completed chunk is included in screening, but Cosmos does not make a separate screening call after each chunk. The short-context pi0.5 configuration also screens every two chunks. These differences are important when interpreting the simplified timeline and cost comparisons.

## 3. Turn a Diagnosis into a Physical Repair, Then Test It

### 3.1 The Judge Produces an Executable Plan

A decision returns

$$
(v,\delta,e,p)=J(z_{j+1:i},h_i),
\qquad
v\in\{\text{ok},\text{intervene},\text{giveup}\},
$$

where $h_i$ is episode history, $\delta$ a diagnosis, $e$ an observable repair expectation, and $p$ a short sequence from the primitive library. An `ok` verdict leaves policy execution undisturbed. An intervention takes control of execution, rather than handing only a corrective sentence back to the policy.

| Primitive | Effect |
|---|---|
| Move to point | Move to an explicit world target or a pixel target back-projected by the executor |
| Move by offset | Apply a relative translation |
| Lift | Raise the end effector |
| Rotate wrist | Apply a wrist-angle change |
| Set gripper | Open or close the gripper |
| Retreat | Return to a recorded earlier arm pose chosen by the judge |
| Move both arms | Execute synchronized bimanual motion |
| Act | Run the supervised embodied policy for a specified chunk size |
| Hand back | End takeover and resume ordinary policy execution |

Repair executes segment by segment, with a fresh image and execution log after each segment. The judge can continue, decide that the problem is fixed, or give up. A repair is limited to **eight primitives and three plan segments**; full configurations allow at most eight interventions per episode. All executed motion, gripper actions, and controller calibration consume the same task-step budget as policy actions.

For a pixel $(r,c)$, the executor reads **one calibrated depth sample** $d=D[r,c]$. In conventional camera notation, its geometric operation can be summarized as

$$
\mathbf{x}_w
=R_{wc}\!\left(dK_{\mathrm{cam}}^{-1}[c,r,1]^\top\right)
+\mathbf{t}_{wc}+\Delta z\,\mathbf{e}_z.
$$

Here $K_{\mathrm{cam}}$ is camera intrinsics, $(R_{wc},\mathbf{t}_{wc})$ maps camera to world coordinates, and $\Delta z$ is an optional vertical offset. This restates Appendix C's inverse projection; the VLM identifies a pixel while the executor supplies metric geometry. A single sample is used, not a robust median over an object region, so pixel/depth errors can affect contact precision.

### 3.2 Reflection Requires an Observable Expectation

Each intervention states what should be visible afterward, such as a gripper aperture remaining sufficiently open during a lift. After control returns, the **next supervised window receives an observation-only judge check**, even if the screener does not flag it. That check must first decide whether the expectation held and cannot itself initiate another repair.

If the expectation holds, the judge must not repeat the repair. If it fails, a later intervention must change the kind of repair instead of merely trying another offset. This connects the diagnosis to a testable physical consequence and reduces repeated ineffective corrections.

These expectations are model-interpreted observations, not access to ground-truth grasp or task-success flags. Their reliability remains limited by the camera evidence and the judge.

### 3.3 Carrying a Missed Grasp Through the Loop

![Same missed lime grasp followed by policy-only failure or Spotter recovery](../../../../assets/Spotter/Spotter_missed_grasp_repair.png)

*Paper Figure 1. The grasp misses at step 160, and by step 192 the arm has left the pan. Spotter returns the arm to an earlier pose and corrects the lateral alignment; it does not restore the simulator state.*

For the illustrated lime-to-plate task:

1. The policy attempts the grasp and then moves toward the plate with an empty gripper.
2. Screening covers the failed attempt even though execution has advanced. The judge uses the history and fresh repair checkpoint to diagnose the miss.
3. Because the attempt was offset to the left, the repair retreats to an earlier arm pose and shifts the gripper right before reattempting.
4. The policy resumes from the resulting physical state. Subsequent supervision checks the repair's expected effect before further intervention.

**Retreat restores an arm pose, not the scene.** Objects retain their changed positions, and contacts are not undone by a simulator reset. An object still on the table can be re-localized; one dropped out of view or reach may make recovery impossible. Bounded supervision reduces unobserved execution but cannot prevent every irreversible consequence.

## 4. What Is Frozen, Learned, or Retrieved?

### 4.1 Runtime Adaptation Does Not Update Model Weights

Spotter introduces no new policy pretraining or gradient-based post-training. Main simulation experiments use released Cosmos Policy and pi0.5 checkpoints; RoboTwin pi0.5 is trained on clean data only. The real-robot pi0.5 is separately fine-tuned on teleoperated demonstrations for its three tasks.

The recovery fine-tuning in Section 1 is a **diagnostic baseline**, not a Spotter training requirement. The supplied paper reports its outcome but does not give a complete optimizer/data-size recipe for that baseline or for real-robot fine-tuning.

External experience still matters. In RoboCasa, GPT explores separate learning episodes, may reset/retry there, and distills a frozen library of **77 general lessons**. Entries preserve reusable physical advice while removing privileged and scene-specific information. Evaluation uses held-out object instances and disjoint seeds; the library stays read-only, and test episodes do not reset for recovery.

- **“Zero-shot” in the paper:** no retrieved example demonstration, but still access to the frozen general-lesson library.
- **1-shot:** additionally retrieve one episode, up to 12 images, from a separate example library. Retrieval prefers the same policy/task, then the same task, similar tasks, and finally the broader library.

This is experience-based preparation and inference-time retrieval, not evidence-free zero-shot control or online weight learning. The same-policy/task preference also means that 1-shot need not be transfer to a wholly unseen task.

### 4.2 Context and Serving Settings Affect the Comparison

The supervisor sees time-ordered third-person/wrist montages, robot telemetry, action history, remaining budget, empty-grasp events, retrieved lessons, tools, and the previous expectation. Repair decisions additionally receive a fresh checkpoint image.

| Configuration | Short context | Full context / 1-shot |
|---|---|---|
| Recent telemetry | 3 chunks | 40 for Cosmos; 10 for pi0.5 |
| Recent Qwen judge turns retained | 3 | 20 |
| Context restart threshold | 22K tokens | 110K tokens |
| Intervention cap | 3 | 8 |
| Ordinary judge trigger | 1 flagged check | 2 consecutive flagged checks |

Older telemetry/turns are summarized; Qwen restarts with a recap. GPT instead uses server-side conversation continuation. Thus, short-versus-full results change several settings together and do **not** isolate context length alone.

Qwen3.8-27B-FP8 is served with vLLM and prefix caching, one replica on each of four H20 GPUs. Full contexts permit 131,072 tokens and 60 images; short contexts permit 32,768 tokens and 20 images. Judge temperature is 0.2 with at most 3,000 output tokens; screening uses temperature 0 and at most 120, both with thinking disabled. GPT uses medium reasoning effort.

## 5. Does Repair Improve Task Completion Under the Same Budget?

### 5.1 RoboCasa: More Than Extra Steps or a Fixed Retry

The main study uses **24 tasks × 50 episodes = 1,200 episodes per evaluation**, including eight pick-and-place tasks and sixteen mechanism tasks. Spotter and its matched policy-only baseline receive **1.8× the official task-step limit**, with repair steps charged to that limit.

Selected Table 4 overall success rates are:

| Method | Cosmos Policy | pi0.5 |
|---|---:|---:|
| Policy alone, official 1.0× budget | 66.7% | 62.5% |
| Policy alone, matched 1.8× budget | 67.9% | 64.3% |
| Spotter Qwen, short context | 69.3% | 67.1% |
| Spotter Qwen, full context | 71.7% | 68.4% |
| Spotter Qwen, 1-shot | 71.8% | 70.4% |
| Spotter GPT, with screener | 70.0% | 68.8% |
| Spotter GPT, no screener | 72.0% | 70.8% |
| Spotter GPT, 1-shot | **73.5%** | **71.8%** |

The 1-shot GPT gains are **5.6 and 7.5 percentage points over the matched 1.8× baselines**, rather than over official-budget policies. Pick-and-place improves by 10.3 and 9.0 points, versus 3.3 and 6.7 on mechanism tasks. Simply extending the budget gives only 1.2 and 1.8 points. Fixed retreat-and-retry remains close to the original policy even with oracle failure detection, supporting the need to change the repair itself.

The 1-shot effect is not uniform: Qwen/Cosmos moves only 71.7% → 71.8%, because pick-and-place gains offset losses on mechanism tasks. Qwen/pi0.5 improves 68.4% → 70.4%. Removing the GPT screener gains two points on each backbone, showing that screening reduces cost at some loss of supervision effectiveness.

Harness VLA with Qwen reaches 71.3%/69.1% when given the success predicate, and 65.2%/60.5% without it. **This is the paper's rerun**, using 40 interaction rounds and a 5,000-step budget, not the original Harness VLA result or a matched-budget comparison. It supports examining supervision cost and information access; it does not establish a universal ranking across planners and settings.

### 5.2 RoboTwin and Physical Hardware

On **RoboTwin 2.0 Hard**, the main study uses 50 tasks with ten episodes each. At the matched 1.8× budget, pi0.5 achieves **47.2%**, Spotter Qwen 1-shot **51.2%**, and Spotter GPT 1-shot **57.0%**. The larger judge helps more on these longer bimanual tasks. Figure 4's other published models use different source protocols, so those bars are not a controlled compute/budget comparison.

On a **Franka Research 3**, the fine-tuned pi0.5 and Spotter GPT 1-shot are tested for ten trials per task:

| Task | pi0.5 alone | With Spotter |
|---|---:|---:|
| Stack cups | 6/10 | 8/10 |
| Cube into cup | 7/10 | 10/10 |
| Cube into cup, then another cube onto plate | 3/10 | 7/10 |
| Total | 16/30 (53.3%) | 25/30 (83.3%) |

This supports feasibility of physical recovery without simulator rollback. Thirty trials on one robot and three tasks do not establish broad hardware reliability; details such as demonstration counts and real-hardware timing are not fully specified in the PDF.

### 5.3 What the Reported Detection/Correction Metrics Mean

For intervened episodes $I$, let $E\subseteq I$ be those the baseline fails on the same scene, and $C\subseteq E$ those made successful by a single repair. The paper defines

$$
\mathrm{Detection}=\frac{|E|}{|I|},
\qquad
\mathrm{Correction}=\frac{|C|}{|E|}.
$$

“Detection” is an **episode-level intervention precision-like measure**, not recall over all failures or direct accuracy of local error diagnosis. About 85% of Qwen-intervened episodes correspond to baseline failures. With one example, a single repair corrects **13.5% with Qwen and 19.2% with GPT**.

These rates help explain the end-task improvement, but should not be directly equated with Section 1's oracle experiment: that experiment measures object lifting over three attempts at saved failed grasps, while this analysis uses scene-matched episode outcomes after one repair.

## 6. Where Does the Time Go?

Parallelism hides supervision on routine execution, but physical repairs and repeated reasoning still cost time. Table 5 makes the distinction visible. Its Harness VLA rows use Qwen with privileged information and the larger budget described above; scene construction is excluded from timing.

| Backbone and configuration | All episodes | Successful episodes | Failed episodes |
|---|---:|---:|---:|
| Cosmos alone, 1.8× | 63 s | 34 s | 125 s |
| Cosmos + Spotter Qwen, full | 181 s | 47 s | 521 s |
| Cosmos + Harness VLA Qwen | 260 s | 146 s | 545 s |
| pi0.5 alone, 1.8× | 37 s | 23 s | 62 s |
| pi0.5 + Spotter Qwen, full | 126 s | 39 s | 313 s |
| pi0.5 + Harness VLA Qwen | 253 s | 164 s | 451 s |

Successful full-context Spotter episodes add only **13/16 seconds** on the two backbones. The overall averages rise much more because difficult failed episodes involve repeated diagnosis and repair. Outcome-conditioned means also compare each method's own successful/failed set, not necessarily the same episodes.

Qwen full-context supervision hides approximately **36/30 seconds per episode** behind Cosmos/pi0.5 execution, relative to blocking execution for the same supervision. On successful episodes, the judge is called only 0.6 times on average and almost never repairs. In contrast, a VLM-led controller must make planning calls even on routine successful trajectories.

The single example raises average Qwen token usage from **87K to 218K** on Cosmos and **77K to 170K** on pi0.5, although wall time changes little. For GPT/Cosmos, screening reduces tokens from **456K to 156K** relative to judging every supervision window, at a two-point success cost. Token counts include cached input; they are not equivalent to output-token cost.

RoboCasa GPT times are omitted because they include network latency. A **separate supplementary RoboTwin run of 150 episodes** compares all systems under 1.8× limits without privileged information: GPT Spotter and GPT-led control both reach 56%, with successful episodes taking **72 s versus 155 s**. Qwen Spotter reaches 53% in 56 s, versus Qwen-led 42% and policy-only 48% in 42 s. These are distinct from the main 500-episode 57.0% result; Qwen timing there assumes one H200 serving one request at a time.

## 7. What Spotter Adds, and What Remains Unresolved

- **Compared with Harness VLA:** the policy is the default executor, and VLM takeover is exceptional. Parallel screening changes when reasoning blocks physical execution; the VLM still uses a primitive library during repair.
- **Compared with RoboHarness:** Spotter emphasizes bounded asynchronous supervision and actual repair motion, whereas RoboHarness selects context/input/control-flow interventions and consolidates experience. Spotter's evaluated lesson/example libraries remain frozen.
- **Compared with AdaHVLA:** reflection changes the current repair strategy and its expected outcome; it does not revise harness source code between rollouts.

The demonstrated gain comes from connecting **richer temporal evidence → a changed physical action → a checked expectation**, while allowing routine policy execution to proceed. Remaining limits follow from that same design: screening can miss semantic errors, the judge can misread contact state, sparse depth targets can be imprecise, and delayed detection cannot rescue every irreversible action. The short/full and 1-shot comparisons also combine several changes, so the paper does not independently isolate every contribution of history, reflection, and repair scheduling.
