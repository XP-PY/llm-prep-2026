# [Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence](https://arxiv.org/abs/2608.16590)

> **Brief:** Zetta improves a frozen robot policy by evolving **executable runtime critics, recovery playbooks, and tools**. Frequent code-based checks identify physical failures, bounded recoveries restore useful execution states, and rollout evidence drives validation-gated harness revisions. Its companion system, **Z-Infra**, separates simulation workers from model serving to accelerate the rollouts needed for this adaptation.

**Source:** Ding et al., 2026; supplied 47-page PDF, arXiv:2608.16590v1 dated August 17, 2026, with August 18 on the title page. These notes follow its methodology, simulation results, Z-Infra measurements, and Appendices A–D. The broad LIBERO-Pro aggregate is distinguished from the Goal-only headline result below.

## Convenient Links

- [Paper](https://arxiv.org/abs/2608.16590)
- [Project](https://air-embodied-brain.github.io/zetta)
- Related harnesses: [Harness VLA](./Harness_VLA.md), [AdaHVLA](./AdaHVLA.md), [EmbodiedRSI](./EmbodiedRSI.md), [RegenHarness](./RegenHarness.md), [Spotter](./Spotter.md)
- Policies and benchmarks: [pi0.5](../Policies/Pi_0_5.md), [GR00T N1 background](../Policies/GR00T_N1.md), [LIBERO](../Datasets/LIBERO.md), [RoboCasa365](../Datasets/RoboCasa365.md)
- Background: [Agent Memory](../../../Agent_Systems/Memory_Systems.md), [Skill Systems](../../../Agent_Systems/Skill_Systems.md)

## 1. Why Evolve the Checks Along with the Recovery?

A policy can execute a plausible transport motion after dropping the object. A recovery routine helps only if the system recognizes that transport is now invalid, invokes the appropriate correction, and verifies the new grasp before continuing.

Zetta therefore treats **failure detection and recovery as a coupled capability**. A frozen VLA may already use fresh observations, but that alone does not ensure explicit checks of contact, grasp retention, or task completion. Zetta adds executable monitors and learns when their evidence should redirect execution.

The design separates three timescales:

| Loop | What changes? | Purpose |
|---|---|---|
| Within a rollout | Execution mode, physical state, recovery attempts | Detect problems and correct the active episode |
| Between rollout batches | Candidate critic code, recovery logic, tools | Turn recurring failures into proposed improvements |
| Candidate admission and consolidation | Validated skill packages and harness versions | Retain improvements that survive replay and new seeds |

“Self-evolution” concerns **external programs and execution rules**, not online gradient updates to the base policy. Likewise, a runtime critic is a monitoring function, not necessarily an RL value network.

## 2. Runtime Architecture: Evidence Proposes, Adjudication Decides

![Zetta execution, failure analysis, harness repair, and consolidation](../../../../assets/Zetta/Zetta_evolution.png)

*Paper Figure 2. Parallel executions produce reference successes and failures. Diagnosis and repair update the harness, while consolidation packages reusable mechanisms for later rollouts.*

### 2.1 Frozen Models, Evolvable Execution Context

The harness is

$$
\mathcal{H}=(C,R,\mathcal{T}),
$$

where $C$ contains runtime critics, $R$ contains recovery playbooks, and $\mathcal{T}$ contains executable tools. Tools can be selected, adapted, or synthesized during evolution; the design is broader than tuning a fixed list of tool arguments.

Two operators remain fixed: the action policy $\pi_\theta$ and the Orchestrator $\mathcal{A}_{\mathrm{orch}}$. The latter's available evidence and actions change as the harness changes, so a fixed reasoning operator can still produce different decisions.

With task goal $g$, policy observation $o_t$, and a recent execution window $W_t$, a schematic runtime is

$$
a_t=\pi_\theta(o_t,g),\qquad \theta\text{ fixed},
$$

$$
P_t=C(W_t)=\langle e_t,\hat\sigma_t\rangle,
\qquad
\sigma_t=\mathcal{A}_{\mathrm{orch}}(P_t,R,\mathcal{T},K).
$$

$e_t$ is auditable failure evidence, $\hat\sigma_t$ a suggested mode, and $K$ task knowledge such as milestones, constraints, and success conditions. Mode $\sigma_t=0$ uses the nominal policy; positive modes select specialized tools or recoveries.

The critic **does not itself execute a repair or declare success**. The Orchestrator approves interventions, and official task predicates determine episode success. Policy observations and critic evidence are not assumed to be the same information channel.

### 2.2 What a Critic Can Check

The case studies monitor commanded versus realized end-effector motion, gripper aperture, bilateral finger contact, object–gripper drift, grasp retention, transport progress, supported placement, and post-release separation.

```text
Frozen VLA or active recovery executes an action chunk
  -> collect state, motion, contact, and progress evidence
  -> code-based critic proposes continue / intervene / change mode
  -> Orchestrator adjudicates the evidence
       normal -> keep using the VLA
       failure -> bounded recovery/tool execution
  -> check failure clearance and stability
  -> return to the VLA or continue recovery
  -> stop only on official success or the applicable failure/budget limit
```

The paper describes critics as running at action frequency or above policy-inference frequency. It does **not** report a universal critic frequency in Hz or establish hard real-time deadlines. Appendices C/D express the checks at action-chunk boundaries, so this should not be read as evidence that a frontier language model reasons at every motor timestep.

### 2.3 Re-entry Is a Physical Predicate

Returning control immediately after issuing a regrasp command can reproduce the original failure. Zetta specifies a re-entry contract:

$$
\Psi(s_t)=
\mathbf{1}[\operatorname{FailureCleared}(s_t)]
\land
\mathbf{1}[\operatorname{Stability}(s_t)>\gamma].
$$

The first condition resolves the critic's original evidence; the second checks stable contact using force/torque-related signals. The paper does not fully define the stability score, threshold, or required observation duration. The inequality is an abstract acceptance condition, not an instruction to increase raw force or torque oscillations.

This contract supplies the link between **a recovery action being completed** and **the VLA receiving a state from which nominal execution is appropriate**. It is a specified control discipline, not a general hardware-safety proof.

## 3. From Failure Traces to a Revised Harness

### 3.1 Profile Valid Rollouts Before Changing Code

Initial development rollouts establish a policy-only baseline. The system retains append-only traces containing observations/actions, semantic milestone status, and auxiliary physical signals such as contacts and collision intensity.

Infrastructure-invalid attempts, such as simulator crashes or network failures, are rerun with the same logical seed. They are separated from valid policy failures so an infrastructure outage does not become evidence for a grasp-recovery patch.

Successful traces form a milestone-indexed reference collection. For a failed episode, the **first missing milestone** narrows the investigation to the transition that never completed. The **earliest observable divergence** then asks when behavior first departed from the successful reference pattern.

These are different localization tools: “the object never reached the bowl” identifies a missing outcome, whereas “the object separated from the fingers during lifting” identifies an earlier physical event that explains it.

### 3.2 Diagnose a Mechanism, Not Merely the Final Failure

Failures are clustered using divergence context, missing milestones, relative poses, and active tools. A representative medoid seed receives detailed diagnosis, followed by checks that the proposed explanation also fits other cluster members.

The prescribed inspection order is:

1. **Evaluation:** is the success/milestone logic wrong?
2. **Critic:** was the event missed or falsely detected?
3. **State:** are pose/contact representations correct?
4. **Planning/control:** does execution fail despite correct state information?
5. **Recovery:** is the playbook itself causing the problem?
6. **Parameters:** do gains or thresholds need adjustment?

The ordering favors the highest-level effective repair before changing low-level parameters. Diagnosis anchors visual reasoning to one selected camera view; insufficient visual evidence is checked against simulator state and physical signals instead of casually switching viewpoints.

“Causal diagnosis” here is an agent-guided explanation tested through revised rollouts. The paper does not provide a formal causal-identification guarantee, nor complete definitions for the divergence distance and thresholds.

### 3.3 Validate Locally, Then Generalize Across Seeds

The repair agent updates the required subset of critics, recoveries, and tools. Validation first checks whether the revised critic recognizes the original divergence, then runs a **fresh closed-loop episode from the initial state**. Success must satisfy the official goal and the adjudication/re-entry rules; diagnosing a recorded failure correctly is insufficient.

The generalization agent merges local patches into reusable conditions and object-relative behaviors. Its package includes:

| File/directory | Role |
|---|---|
| `SKILL.md` | Milestones, adjudication rules, and re-entry conditions |
| `tools/` | Executable critics and tool implementations |
| `plans/` | Recovery procedures keyed to physical evidence |
| `params/` | Thresholds and numerical operating boundaries |

This packaging is part of Zetta's proposed method; it does not imply that arbitrary generated code becomes a safe motor skill without validation.

The admission protocol requires **100% regression success on the originating failure cluster**, followed by evaluation on separate held-out seeds. The 100% criterion concerns those historical failures, not a promise of perfect held-out performance.

If a held-out failure is used to modify the harness, that seed becomes development data and **new unseen seeds must replace it for final validation**. This distinction is essential: an episode cannot remain a clean test after its outcome has guided a patch.

The optimization can be summarized as

$$
\mathcal{H}_{k+1}=\mathcal{A}_{\mathrm{evo}}
(\mathcal{D}_{\mathrm{fail}}^{(k)},\mathcal{H}_k),
\qquad
\max_{\mathcal{H}}\mathbb{E}[\operatorname{Success}(\tau)
\mid\pi_\theta,\mathcal{A}_{\mathrm{orch}},\mathcal{H}].
$$

The paper's “SGD-like” analogy refers to bounded updates in code/skill space. It does not introduce differentiable gradients through the simulator or optimize the VLA's weights.

## 4. A Worked Grasp–Transport–Placement Example

![Lost grasp, re-approach, generated regrasp, resumed transport, and stable placement](../../../../assets/Zetta/Zetta_recovery_sequence.png)

*Paper Figure 12. Several different recoveries can occur within one episode. An invalid regrasp proposal invokes GraspGen; near-goal risk invokes placement code. Each local repair is checked before execution proceeds.*

Appendices C/D make the critic–recovery connection concrete:

1. **Approach:** if the gripper fails to acquire the object, stop the ineffective motion, open, and restage using current object-relative geometry.
2. **Acquire:** generate a grasp different from remembered failed proposals; execute approach and a short stability/lift check.
3. **Transport:** verify that the object remains retained. A moving empty gripper is not valid progress. Grasp loss triggers a bounded release/restage/regrasp sequence.
4. **Align:** if transport stalls while the grasp is stable, preserve the grasp and switch to fine arm alignment rather than repeatedly issuing coarse motion.
5. **Place:** lower until the required support/containment condition is reached, release, and retreat until separation is verified.

Failed-grasp memory prevents retrying equivalent unsuccessful proposals within the recovery budget. Rebinding the object and destination lets the same physical logic serve related tasks without replaying a source trajectory.

For LIBERO-Pro Goal-T2, the selected internal checkpoints illustrate why the right check matters: **10%** policy-only success rises to **15%** after staging changes, then **95%** after a retained-object critic addresses grasp loss before transport. The paper calls the abrupt improvement an “Aha moment.” These are selected intermediate code versions, not a neural-model training curve or proof of an unbounded scaling law.

The broader case study also distinguishes two kinds of composition: Figure 12 chains repairs **inside one episode**, while Figure 13 adds capabilities **across successive harness revisions**. They should not be counted as the same experimental intervention.

## 5. Z-Infra: Make Repeated Experiments Cheaper

![Z-Infra control plane, environment workers, rollout/model workers, and shared resources](../../../../assets/Zetta/Zetta_Z_Infra.png)

*Paper Figure 3. Environment state and neural inference are managed by separate worker layers. The agent requests an operation without choosing the physical CPU/GPU placement.*

### 5.1 Separate Mutable Environments from Shared Model Serving

The infrastructure exposes sessions, episodes, and steps through a control plane backed by Ray:

| Layer | Main responsibility |
|---|---|
| Control plane / Gateway | Session registry, routing, resource admission, health monitoring |
| Environment workers | Isolated simulation state, stepping, reset, rendering, observations |
| Rollout workers | GPU-resident policy/perception inference, compatible batching, asynchronous responses |

Sessions persist across episodes. `reset` starts an episode, `observe` reads without stepping, and `action_step` executes supplied actions. `policy_infer` requests actions without executing them; `policy_step` combines observe/infer/step; `run_episode` executes a complete worker-side rollout. These APIs serve different control needs rather than forcing all logic through an LLM turn.

Environment workers share read-only MuJoCo model structure between compatible slots while keeping each slot's `mjData` and episode state private. A C++ stepping path combines control, interpolation, physics, and rendering, releasing the Python GIL during its critical section. Resetting one slot does not reset the others.

Model workers batch requests only when model identity, modalities, and tensor shapes are compatible. Bounded channels provide backpressure, and routing tokens deliver results to the correct session. Worker failure marks affected sessions lost; recreation is not a guarantee of restoring a failed episode's exact mid-trajectory state.

### 5.2 Model Partitioning and Quantization Are Separate Optimizations

For PyTorch pi0.5, Z-Infra separates the VLM encoder and action expert into independently scheduled processes, exchanging intermediate activations through CUDA IPC. The paper reports **53% lower average inference latency** and **2.4× goodput under a 200 ms SLO** relative to monolithic serving.

An optional quantization path uses **W8A8 in compute-heavy prefix MLPs**, retaining other modules in BF16. On RTX 4090, the reported speedup is **1.18–1.32×** without success degradation in the tested LIBERO setting. This is not a claim that every module can be quantized equally without affecting control.

These serving improvements accelerate rollout production. They do not themselves constitute a new recovery mechanism, so infrastructure speed and harness success should be evaluated separately.

## 6. Evaluation: Scope, Adaptation, and Final Test Performance

### 6.1 What Is Trained, Evolved, and Tested?

- **LIBERO-Pro:** frozen pi0.5, with 50 development seeds/task excluding seeds 1–20; final evaluation uses seeds 1–20. Development targets the largest failure cluster and continues until its specified development success criterion reaches at least 50%.
- **RoboCasa:** frozen, previously post-trained GR00T N1.5; 50 random development seeds/task and a disjoint set of 50 final test seeds/task.
- **Compute:** the harness experiments report eight RTX 4090 GPUs. The separate infrastructure comparison uses eight A100 GPUs.

Zetta adds no policy pretraining or fine-tuning during evolution. Its learning data are rollout traces and task-specific physical evidence, not additional gradient-training demonstrations for the VLA. The exact checkpoint preparation, agent-model configuration, numerical episode horizons, and total adaptation/repair budgets are not fully specified in the supplied PDF.

**Information access matters:** diagnosis may inspect internal simulator state; the appendix critics use official grasp predicates, finger contact, relative object geometry, and official task-success predicates. These are not established as estimates obtainable from RGB alone. The results therefore should not be directly compared with sensor-only harnesses that withhold object state, contacts, or evaluator feedback.

### 6.2 LIBERO-Pro: Goal-Only and Broader Averages Differ

Table 3 covers **40 task-setting pairs**: ten tasks each in Goal-T, Goal-S, LIBERO-10-T, and LIBERO-10-S. T denotes instruction/target redirection; S denotes object-position swapping. This is not every suite in the complete LIBERO-Pro benchmark.

| Setting | Frozen pi0.5 | Zetta | Gain |
|---|---:|---:|---:|
| Goal-T | 31.0% | 92.5% | 61.5 points |
| Goal-S | 38.0% | 89.0% | 51.0 points |
| LIBERO-10-T | 50.0% | 63.0% | 13.0 points |
| LIBERO-10-S | 9.0% | 40.0% | 31.0 points |
| All 40 settings | 32.00% | **71.13%** | 39.13 points |

The abstract's **90.8%** matches the **Goal-only average**: $(92.5+89.0)/2=90.75\%$. Its corresponding baseline is $(31+38)/2=34.5\%$. This explains the headline numerically; it should not replace the 71.13% aggregate over Goal plus LIBERO-10.

Zetta improves 32 task-setting pairs and matches eight. Some long-task settings remain at zero success, so large Goal-suite gains do not establish that multi-stage execution is solved.

### 6.3 RoboCasa: Eighteen Atomic Tasks

Table 2 evaluates the complete **18-task Atomic-Seen split of RoboCasa 1.0.1**, rather than all RoboCasa365 tasks or its composite-task suite. The macro-average rises from **73.56% to 93.56%**. Selected task results are:

| Task | Frozen GR00T N1.5 | Zetta |
|---|---:|---:|
| PickPlaceSinkToCounter | 58% | 96% |
| PickPlaceDrawerToCounter | 48% | 86% |
| SlideDishwasherRack | 76% | 94% |
| CloseBlenderLid | 50% | 100% |
| All 18 tasks | 73.56% | **93.56%** |

Four cumulative global repair rounds yield **73.56 → 78.71 → 84.85 → 90.54 → 93.56%**. Each checkpoint retains validated earlier capabilities. This supports improvement with the reported experience budget, but does not isolate the contributions of critics, recovery code, added tools, or infrastructure in a comprehensive component ablation.

### 6.4 Cross-Task Transfer Is Zero-Shot with Respect to Harness Evolution

Pregrasp, regrasp, and placement skills evolved on PnP-Stove transfer to related targets without target-specific repair rounds:

| Target task | Parent | Transferred harness |
|---|---:|---:|
| PnP-Sink | 58% | 82% |
| PnP-Cabinet | 62% | 80% |
| PnP-Toaster | 72% | 90% |
| Average | 64% | 84% |

These are transfer checkpoints, rather than the independently evolved per-task harnesses in Table 2. Articulated-interaction skills from TurnOffStove similarly raise the average over three related tasks from **64% to 80%**. LIBERO-Pro examples transfer cumulative skills using matched seeds, policy RNGs, checkpoints, and execution budgets.

“Zero-shot” here means **no additional target-task harness evolution**. It does not establish that the base VLA never saw related tasks during its original training, or that no task goal/context is supplied. The demonstrated transfer is primarily between related physical failure mechanisms.

## 7. Efficiency Results: Compare the Same Load Before Quoting a Speedup

The Z-Infra comparison uses the same model checkpoints and action budgets on **eight A100s**. Figure 15 reports valid rollout throughput:

| Concurrency | Zetta + Z-Infra | Zetta without Z-Infra | RPent |
|---|---:|---:|---:|
| 8 | 12.18 episodes/min | 2.71 | 1.10 |
| 16 | 22.09 episodes/min | 2.88 | 1.72 |
| 32 | 32.8 episodes/min | OOM | OOM |
| 64 | 35.1 episodes/min | OOM | OOM |

At concurrency 16, throughput is **7.7×** the version without Z-Infra and **12.8×** RPent. Episode latency is about **43 s versus 112 s and 513 s**, respectively. At concurrency 64, Z-Infra reaches 35.1 episodes/min while episode latency rises to 95 s; throughput and individual-request latency are different objectives.

The headline approximately 20.6× throughput improvement compares peak reported rates at different supported concurrency levels. Likewise, the abstract's 11.1× “inference” speedup should not be confused with the separate 53% neural-inference latency reduction from partitioning: Section 4.6 primarily measures **episode wall-clock latency** and reports a roughly 12× RPent comparison at concurrency 16.

One implementation boundary remains unclear. Section 2 describes an online multimodal Orchestrator that approves mode transitions, while Section 4.6 attributes low overhead to agents running only in offline reflection/evolution and lightweight critics running online. The PDF does not fully specify how those descriptions are reconciled. It supports a fast code-based monitoring path, but not an assumption that an LLM is invoked on every critic check.

## 8. Relation to Other Harnesses and Evidence Limits

- **Harness VLA:** both compose frozen learned control with specialized tools. Zetta additionally evolves the executable conditions that request recovery and admit a return to the nominal policy.
- **Spotter:** both detect failures during execution. Spotter uses parallel VLM screening/judgment and primitive repairs; Zetta emphasizes generated code critics, validation across seeds, and revisions between rollout batches.
- **AdaHVLA:** both search harness programs rather than policy weights. Zetta focuses its evolvable state on critics, playbooks, and tools, and supplies dedicated rollout infrastructure.
- **RegenHarness:** both separate proposals from verified progress, but their contracts and empirical claims differ. Zetta reports simulation improvement curves; those curves do not validate every proposed governance guarantee.

The main lesson is a connected loop: **localize a recurring physical failure → encode its precursor as a critic → repair the state → verify re-entry → retain the mechanism only after validation**. Faster rollout production lets this search proceed more quickly, while the validation boundary determines whether the resulting code generalizes.

The main quantitative evidence remains simulation-based; the conclusion presents extension to real robots as future work. Simulator-state access, unspecified agent/threshold/budget settings, limited component isolation, and the runtime-adjudication ambiguity constrain reproducibility and hardware conclusions. Neither task success nor the prescribed re-entry checks establish a general guarantee of safe physical execution.
