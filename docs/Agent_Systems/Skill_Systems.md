# Skill Systems

## 1. What a skill is

A **skill** is a reusable task-level capability package for an agent system.

It is not just a single tool and not just a prompt.

A skill usually bundles:

* task instructions
* a preferred workflow
* decision rules
* optional tool usage guidance
* optional reference files, templates, or scripts

So the clean mental model is:

> a tool gives the agent an action, while a skill gives the agent a method.

---

## 2. Why skills exist

As agents become more capable, a new problem appears:

* they may have many tools
* they may have enough model intelligence
* but they still do not reliably know **how to do a class of tasks well**

This is where skills become useful.

A skill helps encode:

* the right order of operations
* the right failure checks
* the right criteria for stopping
* the right way to use tools for a specific task family

Without skills, an agent often behaves like this:

* it has the raw tools
* it can call them
* but it still performs a task inconsistently

With skills, the system can reuse a stronger task recipe.

---

## 3. Skill vs tool

This distinction is fundamental.

## 3.1 Tool

A tool is an executable capability.

Examples:

* search the web
* read a file
* query a database
* run a Python script

A tool answers:

> What action can the agent perform?

## 3.2 Skill

A skill is a reusable task procedure or operating method.

Examples:

* how to summarize an academic paper
* how to triage a mailbox
* how to review a pull request
* how to run an evaluation workflow

A skill answers:

> How should the agent approach this kind of task?

So:

* **tool** = action
* **skill** = method

---

## 4. Skill vs prompt

A prompt is usually one instruction object or template.

A skill is broader than that.

A skill may contain prompts, but it also contains:

* workflow logic
* routing rules
* references
* constraints
* examples
* optional scripts

You can think of it this way:

* a **prompt** is one input formulation
* a **skill** is a reusable operating manual for a task family

---

## 5. Skill vs MCP

These are also different layers.

* **MCP** standardizes how external tools, resources, and prompts are exposed
* **skill** tells the agent how to use available capabilities well for a certain kind of task

So:

* MCP is a protocol layer
* a skill is a task-execution layer

They often work together:

* MCP can provide the tools and resources
* a skill can explain how to combine them into a reliable workflow

---

## 6. What a skill usually contains

A practical skill often includes several parts.

## 6.1 Core instructions

These define:

* what the skill is for
* when to use it
* what success looks like

## 6.2 Workflow guidance

This is often the most important part.

It may define:

* recommended step order
* which tools to prefer first
* what to verify before finishing
* when to stop and ask for clarification

## 6.3 Reference material

A skill may include supporting files such as:

* examples
* templates
* domain notes
* API usage references
* task-specific evaluation rubrics

## 6.4 Optional scripts or assets

Some skills are stronger because they ship:

* helper scripts
* code templates
* reusable config files
* evaluation helpers

This is why a good skill is often more useful than a long static note.

---

## 7. How a skill is used at runtime

A useful skill system usually follows a simple pattern:

1. determine whether the current task matches a skill
2. load the skill instructions
3. extract only the necessary references or scripts
4. execute the workflow using available tools
5. return a result that matches the skill's success criteria

A good runtime does not load every possible skill in full.

It should:

* keep context small
* load only relevant references
* use the skill as a focused operating guide

---

## 8. Why skills matter in agent systems

Skills are especially valuable when:

* the task family repeats
* the workflow is non-trivial
* failure modes are predictable
* the raw tools alone do not guarantee good behavior

This is common in:

* code review
* paper summarization
* inbox triage
* evaluation workflows
* research assistance
* deployment playbooks

In these cases, the real challenge is often not "can the agent call a tool?"

The real challenge is:

> can the agent follow a reliable procedure?

Skills help close that gap.

---

## 9. When to create a skill

Create a skill when most of these are true:

* the task occurs repeatedly
* the task has a recognizable workflow
* the workflow benefits from stable instructions
* tool use alone is not enough
* the task has common mistakes worth preventing

Do **not** create a skill just because a topic is interesting.

A skill should solve a real repeatability problem.

---

## 10. When not to create a skill

A skill is probably unnecessary if:

* the task is one-off
* the task is trivial
* there is no stable workflow
* the task is already well handled by one direct tool call
* the instructions would be shorter than the overhead of maintaining a skill

Over-creating skills can make a system more complex without improving performance.

---

## 11. A good skill design checklist

A strong skill usually has these properties:

* **clear trigger**
  the runtime can tell when the skill should be used
* **clear scope**
  it solves a specific task family, not everything
* **clear workflow**
  it defines a practical step order
* **clear stopping condition**
  it says what "done" means
* **clear failure handling**
  it explains what to do when the workflow breaks
* **small context footprint**
  it does not require loading huge amounts of irrelevant material

---

## 12. Common skill failure modes

Bad skills usually fail in one of these ways:

* **too broad**
  they try to solve an entire domain instead of one task family
* **too vague**
  they describe concepts but not actual workflow steps
* **tool-blind**
  they do not explain how available tools should be used
* **context-heavy**
  they require loading too much material to be practical
* **stale**
  they reference workflows or scripts that no longer match the system

So a useful skill is not just informative. It must remain operational.

---

## 13. Skill design for a research agent

For a research assistant system, likely high-value skills include:

* **paper summarization**
  how to read a paper and produce a grounded summary
* **paper comparison**
  how to compare methods, datasets, assumptions, and limitations
* **evidence-grounded QA**
  how to answer a literature question using only available sources
* **corpus review**
  how to review staged papers before promoting them into the main corpus
* **benchmark case authoring**
  how to write a new gold case that is evaluable and reproducible

Notice that these are not just tools.

They are reusable ways of working.

---

## 14. Skill, memory, and trace

A useful mental connection is:

* **skill** provides the procedure
* **memory** stores long-lived task or user context
* **trace** records what happened during execution

These three should not be confused.

For example:

* the skill says how to do a paper comparison
* memory stores what papers the user already cares about
* trace records which documents were actually read in this run

---

## 15. Practical decision rules

Keep these rules in mind:

* if you need an action, think **tool**
* if you need readable content, think **resource**
* if you need reusable task guidance, think **skill**
* if you need standardized external capability access, think **MCP**

And one final rule:

> do not create a skill because a task is difficult; create a skill because the workflow is reusable.
