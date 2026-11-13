# Agent Systems Basics

## 1. Why "tool use" is not automatically an agent

A system is not an agent just because it can call a tool.

The key question is:

> Can it use the current state and feedback to decide what to do next?

That gives a practical boundary:

* **Prompt only**: the model answers directly from the prompt and its parameters.
* **Prompt + Tool**: the model can call a tool, but often only once or in a fixed pattern.
* **RAG**: the system retrieves evidence first, then answers based on that evidence.
* **Agent**: the system performs **multi-step decision making** around a task goal, using tools, observations, and state to keep moving forward.

---

## 2. Minimal agent loop

A minimal agent usually contains:

1. **Task input**
   A user question, instruction, or goal.
2. **Tool layer**
   Search, retrieval, database queries, calculators, file I/O, APIs, and similar capabilities.
3. **State**
   What has already been tried, which documents were read, which sub-goals are done, and whether the system is blocked.
4. **Controller / decision step**
   The logic that decides the next action: search again, refine the query, call another tool, ask a clarification question, or finish.
5. **Termination condition**
   A rule for deciding when the task is complete, blocked, or should return "insufficient evidence".
6. **Trace**
   A structured record of actions, observations, and decisions.

---

## 3. Why trace is a first-class object

Trace is not only a debug log.

It also serves as:

* **working state**
  the system needs to know what has already happened
* **evaluation evidence**
  many failures are only visible in the execution path, not in the final answer
* **future training data**
  high-quality traces can later become trajectory data

Without trace, a system easily repeats the same search, forgets earlier evidence, or makes it impossible to diagnose whether the error came from planning, retrieval, tools, or answer generation.

---

## 4. Retry and reflection

Two common mechanisms are:

* **Retry**
  Run another attempt after a clear failure signal, such as empty retrieval, tool error, or missing evidence.
* **Reflection**
  Inspect the current result and ask what is wrong before acting again.

They are related but not identical:

* retry answers "should I try again?"
* reflection answers "what likely went wrong?"

In practice, a simple agent often starts with retry first, then adds lightweight reflection rules later.

---

## 5. Common failure layers

A useful failure taxonomy for agent systems is:

* **prompt / instruction failure**
  the task was underspecified or the system misunderstood the request
* **planning failure**
  the system chose the wrong next step
* **retrieval failure**
  relevant evidence was not found
* **tool-use failure**
  the wrong tool was called, or it was called with bad arguments
* **evidence selection failure**
  the right document existed, but the wrong part was used
* **answer synthesis failure**
  the evidence was correct, but the final answer misused it

This taxonomy is important because different failures need different fixes. Better prompts do not fix a broken retriever, and better retrieval does not fix evidence misuse in the final answer.

---

## 6. Practical mental model

Use this simple progression:

* **Prompt only**: answer
* **RAG**: retrieve, then answer
* **Prompt + Tool**: call an external function
* **Agent**: decide, act, observe, update state, and continue until done

That is the main shift in Part 4:

> an agent is not defined by having more components, but by using state and feedback to drive a multi-step task loop.
