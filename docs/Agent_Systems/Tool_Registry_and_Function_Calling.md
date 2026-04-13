# Tool Registry and Function Calling

## 1. Why this note exists

Several agent concepts are easy to confuse:

* `tool`
* `tool registry`
* `function calling`
* `MCP`

They are related, but they do **not** solve the same problem.

The clean mental model is:

* a **tool** is one action
* a **tool registry** is the internal directory of available actions
* **function calling** is how the model requests one action in structured form
* **MCP** is how external capabilities are exposed and connected

---

## 2. Tool vs tool registry vs function calling vs MCP

## 2.1 Tool

A tool is one executable capability.

Examples:

* search papers
* query a database
* read a file
* call a calculator
* summarize a document set

A tool answers:

> What can the system do?

## 2.2 Tool registry

A tool registry is an application-side management layer.

It usually stores:

* tool names
* descriptions
* argument schemas
* permission or routing rules
* optional grouping metadata

A tool registry answers:

> Which tools are available to this runtime right now, and how are they represented internally?

## 2.3 Function calling

Function calling is a model-side structured output behavior.

It allows the model to:

* decide that a tool should be used
* choose a tool name
* generate structured arguments

Function calling answers:

> How does the model request one specific tool invocation?

## 2.4 MCP

MCP is a protocol layer for external capability access.

It standardizes how outside systems expose:

* tools
* resources
* prompts

MCP answers:

> How are external capabilities connected and represented in a consistent way?

---

## 3. What a tool object should contain

A useful tool object usually includes:

* `name`
* `description`
* input schema
* output shape or conventions
* execution handler
* optional permission level
* optional tags or groups

For a small project, this may be enough.

For example:

* `search_papers(query, top_k)`
* `lookup_doc(doc_id)`
* `compare_methods(doc_ids, focus)`

The important point is:

> a tool object is not the same thing as the model deciding to call it.

The tool object exists at the system level.  
Function calling happens at the model interaction level.

---

## 4. Why a tool registry exists

As soon as a system has more than a few tools, managing them ad hoc becomes messy.

Without a registry, you often get:

* duplicated tool definitions
* inconsistent descriptions
* hidden routing logic
* poor permission control
* difficulty deciding which tools the model should see

A tool registry helps by centralizing:

* registration
* filtering
* visibility rules
* grouping
* execution metadata

So the registry is not “the tool itself.”

It is the organizational layer around tools.

---

## 5. What function calling actually does

Function calling is often misunderstood as “the tool system.”

It is not.

It is only the mechanism by which the model produces a structured request such as:

* tool name
* arguments

This means function calling is usually responsible for:

* deciding whether to call a tool
* choosing which tool to call
* producing arguments in a machine-readable format

It is **not** responsible for:

* storing the master list of tools
* implementing the tools
* discovering external capability providers
* deciding broader runtime policy by itself

So function calling is a request mechanism, not the whole tool architecture.

---

## 6. Why function calling alone is often enough

Many small projects do **not** need MCP.

Plain function calling is often enough when:

* the tool set is small
* the tools are fixed
* everything is local to the project
* the team controls the full stack
* there is no need for dynamic external capability discovery

Typical examples:

* a research assistant with three local tools
* a coding helper with fixed file and search functions
* a report generator with local retrieval and formatting tools

In these cases, a simple stack works:

1. define local tools
2. register them internally
3. expose them to the model through function calling
4. execute them in application code

That is often the best choice early on.

---

## 7. When a tool registry becomes necessary

You may start with local function calling and still need a registry once:

* tool count grows
* some tools should only be visible in certain tasks
* permission rules differ by tool
* tools need tagging or grouping
* multiple runtimes share overlapping tools

So the practical evolution often looks like:

1. direct tools
2. local function calling
3. internal tool registry
4. optional external protocol layer such as MCP

This is a common progression in real systems.

---

## 8. When MCP becomes worth adding

MCP becomes more attractive when:

* capabilities come from multiple external providers
* you want tools, resources, and prompts under one protocol
* you want capability providers to be reusable across projects
* you want to avoid custom adapters for every integration
* the system is becoming more like a platform than a single app

So the real question is not:

> “Do I have tools?”

The real question is:

> “Has external capability integration itself become a systems problem?”

If yes, MCP starts to make sense.

---

## 9. How these layers work together

A practical runtime can look like this:

1. external capability providers expose services directly or through MCP
2. the application ingests those capabilities into an internal representation
3. the tool registry stores which tools are available
4. the model uses function calling to request one tool
5. the runtime validates, executes, and records the result

This is why the terms should not be collapsed.

They belong to different layers:

* external exposure layer: `MCP`
* internal organization layer: `tool registry`
* model request layer: `function calling`
* execution unit: `tool`

---

## 10. Common failure modes

Bad system design often comes from mixing these layers.

Typical mistakes include:

* treating function calling as the whole tool architecture
* having tools but no registry, then losing control as the tool set grows
* introducing MCP too early for a tiny local project
* forcing everything into tools, even when some objects are really resources or prompts
* exposing too many tools to the model without filtering

The result is usually:

* noisy tool selection
* harder debugging
* poor scaling
* weaker security and permission control

---

## 11. Practical design advice for a research agent

For a research assistant project, a sensible early design is:

* local tools such as:
  * `search_papers`
  * `lookup_doc`
  * `summarize_doc`
* a small internal tool registry
* function calling for structured invocation
* no MCP at first unless external integration pressure appears

Later, MCP becomes more attractive if:

* you want to expose your research corpus as reusable resources
* multiple agents should share the same paper search service
* you want prompt templates and corpus access under one protocol
* the project grows into a reusable platform

So the likely progression is:

* early stage: local tools + registry + function calling
* later stage: optional MCP for reusable externalized capability access

---

## 12. A simple set of decision rules

Keep these rules in mind:

* if you need one action, think **tool**
* if you need an internal directory of actions, think **tool registry**
* if you need the model to request an action in structured form, think **function calling**
* if you need standardized external capability access, think **MCP**

And one more practical rule:

> start with the smallest architecture that gives you control, then add protocol layers only when the integration problem becomes real.
