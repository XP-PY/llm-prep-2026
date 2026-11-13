# Model Context Protocol (MCP)

## 1. Why MCP exists

As soon as an agent needs to interact with the outside world, a practical problem appears:

* file systems use one interface
* databases use another
* GitHub APIs use another
* cloud services use another
* every model provider exposes tool calling in a slightly different way

Without a protocol layer, integrating tools quickly becomes repetitive and fragile.

You often end up writing:

* one adapter per service
* one schema per model provider
* one error-handling path per integration
* one custom wrapper per project

MCP exists to reduce that integration burden.

Its core goal is:

> provide a standardized way for models or agents to discover and use external tools, resources, and prompts.

A useful mental model is:

> **Function calling teaches the model how to request a tool. MCP standardizes how the tool ecosystem is exposed and accessed.**

---

## 2. The core problem MCP solves

The main pain point is not that tools exist.  
The pain point is that tool access is usually fragmented.

Without MCP, a project often needs to manually solve:

* how tool schemas are described
* how tools are discovered
* how different model vendors represent tool calls
* how local tools and remote tools are launched
* how context objects beyond tools are exposed

MCP gives a common protocol boundary so that:

* tool providers expose capabilities once
* clients connect in a consistent way
* models do not need a custom adapter for every tool source

This is why people often compare MCP to a hardware connector standard such as USB-C:

* it does not define the application logic itself
* it defines the interface and communication contract

---

## 3. MCP is not the same as function calling

This distinction matters a lot.

## 3.1 What function calling does

Function calling is usually a model capability.

It allows the model to:

* decide that a tool should be used
* choose a function name
* generate arguments in a structured form

But function calling alone does **not** standardize:

* how tools are discovered from external servers
* how different tool providers are connected
* how tools, resources, and prompts are exposed under one protocol
* how local and remote tool servers are launched

## 3.2 What MCP adds

MCP is an infrastructure protocol.

It standardizes:

* how a client connects to a server
* how available tools are listed
* how a tool is invoked
* how resources are read
* how prompt templates are exposed

So the practical relationship is:

* **function calling**: model-side ability to request tool use
* **MCP**: protocol-side standard for exposing and accessing capabilities

These are complementary, not competing ideas.

---

## 4. MCP is also not the same as a tool registry

A tool registry is usually an application-level object inside your agent framework.

It may answer:

* what tools are currently available to this agent?
* how are they grouped?
* what descriptions should be shown to the model?

MCP sits lower than that.

It provides a protocol for external capability providers.

A practical stack often looks like:

1. **MCP server**
   exposes tools/resources/prompts
2. **MCP client**
   connects to the server and discovers capabilities
3. **tool registry**
   stores the discovered tools in the application's internal tool system
4. **agent**
   uses the tool registry during reasoning or execution

So:

* MCP is a protocol boundary
* a tool registry is an application-side organization layer

---

## 5. The MCP architecture

MCP is often explained with a three-layer model:

* **Host**
* **Client**
* **Server**

## 5.1 Host

The host is the user-facing environment or application.

Examples:

* a desktop app
* a coding assistant
* an agent runtime
* a chat interface

The host is responsible for:

* receiving user input
* coordinating the model interaction
* deciding when MCP clients should be used

## 5.2 Client

The MCP client is the protocol-side connector.

It is responsible for:

* connecting to MCP servers
* listing available tools
* reading resources
* fetching prompt templates
* calling tools with arguments

The client is where the protocol becomes operational.

## 5.3 Server

The MCP server exposes capabilities to clients.

A server may provide:

* file-system tools
* GitHub search tools
* database access tools
* internal business APIs
* custom domain-specific tools

The server is where the actual business logic or system access lives.

The architecture separates concerns cleanly:

* host focuses on user and model orchestration
* client focuses on protocol communication
* server focuses on capability implementation

---

## 6. The three core MCP capability types

MCP is not only about tools.

A very useful distinction is:

* **Tools**
* **Resources**
* **Prompts**

## 6.1 Tools

Tools are active capabilities.

They do things.

Examples:

* read a file
* write a file
* search GitHub repositories
* query a database
* call a weather API

Use tools when the agent must perform an action or invoke a function.

## 6.2 Resources

Resources are passive data objects.

They provide content rather than actions.

Examples:

* a file exposed as a readable object
* a configuration object
* a document
* a generated view of some stored data

Use resources when the agent needs to consume data but not necessarily trigger an active operation.

## 6.3 Prompts

Prompts are reusable templates or guided prompt objects.

They provide:

* prompt scaffolds
* task-specific instructions
* reusable generation templates

Use prompts when you want a server to ship reusable prompt assets alongside tools and resources.

This distinction is important:

* **tools** do
* **resources** provide
* **prompts** guide

---

## 7. The basic MCP workflow

A typical MCP interaction looks like this:

1. the client connects to a server
2. the client lists available tools, resources, and prompts
3. capability descriptions are converted into model-usable context
4. the model decides whether to use one of them
5. the client executes the requested tool call or resource access
6. the result is returned to the model
7. the model produces a final answer or next action

Two practical observations matter here:

### 7.1 Tool descriptions matter a lot

The model does not "understand the tool" magically.
It relies on:

* the tool name
* the description
* the argument schema

If those are weak, the model will call the wrong tool or provide bad arguments.

### 7.2 Tool discovery is part of the system

Unlike hardcoded local functions, MCP lets the client discover capabilities dynamically.

That means the system can become more extensible:

* add a new MCP server
* connect the client
* discover new tools
* expose them to the agent

without rewriting the entire tool integration stack.

---

## 8. Transport modes

One of MCP's useful properties is transport independence.

The same protocol can be used over different transport layers.

Common patterns include:

## 8.1 Memory transport

Best for:

* tests
* demos
* in-process prototyping

It is simple and lightweight, but not a production-oriented deployment mode.

## 8.2 Stdio transport

Best for:

* local development
* local scripts
* tool servers started as subprocesses

This is a very common developer workflow:

* start a local MCP server process
* communicate through standard input and output

This is especially convenient for:

* local filesystem servers
* Python-based custom servers
* community MCP servers started with `npx`

## 8.3 HTTP transport

Best for:

* remote services
* service-oriented deployments
* production systems

This is more natural when the server is a remote service rather than a local subprocess.

## 8.4 SSE and streamable HTTP

Best for:

* streaming workflows
* long-running interactions
* remote systems where incremental output matters

The practical lesson is:

* local development often starts with **stdio**
* production systems often move toward **HTTP or streaming transports**

---

## 9. Why MCP is useful in agent systems

An agent system benefits from MCP for at least four reasons.

## 9.1 Standardized capability access

The agent does not need a custom hand-written adapter for every service.

## 9.2 Better extensibility

You can connect a new MCP server and let the client discover its capabilities.

## 9.3 Separation of concerns

Tool implementation lives in servers.
Agent orchestration stays in the host and client side.

## 9.4 Better ecosystem reuse

A mature MCP ecosystem means:

* community file-system servers
* GitHub servers
* database servers
* domain-specific servers

can all be integrated through the same protocol boundary.

This is especially valuable when your agent needs many capabilities but you do not want to hand-build every tool wrapper.

---

## 10. MCP in a practical agent stack

In real agent engineering, MCP is usually one layer inside a larger system.

A very practical stack is:

1. **model**
   decides whether to use a capability
2. **agent runtime**
   manages reasoning, control flow, state, and trace
3. **tool registry**
   stores locally available tools
4. **MCP client**
   connects to one or more MCP servers
5. **MCP servers**
   expose tools, resources, and prompts

This means MCP does not replace:

* planning
* memory
* trace management
* evaluation
* fallback logic

It only standardizes one important boundary:

> how external capabilities are exposed and consumed.

---

## 11. MCP and automatic tool expansion

Some agent frameworks wrap an MCP server as a single high-level tool object, then expand it internally into many callable tools.

That means:

* the user adds one MCP integration
* the framework discovers many server tools
* each server tool becomes an agent-callable tool

This is convenient, but it also creates new engineering concerns:

* name collisions
* tool overload
* poor tool selection if descriptions are weak
* too many tools in the model context

So automatic expansion is useful, but should be controlled.

Good practices include:

* prefix tool names by server
* expose only the tools that matter for the current agent
* curate descriptions carefully

---

## 12. When to use MCP, A2A, or ANP

These protocols solve different communication problems.

## 12.1 Use MCP when the problem is tool access

Examples:

* local files
* GitHub
* databases
* custom APIs
* business logic services

If the agent needs to access external capabilities, MCP is the right starting point.

## 12.2 Use A2A when the problem is agent-to-agent collaboration

Examples:

* one agent researches
* one agent writes
* one agent reviews

If the main problem is peer-to-peer collaboration between specialized agents, A2A is a better fit.

## 12.3 Use ANP when the problem is large-scale service discovery

Examples:

* many agents or services
* dynamic capability routing
* network-level discovery

If the main problem is network-wide discovery and coordination, ANP is the better abstraction.

So the shortest distinction is:

* **MCP**: agent-to-tool
* **A2A**: agent-to-agent
* **ANP**: agent-network infrastructure

---

## 13. Custom MCP servers

Using public MCP servers is useful, but many real projects eventually need custom servers.

Common motivations include:

* exposing internal business logic
* safely accessing private data
* wrapping domain-specific workflows
* performance optimization for frequent calls
* integrating proprietary services or hardware

A custom MCP server is a good design when you want:

* a clean protocol boundary
* reusable capability exposure
* tool access that is independent of one particular model vendor

### 13.1 What a custom MCP server should expose well

At minimum:

* a clear server name and purpose
* well-described tools
* precise argument schemas
* explicit error behavior

If the tool is poorly described, the protocol is technically correct but the model may still use it badly.

### 13.2 Good custom server candidates

For your future research agent, strong candidates include:

* local paper corpus search
* citation lookup
* paper metadata retrieval
* note or experiment log access
* benchmark comparison utilities

This is often cleaner than embedding all project logic directly inside one monolithic agent runtime.

---

## 14. MCP design advice for your research agent

Because your project is moving toward a research assistant, MCP becomes useful once your system needs clean external capability boundaries.

### 14.1 What should probably remain local first

At the beginning, keep these inside the agent codebase:

* simple search over your local corpus
* local trace recording
* benchmark and evaluation logic

These parts are still changing quickly, so introducing protocol boundaries too early may create unnecessary complexity.

### 14.2 What is a good first MCP boundary

The first useful MCP server for your project is likely one of:

* **paper search server**
  search local or external paper metadata
* **corpus access server**
  expose document and chunk retrieval
* **project notes server**
  expose structured project memory or logs

These are stable enough to benefit from standardization.

### 14.3 Do not use MCP just because it is fashionable

MCP is valuable when:

* capabilities need to be reused
* capability boundaries need to be explicit
* multiple clients or models may need the same capability

It is not valuable if:

* the capability is tiny and project-local
* the code is still changing too fast
* the protocol boundary adds more complexity than reuse

So the practical rule is:

> add MCP when a capability deserves to become a reusable service boundary, not merely when it exists.

---

## 15. Common MCP failure modes

### 15.1 Treating MCP as if it replaces planning

It does not.
It standardizes access, but the agent still needs:

* planning
* state management
* retry
* reflection
* evaluation

### 15.2 Exposing too many tools at once

More tools do not automatically mean better behavior.

Too many tools create:

* tool confusion
* context bloat
* weaker tool selection

### 15.3 Weak tool descriptions

The protocol may be valid, but the model still calls tools badly if the descriptions are vague.

### 15.4 Using MCP where a plain local function would be enough

Not every utility deserves a protocol boundary.

### 15.5 Ignoring transport and deployment fit

Using the wrong transport for the environment creates unnecessary friction.

For example:

* stdio is often perfect locally
* remote services usually deserve HTTP or streaming transport

---

## 16. A minimal practical MCP blueprint

If you are adding MCP to a real project, a safe progression is:

### Phase 1

* understand host, client, and server roles
* connect to one existing MCP server
* inspect available tools
* call them manually through a client

### Phase 2

* integrate one MCP-backed capability into an agent
* control tool naming and exposure
* log tool calls in trace

### Phase 3

* add a custom MCP server for one stable project capability
* define strong tool descriptions and schemas
* add evaluation for tool-use correctness

### Phase 4

* connect multiple MCP servers
* add routing and capability curation
* combine MCP with memory, retrieval, and agent planning

---

## 17. A few decision rules worth remembering

* MCP does not replace function calling; it complements it.
* MCP is a protocol boundary, not an agent architecture.
* Tools, resources, and prompts are different objects and should stay conceptually distinct.
* Tool descriptions and schemas are part of the system design, not just documentation.
* Use MCP when a capability is stable enough to become a reusable service boundary.
* Do not expose every possible tool to every agent.
* For local prototypes, stdio is often enough; for deployed services, HTTP-based transports become more natural.
