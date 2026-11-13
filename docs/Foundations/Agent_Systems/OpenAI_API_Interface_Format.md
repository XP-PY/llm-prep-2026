# OpenAI API Interface Format

## 1. Why this note exists

OpenAI APIs are easy to confuse because several layers are mixed together:

* HTTP request format
* model input format
* output object format
* tool calling format
* conversation state format

The clean mental model is:

> The API request sends context and capabilities to the model. The response returns typed output items that the application must parse, execute, or display.

This note focuses on interface shape, not model selection or pricing.

---

## 2. The two main text-generation interfaces

For most agent and assistant systems, there are two formats worth knowing.

## 2.1 Responses API

The Responses API is the newer unified interface.

It is designed around:

* `input`: text, messages, images, files, previous tool outputs, or prior model items
* `instructions`: system-level or developer-level guidance
* `tools`: functions or built-in tools the model may call
* `output`: typed items returned by the model

Use it as the default mental model for new projects.

## 2.2 Chat Completions API

Chat Completions is the older conversation-shaped interface.

It is designed around:

* `messages`: a list of role-tagged messages
* `choices`: one or more generated alternatives
* `message.content`: the assistant text
* `message.tool_calls`: tool requests when function calling is used

It is still useful to understand because many examples, libraries, and older systems use this format.

---

## 3. HTTP envelope

At the transport layer, OpenAI's REST APIs follow the same basic pattern:

```http
POST https://api.openai.com/v1/responses
Content-Type: application/json
Authorization: Bearer $OPENAI_API_KEY
```

The API key is a server-side secret.

Do not expose it in browser code, mobile apps, public notebooks, or committed config files.

A minimal raw request looks like:

```json
{
  "model": "model-id",
  "input": "Explain what an agent loop is in one paragraph."
}
```

The corresponding SDK call usually wraps this HTTP request:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="model-id",
    input="Explain what an agent loop is in one paragraph.",
)

print(response.output_text)
```

The important point:

> The SDK does not change the logical interface. It only gives you a language-native way to build the same request and parse the same response.

---

## 4. Responses API request format

The simplest Responses request has two fields:

```json
{
  "model": "model-id",
  "input": "Summarize RAG in three bullets."
}
```

For more control, split durable instructions from user input:

```json
{
  "model": "model-id",
  "instructions": "Answer concisely. Prefer technical precision over marketing language.",
  "input": "What problem does function calling solve?"
}
```

For multi-turn or multimodal input, `input` can be a list of items:

```json
{
  "model": "model-id",
  "input": [
    {
      "role": "developer",
      "content": "You are writing concise study notes."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Explain the difference between tools and tool calls."
        }
      ]
    }
  ]
}
```

Common input message roles are:

* `developer` or `system`: high-priority instructions
* `user`: the user's task or question
* `assistant`: previous model output included as context

Common content item types include:

* `input_text`
* `input_image`
* `input_file`

So the request shape is:

```text
request
|-- model
|-- instructions       optional global instruction
|-- input              string or item list
|-- tools              optional tool definitions
|-- text.format        optional structured output schema
|-- stream             optional streaming mode
|-- store              optional state storage control
|-- previous_response_id or conversation
```

---

## 5. Responses API output format

A Responses API result is a typed `response` object.

The top-level fields usually include:

* `id`
* `object`
* `created_at`
* `status`
* `model`
* `output`
* `usage`

The important field is `output`.

It is an array of typed items, not just a single string:

```json
{
  "id": "resp_...",
  "object": "response",
  "status": "completed",
  "model": "model-id",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "An agent loop repeatedly decides, acts, observes, and updates state."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 42,
    "output_tokens": 18,
    "total_tokens": 60
  }
}
```

Many SDKs expose a convenience field:

```python
print(response.output_text)
```

That is useful for simple text output, but an agent runtime should usually inspect `response.output` directly because it may contain:

* `message`
* `reasoning`
* `function_call`
* built-in tool calls such as search or file search
* tool call outputs carried forward from earlier turns

The key rule:

> Do not assume `output[0]` is always the final assistant text. Parse by `type`.

---

## 6. Chat Completions format

Chat Completions uses `messages` instead of `input`.

Request:

```json
{
  "model": "model-id",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise technical tutor."
    },
    {
      "role": "user",
      "content": "Explain function calling."
    }
  ]
}
```

Response:

```json
{
  "id": "chatcmpl_...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Function calling lets the model request external actions in a structured format."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 14,
    "total_tokens": 46
  }
}
```

The practical difference:

* Responses: `input` goes in, typed `output` items come back
* Chat Completions: `messages` go in, `choices[*].message` comes back

For new agent systems, Responses maps more naturally to tool calls, reasoning items, and stateful interactions.

---

## 7. Function calling format

Function calling means:

> The model does not directly run your code. It emits a structured request for your application to run code.

In the Responses API, a function tool is usually declared in `tools`:

```json
{
  "model": "model-id",
  "tools": [
    {
      "type": "function",
      "name": "search_notes",
      "description": "Search local study notes by keyword.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query."
          },
          "top_k": {
            "type": ["integer", "null"],
            "description": "Maximum number of results."
          }
        },
        "required": ["query", "top_k"],
        "additionalProperties": false
      },
      "strict": true
    }
  ],
  "input": "Find notes about MCP."
}
```

If the model decides to use the function, the response may contain a tool call item:

```json
{
  "type": "function_call",
  "call_id": "call_123",
  "name": "search_notes",
  "arguments": "{\"query\":\"MCP\",\"top_k\":5}"
}
```

Your application then:

1. parses `arguments`
2. runs `search_notes`
3. appends a `function_call_output`
4. sends the next request

The tool result item looks like:

```json
{
  "type": "function_call_output",
  "call_id": "call_123",
  "output": "{\"matches\":[\"MCP_Protocol.md\"]}"
}
```

The `call_id` is the join key.

Without it, the model cannot reliably connect a tool result to the original tool call.

For Chat Completions, the function schema is commonly wrapped under `function`:

```json
{
  "type": "function",
  "function": {
    "name": "search_notes",
    "description": "Search local study notes by keyword.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string"
        }
      },
      "required": ["query"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

So the most common bug is mixing the two schemas.

---

## 8. Strict schemas and structured outputs

There are two related but different schema uses:

* function schema: constrains tool call arguments
* output schema: constrains the final model answer

For function calling, `strict: true` makes the generated arguments follow the declared schema more reliably.

Strict function schemas should:

* set `additionalProperties: false`
* mark all object properties as `required`
* represent optional values with a nullable type such as `["string", "null"]`

For final JSON output in the Responses API, use `text.format`:

```json
{
  "model": "model-id",
  "input": "Extract the paper title and method name.",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "paper_summary",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "title": {
            "type": "string"
          },
          "method": {
            "type": "string"
          }
        },
        "required": ["title", "method"],
        "additionalProperties": false
      }
    }
  }
}
```

Use this when downstream code expects a stable JSON object instead of free-form prose.

---

## 9. Conversation state

Each request can be treated as stateless unless you deliberately provide state.

There are three common patterns.

## 9.1 Manual history

Send previous user and assistant turns again:

```json
{
  "model": "model-id",
  "input": [
    {
      "role": "user",
      "content": "Explain RAG."
    },
    {
      "role": "assistant",
      "content": "RAG retrieves evidence before generation."
    },
    {
      "role": "user",
      "content": "Now compare it with tool calling."
    }
  ]
}
```

This is explicit and portable, but the application must manage context length.

## 9.2 Previous response chaining

Use `previous_response_id` when you want the platform to link the new turn to an earlier response:

```json
{
  "model": "model-id",
  "previous_response_id": "resp_123",
  "input": "Continue with an example."
}
```

## 9.3 Conversation object

Use a conversation object when the application wants a longer-lived server-side state container.

The main engineering question is:

> Should state live in my application database, or should I delegate part of it to the API?

Manual history gives maximum control. Server-side state reduces request payload management.

---

## 10. Streaming format

Without streaming, the application waits for a complete response object.

With streaming, the API returns incremental events:

```json
{
  "model": "model-id",
  "input": "Draft a short outline.",
  "stream": true
}
```

Streaming changes the application logic:

* render partial text as deltas arrive
* watch for tool call argument deltas
* assemble final output from events
* handle interruption, retry, and partial state carefully

Use streaming for user experience or long-running tool interactions, not because the underlying task is simpler.

---

## 11. Common implementation mistakes

The most common mistakes are:

* treating `response.output` as if it were always plain text
* mixing Responses function schema with Chat Completions function schema
* forgetting to send `function_call_output` with the matching `call_id`
* validating final text as JSON without using a structured output schema
* putting API keys in frontend code
* dropping previous reasoning or tool-call items during multi-step tool loops
* assuming a model never emits multiple output items

A robust parser should switch on `type` and handle unknown future item types defensively.

---

## 12. Practical mental model

Use this mapping:

* `model`: which model should run
* `instructions`: how the model should behave
* `input` or `messages`: what context the model sees
* `tools`: what actions the model may request
* `tool_choice`: whether tool use is forbidden, allowed, required, or constrained
* `text.format`: what shape final text should follow
* `output`: what the model actually produced
* `call_id`: how a tool result links back to a tool request
* `usage`: how many tokens were consumed

The shortest useful summary:

> Responses API is item-based. Chat Completions is message-based. Function calling is a structured request from the model, not the execution of the function itself.

---

## 13. References

Official OpenAI documentation:

* [Responses Overview](https://developers.openai.com/api/reference/responses/overview)
* [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
* [Function calling](https://developers.openai.com/api/docs/guides/function-calling)
* [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
* [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
