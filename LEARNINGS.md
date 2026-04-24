# Learnings

This file is a running log of things we discovered, got wrong, corrected, or found surprising while building Homunculus. It is meant to be kept up to date as we progress through each phase — a trail of our actual thought process rather than a polished post-mortem.

---

## Phase 1 — Foundation: Raw HTTP Client + Docker Compose

### The Ollama API is just HTTP + newline-delimited JSON

There is no magic. `POST /api/chat` takes a JSON body and either returns one JSON object (stream=false) or a stream of newline-separated JSON objects (stream=true). Each streaming object has `done: false` until the last one. Building the client against raw `net/http` makes this completely visible — using the Ollama SDK would have hidden all of it.

### Streaming requires scanning line by line, not decoding the whole body

`json.NewDecoder(resp.Body).Decode(&cr)` works for non-streaming responses. For streaming you need `bufio.Scanner` to split on newlines and `json.Unmarshal` each line individually. The Go JSON decoder reads ahead and would consume more of the body than intended.

### Docker health checks need to target the right endpoint

The initial health check on the Ollama container used the wrong path and the container was never considered healthy, causing dependent services to stall. The working check hits `/api/tags`, which returns immediately when Ollama is ready. The lesson: always verify a health check manually (`curl <url>`) before trusting it in Compose.

### GPU passthrough is not automatic — it requires a `deploy:` block

The initial `docker-compose.yml` ran Ollama on CPU. Adding GPU acceleration required a `deploy.resources.reservations.devices` block with `driver: nvidia`. On Windows with WSL2, Docker Desktop picks up NVIDIA drivers automatically; no extra toolkit installation is needed.

### A 5-minute HTTP timeout is not excessive for local LLM inference

`net/http`'s default client has no timeout. We explicitly set 5 minutes on the `httpClient`. This felt long but turned out to be correct — a cold model on first generation (no KV cache) can easily take over a minute even on a GPU.

---

## Phase 2 — ReAct Loop: Text Parsing → Native Tool Calling

### The text-based ReAct loop worked but was fragile

The first implementation of the agent loop had the model output structured text in a specific format (`Thought:` / `Action:` / `Action Input:` / `Final Answer:`), which was then parsed with a custom line-by-line parser. This worked, but had several painful edges:

- The parser had to be case-insensitive (`THOUGHT:`, `Thought:`, `thought:` are all valid)
- Multi-line `Action Input:` required tracking a "currently inside this field" boolean
- The model would sometimes hallucinate `Observation:` lines rather than waiting for the real result — requiring a stop token (`Stop: ["Observation:"]`) to cut generation off at exactly that point
- Malformed responses (model ignoring the format entirely) needed retry logic with a counter

All of this is complexity that existed purely because the tool invocation protocol was embedded in free text.

### Stop tokens are a text-parsing hack that native tool calling eliminates

The stop token `"Observation:"` was added specifically to prevent the model from inventing its own tool results. With native tool calling, the model emits a structured `tool_calls` field instead of text — it never tries to write an observation, so there's nothing to stop.

### Native tool calling requires passing tools as JSON Schema, not as prompt text

The shift from text-based ReAct to native tool calling meant replacing the `{{TOOLS}}` template injection in the system prompt with a `tools` field in the chat request body. Each tool becomes a JSON Schema object:

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": {
        "input": { "type": "string", "description": "..." }
      },
      "required": ["input"]
    }
  }
}
```

Because all our tools take a single string input, wrapping them with a single `input` property was sufficient. More complex tools could expose richer parameter schemas.

### Tool calls appear on the final `done=true` chunk, not during streaming

The initial native tool calling implementation used `stream: false` because it was unclear where tool calls would appear in a streaming response. Turns out: text content arrives on intermediate chunks, and `tool_calls` (if any) appear on the final `done: true` chunk. This means you can stream text content to the user in real time while still receiving the structured tool call at the end — the two aren't mutually exclusive.

The `Thinking` field (for models that expose chain-of-thought) also arrives on intermediate chunks and should be forwarded to the streaming callback separately.

### Tool call arguments need an extraction fallback

The model returns tool arguments as `map[string]any`. Our tools expect a single `string` input, so we extract `args["input"]`. But the model might use a slightly different key name, or pass the argument differently. The fallback of `json.Marshal(args)` and passing the whole JSON blob ensures the tool still gets *something* and can surface a meaningful error rather than silently receiving an empty string.

### Message history must persist across REPL turns

The first version of `Run()` built a fresh `[]Message` slice on every call — only the system prompt and the current query were sent. This meant the model had no memory of previous exchanges in the same session.

Fixing this required moving `messages` from a local variable to a field on `ReActAgent`, initialized with the system prompt in the constructor. Each `Run()` call then appends to the existing history rather than replacing it. The final assistant reply also needs to be appended (it wasn't initially) so the model can refer back to what it said in earlier turns.

### The `/history` command is an invaluable debugging tool

When the model behaves unexpectedly, the single most useful thing to inspect is the exact JSON that was sent to it. Adding `/history` as an in-REPL command that pretty-prints `json.MarshalIndent(ag.Messages(), "", "  ")` immediately made it clear when history was being reset, when tool results were missing from the context, or when the role sequence was wrong.

### Function description and parameter description serve different purposes

Initially the `Tool` interface had a single `Description()` method that was used as the function-level description in the JSON Schema. The parameter's `description` field was left as a generic placeholder ("The input to pass to the tool."), which is useless — models use the parameter description as the primary signal when forming input values.

The right split:

- **Function description** (`Description()`) — answers "what does this tool do?" in one or two sentences. Gives the model enough context to decide *whether* to call the tool.
- **Parameter description** (`InputDescription()`) — answers "what does a valid input look like?" with format rules, constraints, and examples. This is what the model reads when it actually *fills in* the value.

For the calculator, this distinction matters a lot: `^` is bitwise XOR in govaluate, not exponentiation — the correct operator is `**`. Without a precise parameter description spelling this out (and calling it out as `IMPORTANT`), the model will reliably use `^` and get wrong answers. Putting this in the function description alone is not enough; it needs to be right next to the input field where the model is constructing the value.

---

## General — OpenAI-Compatible API

### Prefer `/v1/chat/completions` over Ollama's native `/api/chat`

Ollama exposes two chat APIs: its own `/api/chat` and an OpenAI-compatible `/v1/chat/completions`. The native API is simpler on the surface (NDJSON streaming, options in a nested object) but is Ollama-specific. The OpenAI-compatible endpoint is supported by every serious inference server — Ollama, vLLM, LM Studio, llama.cpp, Together, etc. — making it the right default even when running locally.

### The OpenAI streaming format is SSE, not NDJSON

Ollama's native API streams newline-delimited JSON objects (`{"message":...,"done":false}`). The OpenAI-compatible endpoint uses Server-Sent Events: each line is prefixed with `data: `, and the stream ends with `data: [DONE]`. The scanner must strip the `data: ` prefix before parsing, and skip the sentinel rather than trying to unmarshal it.

### Tool call arguments are a JSON string, not a JSON object

In the OpenAI wire format, `tool_calls[].function.arguments` is a **string containing JSON** (e.g. `"{\"input\":\"2+2\"}"`), not a parsed object. This is intentional — it lets the model produce arguments incrementally as a character stream without the server needing to parse partial JSON. The client must unmarshal the string value to get the actual key/value pairs. Storing arguments as a `string` internally (rather than `map[string]any`) is the simplest approach because it round-trips correctly: the string serializes as a JSON string when sent back in message history, which is exactly what the API expects.

### Tool call arguments stream as incremental string chunks indexed by `index`

In streaming mode, tool calls don't arrive as complete objects on the final chunk the way Ollama's native API does. Instead, each streaming delta may contain a `tool_calls` array where each entry has an `index` field and a partial `function.arguments` string. The client must accumulate these fragments per index and join them before the call can be dispatched. The `id` and `name` arrive on the first delta for a given index; subsequent deltas only carry more argument characters.

When the model issues multiple tool calls in parallel, the chunks from different calls interleave in the stream — the `index` field is the only thing that keeps them separate:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"calculator","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_xyz","function":{"name":"web_search","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\":"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"input\":"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"2+2\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"who won the 2024 election\"}"}}]}}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

The `pending map[int]*partialCall` accumulator in `ChatWithTools` handles this correctly — each index gets its own builder, so interleaved chunks land in the right bucket regardless of arrival order.

**Known improvement:** `react.go` currently executes tool calls sequentially in a `for` loop. When the model returns multiple tool calls they are independent by definition, so they could be dispatched concurrently with `sync.WaitGroup` or `errgroup`. This is worth doing once parallel tool use becomes common.

### Tool result messages require `tool_call_id`

In the OpenAI protocol, a `tool` role message must include a `tool_call_id` matching the `id` from the corresponding tool call. Without it, many servers will reject the request or the model will lose track of which result corresponds to which call. The `id` must be captured from the streamed tool call and threaded through to the result message — it cannot be fabricated.
