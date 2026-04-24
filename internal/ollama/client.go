package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ToolParameterProperty describes one property in a function's parameter schema.
type ToolParameterProperty struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// ToolParameters is the JSON Schema parameters block for a tool function.
type ToolParameters struct {
	Type       string                           `json:"type"`
	Properties map[string]ToolParameterProperty `json:"properties"`
	Required   []string                         `json:"required,omitempty"`
}

// ToolFunction describes the callable function a tool exposes.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  ToolParameters `json:"parameters"`
}

// Tool is the top-level tool definition passed in the tools field of a chat request.
type Tool struct {
	Type     string       `json:"type"` // always "function"
	Function ToolFunction `json:"function"`
}

// ToolCallFunction holds the name and arguments of a model-invoked function.
// Arguments is the raw JSON string exactly as the model produced it
// (e.g. `{"input":"2+2"}`). Callers unmarshal it as needed.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall is one tool invocation returned inside a model message.
type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Function ToolCallFunction `json:"function"`
}

// Message is a single turn in a conversation.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Options controls model sampling behavior. Zero values are omitted so the
// server uses its own defaults.
type Options struct {
	Temperature float64  `json:"temperature,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// chatRequest is the body sent to POST /v1/chat/completions.
type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Tools       []Tool    `json:"tools,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

func newChatRequest(model string, messages []Message, stream bool, tools []Tool, opts *Options) chatRequest {
	r := chatRequest{
		Model:    model,
		Messages: messages,
		Stream:   stream,
		Tools:    tools,
	}
	if opts != nil {
		r.Temperature = opts.Temperature
		r.MaxTokens = opts.MaxTokens
		r.TopP = opts.TopP
		r.Stop = opts.Stop
	}
	return r
}

// openAIToolCallFunc is the function sub-object inside an OpenAI tool call.
// Arguments is a JSON string on the wire.
type openAIToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// openAIToolCall is a single tool call in an OpenAI streaming delta or message.
type openAIToolCall struct {
	Index    int                `json:"index"`
	ID       string             `json:"id"`
	Function openAIToolCallFunc `json:"function"`
}

// openAIDelta is the incremental content of a streaming chunk.
type openAIDelta struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"` // pointer: null vs "" are different
	Reasoning *string          `json:"reasoning"`
	ToolCalls []openAIToolCall `json:"tool_calls"`
}

// openAIMessage is the full message in a non-streaming response.
type openAIMessage struct {
	Role      string           `json:"role"`
	Content   *string          `json:"content"`
	Reasoning *string          `json:"reasoning"`
	ToolCalls []openAIToolCall `json:"tool_calls"`
}

// openAIChoice is one choice in a response object.
type openAIChoice struct {
	Message      openAIMessage `json:"message"` // non-streaming
	Delta        openAIDelta   `json:"delta"`   // streaming
	FinishReason string        `json:"finish_reason"`
}

// openAIResponse is the top-level object returned by /v1/chat/completions.
type openAIResponse struct {
	Choices []openAIChoice `json:"choices"`
}

// Client talks to an OpenAI-compatible chat completions server over HTTP.
// Use New to construct one.
type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

// New returns a Client pointed at baseURL using model as the default model.
// baseURL should be e.g. "http://localhost:11434" (no trailing slash).
func New(baseURL, model string) *Client {
	return &Client{
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute, // model inference can be slow
		},
	}
}

func (c *Client) post(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: do request: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("ollama: unexpected status %d", resp.StatusCode)
	}
	return resp, nil
}

// Chat sends messages and returns the full assistant reply as a single string.
// It blocks until the model finishes generating.
func (c *Client) Chat(ctx context.Context, messages []Message, opts *Options) (string, error) {
	body, err := json.Marshal(newChatRequest(c.model, messages, false, nil, opts))
	if err != nil {
		return "", fmt.Errorf("ollama: marshal request: %w", err)
	}

	resp, err := c.post(ctx, body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var r openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return "", fmt.Errorf("ollama: decode response: %w", err)
	}
	if len(r.Choices) == 0 {
		return "", fmt.Errorf("ollama: response contained no choices")
	}
	if r.Choices[0].Message.Content == nil {
		return "", nil
	}
	return *r.Choices[0].Message.Content, nil
}

// ChatStream sends messages and calls fn for each text chunk as it arrives.
// fn is never called with an empty string. ChatStream returns after the model
// signals it is done via the SSE [DONE] sentinel.
func (c *Client) ChatStream(ctx context.Context, messages []Message, opts *Options, fn func(chunk string)) error {
	body, err := json.Marshal(newChatRequest(c.model, messages, true, nil, opts))
	if err != nil {
		return fmt.Errorf("ollama: marshal request: %w", err)
	}

	resp, err := c.post(ctx, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		data, done, ok := parseSSELine(line)
		if done {
			break
		}
		if !ok {
			continue
		}

		var chunk openAIResponse
		if err := json.Unmarshal(data, &chunk); err != nil {
			return fmt.Errorf("ollama: decode chunk: %w", err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		if r := chunk.Choices[0].Delta.Reasoning; r != nil && *r != "" {
			fn(*r)
		}
		if c := chunk.Choices[0].Delta.Content; c != nil && *c != "" {
			fn(*c)
		}
		if chunk.Choices[0].FinishReason != "" {
			break
		}
	}

	return scanner.Err()
}

// ChatWithTools sends messages and tool definitions and returns the full
// assistant Message. Text chunks are streamed and passed to onChunk as they
// arrive (pass nil to suppress streaming output). The caller checks
// Message.ToolCalls to determine whether the model wants to invoke a tool or
// has produced a plain-text reply.
//
// In the OpenAI streaming protocol, tool call arguments arrive as incremental
// string chunks keyed by index. ChatWithTools assembles them and returns the
// complete ToolCalls slice on the returned Message.
func (c *Client) ChatWithTools(ctx context.Context, messages []Message, tools []Tool, onChunk func(string)) (Message, error) {
	body, err := json.Marshal(newChatRequest(c.model, messages, true, tools, nil))
	if err != nil {
		return Message{}, fmt.Errorf("ollama: marshal request: %w", err)
	}

	resp, err := c.post(ctx, body)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

	// partialCall accumulates a single streamed tool call.
	type partialCall struct {
		id   string
		name string
		args strings.Builder
	}
	pending := map[int]*partialCall{}
	var text strings.Builder

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		data, done, ok := parseSSELine(line)
		if done {
			break
		}
		if !ok {
			continue
		}

		var chunk openAIResponse
		if err := json.Unmarshal(data, &chunk); err != nil {
			return Message{}, fmt.Errorf("ollama: decode chunk: %w", err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		if delta.Content != nil && *delta.Content != "" {
			if onChunk != nil {
				onChunk(*delta.Content)
			}
			text.WriteString(*delta.Content)
		}

		if delta.Reasoning != nil && *delta.Reasoning != "" {
			if onChunk != nil {
				onChunk(*delta.Reasoning)
			}
		}

		for _, tc := range delta.ToolCalls {
			p, ok := pending[tc.Index]
			if !ok {
				p = &partialCall{}
				pending[tc.Index] = p
			}
			if tc.ID != "" {
				p.id = tc.ID
			}
			if tc.Function.Name != "" {
				p.name = tc.Function.Name
			}
			p.args.WriteString(tc.Function.Arguments)
		}

		if chunk.Choices[0].FinishReason != "" {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return Message{}, fmt.Errorf("ollama: scan: %w", err)
	}

	msg := Message{
		Role:    "assistant",
		Content: text.String(),
	}
	for i := range len(pending) {
		p := pending[i]
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID: p.id,
			Function: ToolCallFunction{
				Name:      p.name,
				Arguments: p.args.String(),
			},
		})
	}

	return msg, nil
}

// parseSSELine parses one line from an OpenAI SSE stream.
// Returns (data, done=true, _) for the [DONE] sentinel,
// (data, false, true) for a data line, or (nil, false, false) to skip.
func parseSSELine(line string) (data []byte, done bool, ok bool) {
	if line == "data: [DONE]" {
		return nil, true, false
	}
	if !strings.HasPrefix(line, "data: ") {
		return nil, false, false
	}
	return []byte(line[6:]), false, true
}
