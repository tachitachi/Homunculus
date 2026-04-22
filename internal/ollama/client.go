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

// ToolCallFunction is the function name and arguments the model chose to invoke.
type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolCall is one tool invocation returned inside a model message.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// Message is a single turn in a conversation.
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Options controls model sampling behavior. Zero values are omitted so Ollama
// uses its own defaults.
type Options struct {
	Temperature float64  `json:"temperature,omitempty"`
	NumCtx      int      `json:"num_ctx,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// chatRequest is the body sent to POST /api/chat.
type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  *Options  `json:"options,omitempty"`
	Tools    []Tool    `json:"tools,omitempty"`
}

// chatResponse is one JSON object returned by Ollama — either a streaming
// chunk (Done=false) or the final object (Done=true).
type chatResponse struct {
	Model   string  `json:"model"`
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

// Client talks to an Ollama server over HTTP.
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

// Chat sends messages to Ollama and returns the full assistant reply as a
// single string. It blocks until the model finishes generating.
func (c *Client) Chat(ctx context.Context, messages []Message, opts *Options) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   false,
		Options:  opts,
	})
	if err != nil {
		return "", fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama: unexpected status %d", resp.StatusCode)
	}

	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", fmt.Errorf("ollama: decode response: %w", err)
	}

	return cr.Message.Content, nil
}

// ChatStream sends messages to Ollama and calls fn for each text chunk as it
// arrives. fn is called with the incremental content string; it is never called
// with an empty string. ChatStream returns after the model signals Done.
func (c *Client) ChatStream(ctx context.Context, messages []Message, opts *Options, fn func(chunk string)) error {
	body, err := json.Marshal(chatRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   true,
		Options:  opts,
	})
	if err != nil {
		return fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("ollama: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama: unexpected status %d", resp.StatusCode)
	}

	// Ollama streams newline-delimited JSON objects.
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var cr chatResponse
		if err := json.Unmarshal(line, &cr); err != nil {
			return fmt.Errorf("ollama: decode chunk: %w", err)
		}

		if cr.Message.Content != "" {
			fn(cr.Message.Content)
		}

		if cr.Done {
			break
		}
	}

	return scanner.Err()
}

// ChatWithTools sends messages and tool definitions to Ollama and returns the
// full assistant Message. Text chunks are streamed and passed to onChunk as
// they arrive (pass nil to suppress streaming output). The caller checks
// Message.ToolCalls on the returned Message to determine whether the model
// wants to invoke a tool or has produced a plain-text reply.
//
// Tool calls, if any, may appear before the final done=true chunk (the final
// done=true chunk would have empty content and thinking). Text content
// is accumulated across all chunks and set on the returned Message.Content.
func (c *Client) ChatWithTools(ctx context.Context, messages []Message, tools []Tool, onChunk func(string)) (Message, error) {
	body, err := json.Marshal(chatRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   true,
		Tools:    tools,
	})
	if err != nil {
		return Message{}, fmt.Errorf("ollama: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return Message{}, fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("ollama: do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Message{}, fmt.Errorf("ollama: unexpected status %d", resp.StatusCode)
	}

	var (
		text     strings.Builder
		finalMsg Message
	)
	tool_calls := make([]ToolCall, 0)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var cr chatResponse
		if err := json.Unmarshal(line, &cr); err != nil {
			return Message{}, fmt.Errorf("ollama: decode chunk: %w", err)
		}

		if cr.Message.Thinking != "" {
			if onChunk != nil {
				onChunk(cr.Message.Thinking)
			}
		}

		if cr.Message.Content != "" {
			if onChunk != nil {
				onChunk(cr.Message.Content)
			}
			text.WriteString(cr.Message.Content)
		}

		if cr.Message.ToolCalls != nil {
			tool_calls = append(tool_calls, cr.Message.ToolCalls...)
		}

		if cr.Done {
			// The final chunk carries tool_calls (if any); preserve them and
			// overwrite content with the full accumulated text.
			finalMsg = cr.Message
			finalMsg.Content = text.String()
			finalMsg.ToolCalls = tool_calls
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return Message{}, fmt.Errorf("ollama: scan: %w", err)
	}

	return finalMsg, nil
}
