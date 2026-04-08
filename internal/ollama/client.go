package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Message is a single turn in a conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Options controls model sampling behavior. Zero values are omitted so Ollama
// uses its own defaults.
type Options struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
}

// chatRequest is the body sent to POST /api/chat.
type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  *Options  `json:"options,omitempty"`
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
