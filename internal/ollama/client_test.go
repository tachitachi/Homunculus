package ollama_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/tachitachi/homunculus/internal/ollama"
)

// openAIResponse mirrors the subset of /v1/chat/completions we care about.
type openAIResponse struct {
	Choices []struct {
		Message struct {
			Role    string  `json:"role"`
			Content *string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// mockResponse builds a non-streaming OpenAI-compatible response.
func mockResponse(content string) []byte {
	b, _ := json.Marshal(openAIResponse{
		Choices: []struct {
			Message struct {
				Role    string  `json:"role"`
				Content *string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Message: struct {
					Role    string  `json:"role"`
					Content *string `json:"content"`
				}{Role: "assistant", Content: &content},
				FinishReason: "stop",
			},
		},
	})
	return b
}

// mockSSEStream builds an OpenAI SSE stream that delivers content one
// character at a time, matching the format the client must parse.
func mockSSEStream(content string) []byte {
	type delta struct {
		Content string `json:"content"`
	}
	type choice struct {
		Delta        delta  `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	}
	type chunk struct {
		Choices []choice `json:"choices"`
	}

	var sb strings.Builder
	runes := []rune(content)
	for i, ch := range runes {
		c := chunk{Choices: []choice{{Delta: delta{Content: string(ch)}}}}
		if i == len(runes)-1 {
			c.Choices[0].FinishReason = "stop"
		}
		line, _ := json.Marshal(c)
		fmt.Fprintf(&sb, "data: %s\n\n", line)
	}
	sb.WriteString("data: [DONE]\n\n")
	return []byte(sb.String())
}

func TestChat_Success(t *testing.T) {
	want := "Hello from Gemma!"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("expected /v1/chat/completions, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write(mockResponse(want))
	}))
	defer srv.Close()

	client := ollama.New(srv.URL, "test-model")
	msgs := []ollama.Message{{Role: "user", Content: "Say hello."}}

	got, err := client.Chat(context.Background(), msgs, nil)
	if err != nil {
		t.Fatalf("Chat() returned error: %v", err)
	}
	if got != want {
		t.Errorf("Chat() = %q, want %q", got, want)
	}
}

func TestChat_NonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()

	client := ollama.New(srv.URL, "missing-model")
	_, err := client.Chat(context.Background(), []ollama.Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Chat() expected error for non-200 status, got nil")
	}
}

func TestChatStream_Success(t *testing.T) {
	want := "Hello!"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write(mockSSEStream(want))
	}))
	defer srv.Close()

	client := ollama.New(srv.URL, "test-model")
	msgs := []ollama.Message{{Role: "user", Content: "Say hello."}}

	var got strings.Builder
	err := client.ChatStream(context.Background(), msgs, nil, func(chunk string) {
		got.WriteString(chunk)
	})
	if err != nil {
		t.Fatalf("ChatStream() returned error: %v", err)
	}
	if got.String() != want {
		t.Errorf("ChatStream() assembled %q, want %q", got.String(), want)
	}
}

func TestChatStream_CallbackNotCalledOnEmpty(t *testing.T) {
	// Verify the client skips chunks where content is empty or null.
	emptyStr := ""
	type delta struct {
		Content *string `json:"content"`
	}
	type choice struct {
		Delta        delta  `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	}
	type chunk struct {
		Choices []choice `json:"choices"`
	}

	chunks := []chunk{
		{Choices: []choice{{Delta: delta{Content: &emptyStr}}}},
		{Choices: []choice{{Delta: delta{Content: nil}}}},
	}
	realContent := "Hi"
	chunks = append(chunks, chunk{
		Choices: []choice{{
			Delta:        delta{Content: &realContent},
			FinishReason: "stop",
		}},
	})

	var sb strings.Builder
	for _, c := range chunks {
		line, _ := json.Marshal(c)
		fmt.Fprintf(&sb, "data: %s\n\n", line)
	}
	sb.WriteString("data: [DONE]\n\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sb.String()))
	}))
	defer srv.Close()

	client := ollama.New(srv.URL, "test-model")

	calls := 0
	err := client.ChatStream(context.Background(), []ollama.Message{{Role: "user", Content: "hi"}}, nil, func(chunk string) {
		calls++
		if chunk == "" {
			t.Error("callback received empty chunk")
		}
	})
	if err != nil {
		t.Fatalf("ChatStream() error: %v", err)
	}
	if calls != 1 {
		t.Errorf("callback called %d times, want 1", calls)
	}
}

// TestChat_Integration tests against a real OpenAI-compatible server.
// Run with: INTEGRATION=1 OLLAMA_BASE_URL=http://localhost:11434 go test ./internal/ollama/... -run Integration -v
func TestChat_Integration(t *testing.T) {
	if os.Getenv("INTEGRATION") == "" {
		t.Skip("set INTEGRATION=1 to run integration tests")
	}

	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	model := os.Getenv("MODEL_NAME")
	if model == "" {
		model = "gemma4:e4b"
	}

	client := ollama.New(baseURL, model)
	msgs := []ollama.Message{
		{Role: "user", Content: "Reply with exactly three words: one two three"},
	}

	got, err := client.Chat(context.Background(), msgs, &ollama.Options{Temperature: 0.0})
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if got == "" {
		t.Error("Chat() returned empty string")
	}
	t.Logf("model replied: %q", got)
}
