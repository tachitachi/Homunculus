package ollama_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/tachitachi/homunculus/internal/ollama"
)

// mockResponse builds the JSON body Ollama returns for a non-streaming chat.
func mockResponse(content string) []byte {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type resp struct {
		Model   string `json:"model"`
		Message msg    `json:"message"`
		Done    bool   `json:"done"`
	}
	b, _ := json.Marshal(resp{
		Model:   "test-model",
		Message: msg{Role: "assistant", Content: content},
		Done:    true,
	})
	return b
}

// mockStreamResponse builds the newline-delimited JSON Ollama returns for a
// streaming chat. It splits content into individual-character chunks to verify
// the client assembles them correctly.
func mockStreamResponse(content string) []byte {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type chunk struct {
		Message msg  `json:"message"`
		Done    bool `json:"done"`
	}

	var sb strings.Builder
	for i, ch := range content {
		c := chunk{
			Message: msg{Role: "assistant", Content: string(ch)},
			Done:    i == len(content)-1,
		}
		line, _ := json.Marshal(c)
		sb.Write(line)
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

func TestChat_Success(t *testing.T) {
	want := "Hello from Gemma!"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/chat" {
			t.Errorf("expected /api/chat, got %s", r.URL.Path)
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
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Write(mockStreamResponse(want))
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
	// Verify the client skips empty content chunks (can happen mid-stream).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Send a chunk with empty content followed by a real one.
		lines := []string{
			`{"message":{"role":"assistant","content":""},"done":false}`,
			`{"message":{"role":"assistant","content":"Hi"},"done":true}`,
		}
		w.Write([]byte(strings.Join(lines, "\n") + "\n"))
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

// TestChat_Integration tests against a real Ollama instance.
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
