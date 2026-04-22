package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/tachitachi/homunculus/internal/ollama"
	"github.com/tachitachi/homunculus/internal/tools"
)

const defaultMaxIterations = 10

// ReActAgent runs a tool-calling loop against an Ollama model using the
// model's native tool-calling support. It sends the available tools in each
// request and dispatches whatever tool calls the model returns until the model
// produces a plain-text reply (no tool calls), which is treated as the final
// answer.
type ReActAgent struct {
	Client        *ollama.Client
	Registry      *tools.Registry
	MaxIterations int
	SystemPrompt  string

	// OnAction is called each time the model requests a tool call. Optional.
	OnAction func(tool, input string)
	// OnObservation is called with the tool result after execution. Optional.
	OnObservation func(result string)

	// lastMessages holds the message history from the most recent Run call,
	// captured just before each request to the model. Exposed via Messages().
	lastMessages []ollama.Message
}

// NewReActAgent returns a ReActAgent with sensible defaults.
func NewReActAgent(client *ollama.Client, registry *tools.Registry, systemPrompt string) *ReActAgent {
	return &ReActAgent{
		Client:        client,
		Registry:      registry,
		MaxIterations: defaultMaxIterations,
		SystemPrompt:  systemPrompt,
	}
}

// Messages returns the message history from the most recent Run call, as it
// was just before the last request to the model. Returns nil if Run has not
// been called yet.
func (a *ReActAgent) Messages() []ollama.Message {
	return a.lastMessages
}

// Run executes the tool-calling loop for the given query and returns the final
// answer. Each iteration calls the model with the full message history and the
// tool definitions. If the model responds with tool calls they are executed and
// their results appended to the history before the next iteration. When the
// model responds with no tool calls its content is returned as the answer.
func (a *ReActAgent) Run(ctx context.Context, query string) (string, error) {
	ollamaTools := a.buildTools()
	messages := []ollama.Message{
		{Role: "system", Content: a.SystemPrompt},
		{Role: "user", Content: query},
	}

	for i := range a.MaxIterations {
		// Snapshot before the call so Messages() reflects what was sent.
		a.lastMessages = messages

		msg, err := a.Client.ChatWithTools(ctx, messages, ollamaTools, func(chunk string) {
			fmt.Print(chunk)
		})
		fmt.Println() // newline after streamed output
		if err != nil {
			return "", fmt.Errorf("react: model error on iteration %d: %w", i+1, err)
		}

		// No tool calls — the model produced a final text reply.
		if len(msg.ToolCalls) == 0 {
			return msg.Content, nil
		}

		// Append the assistant turn (containing tool_calls) to history.
		messages = append(messages, msg)

		// Execute each tool call and append results as tool-role messages.
		for _, tc := range msg.ToolCalls {
			name := tc.Function.Name
			input := argString(tc.Function.Arguments)

			if a.OnAction != nil {
				a.OnAction(name, input)
			}

			t := a.Registry.Get(name)
			var obs string
			if t == nil {
				obs = fmt.Sprintf("Error: unknown tool %q", name)
			} else {
				result, err := t.Run(ctx, input)
				if err != nil {
					obs = fmt.Sprintf("Error: %v", err)
				} else {
					obs = result
				}
			}

			if a.OnObservation != nil {
				a.OnObservation(obs)
			}

			messages = append(messages, ollama.Message{
				Role:    "tool",
				Content: obs,
			})
		}
	}

	return "", fmt.Errorf("react: reached max iterations (%d) without a final answer", a.MaxIterations)
}

// buildTools converts the registry into Ollama tool definitions.
// Each tool is exposed with a single "input" string parameter.
func (a *ReActAgent) buildTools() []ollama.Tool {
	names := a.Registry.Names()
	out := make([]ollama.Tool, 0, len(names))
	for _, name := range names {
		t := a.Registry.Get(name)
		out = append(out, ollama.Tool{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters: ollama.ToolParameters{
					Type: "object",
					Properties: map[string]ollama.ToolParameterProperty{
						"input": {
							Type:        "string",
							Description: "The input to pass to the tool.",
						},
					},
					Required: []string{"input"},
				},
			},
		})
	}
	return out
}

// argString extracts the "input" key from tool call arguments as a string.
// If the key is absent or not a string, it falls back to JSON-encoding the
// entire arguments map so the tool still receives something meaningful.
func argString(args map[string]any) string {
	if v, ok := args["input"]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	b, _ := json.Marshal(args)
	return string(b)
}
