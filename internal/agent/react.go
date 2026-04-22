package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/tachitachi/homunculus/internal/ollama"
	"github.com/tachitachi/homunculus/internal/tools"
)

const (
	defaultMaxIterations = 10
	maxMalformedRetries  = 3
)

// ReActAgent runs the Reasoning + Acting loop against an Ollama model.
// It keeps calling the model, dispatching tool calls, and appending
// observations until the model emits a Final Answer or MaxIterations is hit.
type ReActAgent struct {
	Client        *ollama.Client
	Registry      *tools.Registry
	MaxIterations int
	SystemPrompt  string

	// OnThought is called each time a Thought line is printed. Optional.
	// OnAction / OnObservation follow the same pattern.
	OnThought     func(thought string)
	OnAction      func(tool, input string)
	OnObservation func(result string)
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

// Run executes the ReAct loop for the given query and returns the final answer.
// Each iteration calls the model, parses the response, and either dispatches a
// tool or returns the answer. Malformed responses are retried up to
// maxMalformedRetries times per iteration before giving up.
func (a *ReActAgent) Run(ctx context.Context, query string) (string, error) {
	messages := []ollama.Message{
		{Role: "system", Content: a.SystemPrompt},
		{Role: "user", Content: query},
	}

	// Stop generation the moment the model writes "\nObservation:" so it cannot
	// hallucinate the tool result. The real observation is injected by this loop.
	opts := &ollama.Options{
		Stop: []string{"Observation:"},
	}

	malformed := 0

	for i := range a.MaxIterations {
		var sb strings.Builder
		err := a.Client.ChatStream(ctx, messages, opts, func(chunk string) {
			fmt.Print(chunk)
			sb.WriteString(chunk)
		})
		fmt.Println() // newline after streamed output
		if err != nil {
			return "", fmt.Errorf("react: model error on iteration %d: %w", i+1, err)
		}

		response := sb.String()
		a.logThoughts(response)

		parsed := ParseResponse(response)

		switch parsed.Type {
		case TypeFinalAnswer:
			return parsed.Answer, nil

		case TypeAction:
			malformed = 0
			tool := a.Registry.Get(parsed.Tool)
			if tool == nil {
				obs := fmt.Sprintf("Error: unknown tool %q. Available tools: %s",
					parsed.Tool, a.availableToolNames())
				messages = a.appendObservation(messages, response, obs)
				if a.OnObservation != nil {
					a.OnObservation(obs)
				}
				continue
			}

			if a.OnAction != nil {
				a.OnAction(parsed.Tool, parsed.Input)
			}

			result, err := tool.Run(ctx, parsed.Input)
			var obs string
			if err != nil {
				obs = fmt.Sprintf("Error: %v", err)
			} else {
				obs = result
			}

			if a.OnObservation != nil {
				a.OnObservation(obs)
			}

			messages = a.appendObservation(messages, response, obs)

		case TypeMalformed:
			malformed++
			if malformed > maxMalformedRetries {
				return "", fmt.Errorf("react: model produced %d consecutive malformed responses; giving up", malformed)
			}
			obs := "Error: your response did not follow the required format. " +
				"You must use Thought/Action/Action Input or Thought/Final Answer."
			messages = a.appendObservation(messages, response, obs)
			if a.OnObservation != nil {
				a.OnObservation(obs)
			}
		}
	}

	return "", fmt.Errorf("react: reached max iterations (%d) without a final answer", a.MaxIterations)
}

// appendObservation adds the assistant turn and an observation user turn to
// the message history.
func (a *ReActAgent) appendObservation(messages []ollama.Message, assistantText, observation string) []ollama.Message {
	messages = append(messages, ollama.Message{Role: "assistant", Content: assistantText})
	messages = append(messages, ollama.Message{Role: "user", Content: "Observation: " + observation})
	return messages
}

// logThoughts scans the response for Thought: lines and calls OnThought.
func (a *ReActAgent) logThoughts(response string) {
	if a.OnThought == nil {
		return
	}
	for line := range strings.SplitSeq(response, "\n") {
		lower := strings.ToLower(line)
		if strings.HasPrefix(lower, "thought:") {
			a.OnThought(strings.TrimSpace(line[len("thought:"):]))
		}
	}
}

func (a *ReActAgent) availableToolNames() string {
	return a.Registry.Descriptions()
}
