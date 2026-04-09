package agent

import (
	"strings"
)

// ResponseType classifies a model response in the ReAct loop.
type ResponseType string

const (
	// TypeAction means the model wants to call a tool.
	TypeAction ResponseType = "action"
	// TypeFinalAnswer means the model has a complete answer.
	TypeFinalAnswer ResponseType = "final"
	// TypeMalformed means the response didn't follow the ReAct format.
	TypeMalformed ResponseType = "malformed"
)

// ParsedResponse is the structured result of parsing one model turn.
type ParsedResponse struct {
	Type ResponseType

	// TypeAction fields
	Tool  string // tool name from "Action:" line
	Input string // raw input from "Action Input:" line

	// TypeFinalAnswer field
	Answer string // text from "Final Answer:" line
}

// ParseResponse parses a single model response text and returns a
// ParsedResponse. It handles multi-line inputs and is case-insensitive on
// the key prefixes. When both Action and Final Answer appear in one response,
// Final Answer takes precedence.
func ParseResponse(text string) ParsedResponse {
	lines := strings.Split(text, "\n")

	var (
		tool        string
		actionInput strings.Builder
		finalAnswer strings.Builder

		inActionInput bool
		inFinalAnswer bool
	)

	for _, raw := range lines {
		line := strings.TrimRight(raw, " \t\r")
		lower := strings.ToLower(line)

		switch {
		case strings.HasPrefix(lower, "final answer:"):
			inActionInput = false
			inFinalAnswer = true
			finalAnswer.WriteString(strings.TrimSpace(line[len("final answer:"):]))

		case strings.HasPrefix(lower, "action input:"):
			inFinalAnswer = false
			inActionInput = true
			actionInput.WriteString(strings.TrimSpace(line[len("action input:"):]))

		case strings.HasPrefix(lower, "action:"):
			inFinalAnswer = false
			inActionInput = false
			tool = strings.TrimSpace(line[len("action:"):])

		case strings.HasPrefix(lower, "thought:"):
			// Thought lines are logged by the caller; nothing to capture here.
			inActionInput = false
			inFinalAnswer = false

		case strings.HasPrefix(lower, "observation:"):
			// Observation lines should come from us, not the model; reset state.
			inActionInput = false
			inFinalAnswer = false

		default:
			// Continuation line — append to whichever multi-line field is active.
			if inActionInput {
				actionInput.WriteByte('\n')
				actionInput.WriteString(line)
			} else if inFinalAnswer {
				finalAnswer.WriteByte('\n')
				finalAnswer.WriteString(line)
			}
		}
	}

	if answer := strings.TrimSpace(finalAnswer.String()); answer != "" {
		return ParsedResponse{Type: TypeFinalAnswer, Answer: answer}
	}

	if tool != "" {
		return ParsedResponse{
			Type:  TypeAction,
			Tool:  tool,
			Input: strings.TrimSpace(actionInput.String()),
		}
	}

	return ParsedResponse{Type: TypeMalformed}
}
