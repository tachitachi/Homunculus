package tools

import "context"

// Tool is the interface every agent tool must implement.
// Name and Description are injected into the ReAct prompt so the model knows
// what tools are available and how to call them.
type Tool interface {
	// Name returns the identifier the model uses in Action lines, e.g. "calculator".
	Name() string

	// Description returns a ≤50-word summary of what the tool does and what
	// format the input should be in. This is injected verbatim into the prompt.
	Description() string

	// Run executes the tool with the given input string and returns a result
	// string or an error. Errors are returned to the model as observations.
	Run(ctx context.Context, input string) (string, error)
}
