package tools

import "context"

// Tool is the interface every agent tool must implement.
type Tool interface {
	// Name returns the unique identifier used to invoke this tool.
	Name() string

	// Description returns a brief summary of what the tool does.
	// This is used as the function-level description in the tool schema and
	// should answer "what does this tool do?" in one or two sentences.
	Description() string

	// InputDescription returns a precise specification of what a valid input
	// looks like: format, constraints, supported syntax, and at least one
	// example. This is used as the parameter-level description in the tool
	// schema and is the primary signal the model uses when forming its input.
	InputDescription() string

	// Run executes the tool with the given input string and returns a result
	// string or an error. Errors are surfaced to the model as observations.
	Run(ctx context.Context, input string) (string, error)
}
