package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/Knetic/govaluate"
)

// Calculator evaluates mathematical expressions safely using govaluate.
// The model's input is untrusted text; govaluate's parser limits it to
// arithmetic and logical operators — no arbitrary code execution.
type Calculator struct{}

func (Calculator) Name() string { return "calculator" }

func (Calculator) Description() string {
	return "Evaluate a mathematical expression. Input must be a single-line math " +
		"expression using +, -, *, /, **, (, ), and numeric literals. " +
		`Example: "3.14159 * 5 * 5"`
}

func (Calculator) Run(_ context.Context, input string) (s string, e error) {
	defer func() {
		if r := recover(); r != nil {
			if pe, ok := r.(error); ok {
				e = fmt.Errorf("invalid expression, possibly unknown parameter or invalid syntax: %w", pe)
			}
		}
	}()
	expr, err := govaluate.NewEvaluableExpression(strings.TrimSpace(input))
	if err != nil {
		return "", fmt.Errorf("invalid expression: %w", err)
	}

	result, err := expr.Evaluate(nil)
	if err != nil {
		return "", fmt.Errorf("evaluation failed: %w", err)
	}

	return fmt.Sprintf("%v", result), nil
}
