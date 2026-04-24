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
	return "Evaluate a mathematical expression and return the result as a number."
}

func (Calculator) InputDescription() string {
	return `A single-line mathematical expression. Supported syntax:

Arithmetic:    +  -  *  /  %  ** (exponentiation — use ** NOT ^)
Bitwise:       &  |  ^  ~  <<  >>   (^ is bitwise XOR, not exponentiation)
Comparison:    ==  !=  >  <  >=  <=  (return 1.0 or 0.0)
Logical:       &&  ||  !
Ternary:       condition ? valueIfTrue : valueIfFalse
Null coalesce: ??
Grouping:      ( )
Arrays/IN:     value IN (a, b, c)

All numeric literals are float64. Strings must be quoted.
IMPORTANT: ^ is bitwise XOR — always use ** for exponentiation.

Examples:
  3.14159 * 5 ** 2
  (100 - 32) * 5 / 9
  2 ** 10`
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
