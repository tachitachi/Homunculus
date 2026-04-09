package tools_test

import (
	"context"
	"testing"

	"github.com/tachitachi/homunculus/internal/tools"
)

func TestCalculator_BasicArithmetic(t *testing.T) {
	calc := tools.Calculator{}

	cases := []struct {
		expr string
		want string
	}{
		{"2 + 2", "4"},
		{"10 - 3", "7"},
		{"6 * 7", "42"},
		{"10 / 4", "2.5"},
		{"2 ** 8", "256"},
		{"(1 + 2) * 3", "9"},
		{"3.14159 * 5 * 5", "78.53975"},
	}

	for _, tc := range cases {
		got, err := calc.Run(context.Background(), tc.expr)
		if err != nil {
			t.Errorf("expr %q: unexpected error: %v", tc.expr, err)
			continue
		}
		if got != tc.want {
			t.Errorf("expr %q: got %q, want %q", tc.expr, got, tc.want)
		}
	}
}

func TestCalculator_InvalidExpression(t *testing.T) {
	calc := tools.Calculator{}

	_, err := calc.Run(context.Background(), "not a math expression !!!")
	if err == nil {
		t.Error("expected error for invalid expression, got nil")
	}
}

func TestCalculator_Whitespace(t *testing.T) {
	calc := tools.Calculator{}

	got, err := calc.Run(context.Background(), "  1 + 1  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "2" {
		t.Errorf("got %q, want %q", got, "2")
	}
}
