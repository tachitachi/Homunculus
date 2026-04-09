package agent_test

import (
	"testing"

	"github.com/tachitachi/homunculus/internal/agent"
)

func TestParseResponse_FinalAnswer(t *testing.T) {
	text := `Thought: I know this one.
Final Answer: The speed of light is approximately 299,792,458 m/s.`

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeFinalAnswer {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeFinalAnswer)
	}
	want := "The speed of light is approximately 299,792,458 m/s."
	if got.Answer != want {
		t.Errorf("Answer = %q, want %q", got.Answer, want)
	}
}

func TestParseResponse_Action(t *testing.T) {
	text := `Thought: I need to multiply these numbers.
Action: calculator
Action Input: 1337 * 42`

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeAction {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeAction)
	}
	if got.Tool != "calculator" {
		t.Errorf("Tool = %q, want %q", got.Tool, "calculator")
	}
	if got.Input != "1337 * 42" {
		t.Errorf("Input = %q, want %q", got.Input, "1337 * 42")
	}
}

func TestParseResponse_Malformed(t *testing.T) {
	text := "I'm just going to answer you directly without following the format."

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeMalformed {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeMalformed)
	}
}

func TestParseResponse_FinalAnswerWinsOverAction(t *testing.T) {
	// If both appear, Final Answer takes precedence.
	text := `Thought: Let me reconsider.
Action: calculator
Action Input: 2 + 2
Final Answer: The answer is 4.`

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeFinalAnswer {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeFinalAnswer)
	}
}

func TestParseResponse_MultiLineActionInput(t *testing.T) {
	text := `Thought: I need to run some code.
Action: code_runner
Action Input: line one
line two
line three`

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeAction {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeAction)
	}
	if got.Tool != "code_runner" {
		t.Errorf("Tool = %q, want %q", got.Tool, "code_runner")
	}
	wantInput := "line one\nline two\nline three"
	if got.Input != wantInput {
		t.Errorf("Input = %q, want %q", got.Input, wantInput)
	}
}

func TestParseResponse_CaseInsensitiveKeys(t *testing.T) {
	text := `THOUGHT: checking something
FINAL ANSWER: done`

	got := agent.ParseResponse(text)
	if got.Type != agent.TypeFinalAnswer {
		t.Fatalf("Type = %q, want %q", got.Type, agent.TypeFinalAnswer)
	}
	if got.Answer != "done" {
		t.Errorf("Answer = %q, want %q", got.Answer, "done")
	}
}
