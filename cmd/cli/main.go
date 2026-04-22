package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/tachitachi/homunculus/internal/agent"
	"github.com/tachitachi/homunculus/internal/ollama"
	"github.com/tachitachi/homunculus/internal/tools"
)

func main() {
	baseURL := getenv("OLLAMA_BASE_URL", "http://localhost:11434")
	model := getenv("MODEL_NAME", "gemma4:e4b")
	promptsDir := getenv("PROMPTS_DIR", "./prompts")

	systemPrompt, err := loadPrompt(promptsDir + "/system.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading system prompt: %v\n", err)
		os.Exit(1)
	}

	registry := tools.NewRegistry()
	registry.Register(tools.Calculator{})
	registry.Register(tools.NewWebSearch())

	client := ollama.New(baseURL, model)
	ag := agent.NewReActAgent(client, registry, systemPrompt)
	ag.OnAction = func(tool, input string) {
		fmt.Printf("[Real] Action: %s\n", tool)
		fmt.Printf("[Real] Action Input: %s\n", input)
	}
	ag.OnObservation = func(result string) {
		fmt.Printf("[Real] Observation: %s\n\n", result)
	}

	fmt.Println("Homunculus — Phase 2 (ReAct)")
	fmt.Printf("Model: %s at %s\n", model, baseURL)
	fmt.Println("Type your question and press Enter. Ctrl+C to exit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		fmt.Println()

		if query == "/history" {
			b, err := json.MarshalIndent(ag.Messages(), "", "  ")
			if err != nil {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
			} else {
				fmt.Println(string(b))
			}
			fmt.Println()
			continue
		}

		answer, err := ag.Run(context.Background(), query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
		} else {
			fmt.Printf("\nFinal Answer: %s\n", answer)
		}

		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
		os.Exit(1)
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func loadPrompt(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	return strings.TrimSpace(string(b)), nil
}
