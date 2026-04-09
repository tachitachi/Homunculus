package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/tachitachi/homunculus/internal/ollama"
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

	client := ollama.New(baseURL, model)

	messages := []ollama.Message{
		{Role: "system", Content: systemPrompt},
	}

	fmt.Println("Homunculus — Phase 1")
	fmt.Printf("Model: %s at %s\n", model, baseURL)
	fmt.Println("Type your message and press Enter. Ctrl+C to exit.")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		messages = append(messages, ollama.Message{Role: "user", Content: input})

		fmt.Println()

		var reply strings.Builder
		err := client.ChatStream(context.Background(), messages, nil, func(chunk string) {
			fmt.Print(chunk)
			reply.WriteString(chunk)
		})

		fmt.Print("\n\n")

		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			messages = messages[:len(messages)-1]
			continue
		}

		messages = append(messages, ollama.Message{Role: "assistant", Content: reply.String()})
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
