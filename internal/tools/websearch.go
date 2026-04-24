package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// WebSearch queries DuckDuckGo's instant answer API. No API key required.
// Returns up to 3 results formatted as title + URL + snippet.
type WebSearch struct {
	httpClient *http.Client
}

// NewWebSearch returns a WebSearch tool with a sensible HTTP timeout.
func NewWebSearch() *WebSearch {
	return &WebSearch{
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

func (WebSearch) Name() string { return "web_search" }

func (WebSearch) Description() string {
	return "Search the web for current information. Returns up to 3 results with title, URL, and a short snippet."
}

func (WebSearch) InputDescription() string {
	return `A plain-English search query, the same as you would type into a search engine.

Guidelines:
  - Use specific keywords rather than full sentences
  - For factual lookups include relevant context (e.g. "Eiffel Tower height meters")
  - For current events include a year if freshness matters (e.g. "Go 1.24 release notes 2025")
  - Do not use boolean operators or special syntax — plain keywords only

Examples:
  capital of Japan
  Golang context package usage
  Docker compose healthcheck syntax`
}

// ddgResponse is the subset of the DuckDuckGo instant answer JSON we care about.
type ddgResponse struct {
	AbstractText   string     `json:"AbstractText"`
	AbstractURL    string     `json:"AbstractURL"`
	AbstractSource string     `json:"AbstractSource"`
	RelatedTopics  []ddgTopic `json:"RelatedTopics"`
}

type ddgTopic struct {
	Text     string     `json:"Text"`
	FirstURL string     `json:"FirstURL"`
	Topics   []ddgTopic `json:"Topics"` // nested topic groups
}

func (w *WebSearch) Run(ctx context.Context, input string) (string, error) {
	query := strings.TrimSpace(input)
	if query == "" {
		return "", fmt.Errorf("web_search: empty query")
	}

	endpoint := "https://api.duckduckgo.com/?q=" + url.QueryEscape(query) +
		"&format=json&no_html=1&skip_disambig=1"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", fmt.Errorf("web_search: build request: %w", err)
	}
	req.Header.Set("User-Agent", "Homunculus-Agent/1.0")

	resp, err := w.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("web_search: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("web_search: unexpected status %d", resp.StatusCode)
	}

	var ddg ddgResponse
	if err := json.NewDecoder(resp.Body).Decode(&ddg); err != nil {
		return "", fmt.Errorf("web_search: decode response: %w", err)
	}

	var results []string

	// AbstractText is the primary instant answer (Wikipedia-style).
	if ddg.AbstractText != "" {
		results = append(results, fmt.Sprintf("[%s] %s\n  %s", ddg.AbstractSource, ddg.AbstractText, ddg.AbstractURL))
	}

	// Flatten related topics (DuckDuckGo nests some under topic groups).
	flat := flattenTopics(ddg.RelatedTopics)
	for _, topic := range flat {
		if len(results) >= 3 {
			break
		}
		if topic.Text != "" {
			results = append(results, fmt.Sprintf("%s\n  %s", topic.Text, topic.FirstURL))
		}
	}

	if len(results) == 0 {
		return "No results found. Try rephrasing the query.", nil
	}

	return strings.Join(results, "\n\n"), nil
}

func flattenTopics(topics []ddgTopic) []ddgTopic {
	var out []ddgTopic
	for _, t := range topics {
		if t.Text != "" {
			out = append(out, t)
		}
		// Some topics are groups that contain nested Topics instead of Text.
		if len(t.Topics) > 0 {
			out = append(out, flattenTopics(t.Topics)...)
		}
	}
	return out
}
