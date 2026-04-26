package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// DiscordWebhook sends messages to Discord channels via incoming webhooks.
// Webhook URLs are read from environment variables at call time so they can
// be rotated without restarting the process.
//
// Environment variables:
//
//	DISCORD_WEBHOOK_URL          — default channel (used when channel is omitted)
//	DISCORD_WEBHOOK_<NAME>       — named channel, e.g. DISCORD_WEBHOOK_ALERTS
//
// Named channel lookup is case-insensitive: channel "alerts" resolves to
// DISCORD_WEBHOOK_ALERTS.
type DiscordWebhook struct {
	httpClient *http.Client
}

// NewDiscordWebhook returns a DiscordWebhook with a sensible HTTP timeout.
func NewDiscordWebhook() *DiscordWebhook {
	return &DiscordWebhook{
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

func (DiscordWebhook) Name() string { return "discord_send" }

func (DiscordWebhook) Description() string {
	return "Send a message to a Discord channel via a webhook. Use this to notify, report results, or share information with a Discord server."
}

func (DiscordWebhook) InputDescription() string {
	return "A JSON object with the following fields:\n\n" +
		"  message  (required) — The text to send. Supports Discord markdown:\n" +
		"                         **bold**, *italic*, `code`, ```block```, > quote.\n" +
		"  channel  (optional) — Named destination channel. Maps to the environment\n" +
		"                         variable DISCORD_WEBHOOK_<CHANNEL> (uppercase).\n" +
		"                         Omit to use the default DISCORD_WEBHOOK_URL.\n" +
		"  username (optional) — Override the webhook's display name for this message.\n\n" +
		"Examples:\n" +
		"  {\"message\": \"Deployment complete.\"}\n" +
		"  {\"channel\": \"alerts\", \"message\": \"**Error:** build failed on main.\"}\n" +
		"  {\"channel\": \"general\", \"message\": \"Search results ready.\", \"username\": \"Homunculus\"}"
}

// discordInput is the parsed form of the tool's JSON input.
type discordInput struct {
	Message  string `json:"message"`
	Channel  string `json:"channel"`
	Username string `json:"username"`
}

// discordPayload is the Discord webhook request body.
type discordPayload struct {
	Content  string `json:"content"`
	Username string `json:"username,omitempty"`
}

func (d *DiscordWebhook) Run(ctx context.Context, input string) (string, error) {
	var in discordInput
	if err := json.Unmarshal([]byte(strings.TrimSpace(input)), &in); err != nil {
		return "", fmt.Errorf("discord_send: invalid JSON input: %w", err)
	}

	if strings.TrimSpace(in.Message) == "" {
		return "", fmt.Errorf("discord_send: message must not be empty")
	}

	webhookURL, err := resolveWebhookURL(in.Channel)
	if err != nil {
		return "", err
	}

	payload := discordPayload{
		Content:  in.Message,
		Username: in.Username,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("discord_send: marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, webhookURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("discord_send: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("discord_send: request failed: %w", err)
	}
	defer resp.Body.Close()

	// Discord returns 204 No Content on success.
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("discord_send: unexpected status %d", resp.StatusCode)
	}

	channel := in.Channel
	if channel == "" {
		channel = "default"
	}
	return fmt.Sprintf("Message sent to channel %q.", channel), nil
}

// resolveWebhookURL returns the webhook URL for the given channel name.
// An empty channel name resolves to DISCORD_WEBHOOK_URL.
func resolveWebhookURL(channel string) (string, error) {
	var envKey string
	if channel == "" {
		envKey = "DISCORD_WEBHOOK_URL"
	} else {
		envKey = "DISCORD_WEBHOOK_" + strings.ToUpper(channel)
	}

	url := os.Getenv(envKey)
	if url == "" {
		if channel == "" {
			return "", fmt.Errorf("discord_send: environment variable DISCORD_WEBHOOK_URL is not set")
		}
		return "", fmt.Errorf("discord_send: environment variable %s is not set (channel %q)", envKey, channel)
	}
	return url, nil
}
