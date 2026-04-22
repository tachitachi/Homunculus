package tools

import (
	"fmt"
	"strings"
)

// Registry holds the set of tools available to the agent.
type Registry struct {
	tools map[string]Tool
	order []string // insertion order for stable prompt output
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a tool to the registry. Panics on duplicate names so
// misconfiguration surfaces at startup rather than silently at runtime.
func (r *Registry) Register(t Tool) {
	name := t.Name()
	if _, exists := r.tools[name]; exists {
		panic(fmt.Sprintf("tools: duplicate tool name %q", name))
	}
	r.tools[name] = t
	r.order = append(r.order, name)
}

// Get returns the tool with the given name, or nil if not found.
func (r *Registry) Get(name string) Tool {
	return r.tools[name]
}

// Names returns all registered tool names in insertion order.
func (r *Registry) Names() []string {
	names := make([]string, len(r.order))
	copy(names, r.order)
	return names
}

// Descriptions returns a formatted string listing all tools and their
// descriptions, ready to be injected into the ReAct system prompt.
//
// Example output:
//
//   - calculator: evaluate a math expression like "2 + 2 * 3"
//   - web_search: search the web for current information
func (r *Registry) Descriptions() string {
	var sb strings.Builder
	for _, name := range r.order {
		t := r.tools[name]
		fmt.Fprintf(&sb, "- %s: %s\n", name, t.Description())
	}
	return strings.TrimRight(sb.String(), "\n")
}
