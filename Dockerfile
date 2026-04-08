# syntax=docker/dockerfile:1

# ── Build stage ───────────────────────────────────────────────────────────────
FROM golang:1.26.1-alpine AS builder

WORKDIR /src

# Download dependencies first (cached layer unless go.mod/go.sum change).
COPY go.mod go.sum* ./
RUN go mod download

# Copy source and build.
COPY . .

FROM builder AS cli-builder
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /bin/cli ./cmd/cli

# ── CLI runtime image ─────────────────────────────────────────────────────────
FROM alpine:3.20 AS cli

RUN apk add --no-cache ca-certificates

COPY --from=cli-builder /bin/cli /bin/cli

# Prompts are mounted as a volume at runtime (see docker-compose.yml).
# This copy provides a fallback when running the image standalone.
COPY prompts/ /app/prompts/

WORKDIR /app

ENTRYPOINT ["/bin/cli"]
