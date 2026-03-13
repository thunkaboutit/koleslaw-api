# API Testing Guide

Quick reference for testing the prompt enhancement API with `curl`. All examples use the synchronous endpoint and
pipe through `jq` for readability.

## Prerequisites

- API running locally (`make dev`)
- Ollama running with registered prompt-enhance models
- `jq` installed for JSON formatting

## Auto-Detection (Default Behavior)

The API automatically selects the best enhancement type based on prompt characteristics. No `enhancement_type` field
needed.

### Short prompt (expand)

Prompts under 15 words are expanded with detail, constraints, and structure:

```bash
curl -s http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a function that sorts a list"}' | jq .
```

### Medium prompt (refine)

Prompts between 15-80 words are refined for clarity and precision:

```bash
curl -s http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I need a REST API endpoint that accepts a JSON payload with user registration data, validates the email format and password strength, stores the user in a PostgreSQL database, and returns a JWT token for immediate authentication"}' | jq .
```

### Long unstructured prompt (structure)

Prompts over 80 words without existing organization are restructured into clear sections:

```bash
curl -s http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I want to build a system that handles real-time data processing from multiple IoT sensors and it needs to aggregate the data and store it in a time series database and also provide a dashboard for monitoring and alerting and the system should scale to handle thousands of sensors and the data format varies between sensor types and some send JSON and others send binary protobuf messages and we need to normalize everything into a common schema and there should be anomaly detection running on the streams and alerts should go to Slack and PagerDuty"}' | jq .
```

## Explicit Enhancement Type Override

Force a specific enhancement type regardless of prompt length:

```bash
curl -s http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a function that sorts a list", "enhancement_type": "structure"}' | jq .
```

Valid values: `auto` (default), `expand`, `refine`, `structure`.

## Explicit Model Override

Route to a frontier model instead of the default local Ollama model:

```bash
curl -s http://localhost:8000/v1/enhance/sync \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a function that sorts a list", "model": "anthropic/claude-sonnet-4-5"}' | jq .
```

## Streaming (SSE)

Use the streaming endpoint to receive results as server-sent events:

```bash
curl -N http://localhost:8000/v1/enhance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "write a function that sorts a list"}'
```

## Available Models

List all registered models and their tiers:

```bash
curl -s http://localhost:8000/v1/models | jq .
```

## Health Check

Verify the API, database, and Redis are all healthy:

```bash
curl -s http://localhost:8000/v1/health | jq .
```
