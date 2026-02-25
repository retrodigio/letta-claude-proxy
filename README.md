# Letta Claude Proxy

Anthropic Messages API proxy that lets [Letta](https://github.com/letta-ai/letta) use Claude via a MAX subscription OAuth token, routed through the Claude Agent SDK.

## How It Works

```
Letta Server (LiteLLM)  ──▶  Proxy (/v1/messages)  ──▶  Claude Agent SDK  ──▶  Claude
```

The proxy exposes a standard Anthropic `/v1/messages` endpoint. Letta's LiteLLM layer sees this as a native Anthropic API and sends Claude-native formatting (system as top-level param, content blocks, Anthropic tool schema). The proxy routes requests through the Claude Agent SDK using a MAX subscription OAuth token.

## Quick Start

```bash
# One command — extracts OAuth token from Keychain and starts the server
./start.sh
```

The server runs on `http://localhost:8400` by default.

## Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OAuth token (extract from macOS Keychain)
export CLAUDE_CODE_OAUTH_TOKEN=$(security find-generic-password -s "Claude Code-credentials" -w | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('claudeAiOauth',{}).get('accessToken',''))")

# Start the proxy
python proxy.py
```

## Configuring Letta

Point Letta at this proxy as a custom Anthropic endpoint:

```bash
# Environment variables for Letta
export ANTHROPIC_API_BASE=http://localhost:8400
export ANTHROPIC_API_KEY=unused  # Required by LiteLLM but not used by proxy
```

When creating an agent in Letta, select a Claude/Anthropic model. LiteLLM will use the Anthropic provider path and send requests to `http://localhost:8400/v1/messages`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CODE_OAUTH_TOKEN` | *(auto-detected)* | OAuth token from Claude MAX subscription |
| `PROXY_HOST` | `0.0.0.0` | Host to bind to |
| `PROXY_PORT` | `8400` | Port to bind to |
| `PROXY_MAX_TURNS` | `1` | Max agent turns per request |

## OAuth Token Resolution

The proxy finds your OAuth token using this priority:

1. **Environment variable** `CLAUDE_CODE_OAUTH_TOKEN` — best for Docker/EC2/CI
2. **Config file** `~/.letta-claude-proxy/config.json` — best for servers without Keychain
3. **macOS Keychain** — automatic if Claude Code is logged in locally

### Config file (for EC2/Linux/Docker)

**Single token:**
```bash
mkdir -p ~/.letta-claude-proxy
cat > ~/.letta-claude-proxy/config.json << 'EOF'
{
  "oauth_token": "sk-ant-oat01-your-token-here"
}
EOF
chmod 600 ~/.letta-claude-proxy/config.json
```

**Multiple tokens (token pooling):**
```bash
cat > ~/.letta-claude-proxy/config.json << 'EOF'
{
  "oauth_tokens": [
    {"token": "sk-ant-oat01-account1-token", "label": "account-1"},
    {"token": "sk-ant-oat01-account2-token", "label": "account-2"},
    {"token": "sk-ant-oat01-account3-token", "label": "account-3"}
  ]
}
EOF
```

You can also place `config.json` next to `proxy.py` as a fallback.

## Token Pooling & Auto-Rotation

If you have multiple MAX subscriptions, the proxy automatically rotates between them when one hits its rate limit.

**How it works:**
- Requests use the active token until Anthropic returns a rate limit error
- On rate limit, the proxy rotates to the next token (round-robin)
- Eventually cycles back to earlier tokens once their limit window has reset
- If all tokens are currently rate-limited, the error is returned to the caller

**Monitor the pool:**
```bash
curl http://localhost:8400/health | jq .token_pool
```

Returns which token is active and how many requests each has served.

**Env var for multiple tokens (comma-separated):**
```bash
export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-token1,sk-ant-oat01-token2"
```

### Getting your OAuth token

**Option A: Claude CLI (recommended)**
```bash
claude setup-token
```
This opens a browser for authentication and outputs the token in your terminal. Copy it to your config file or env var.

**Option B: macOS Keychain (if already logged in)**
```bash
security find-generic-password -s "Claude Code-credentials" -w | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['claudeAiOauth']['accessToken'])"
```

Copy that token to your EC2 config file or set it as an env var.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API |
| `/health` | GET | Health check |

## Supported Features

- System prompts (top-level `system` parameter)
- Multi-turn conversations with content blocks
- Tool/function definitions (Anthropic `input_schema` format)
- Tool use and tool result message types
- Streaming responses (Anthropic SSE format: message_start, content_block_delta, message_stop)
- Multiple Claude models (Sonnet, Opus, Haiku)

## Testing

```bash
# Health check
curl http://localhost:8400/health

# Messages API (non-streaming)
curl http://localhost:8400/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Messages API (streaming)
curl http://localhost:8400/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
