# Letta Claude Proxy

## Goal
Build a Python proxy server that lets Letta Server use Claude via a MAX subscription OAuth token, routed through the Claude Agent SDK.

## Architecture
- FastAPI server exposing OpenAI-compatible `/v1/chat/completions` endpoint
- Internally uses `claude_agent_sdk.query()` with `allowed_tools=[]` for plain completions
- Accepts standard OpenAI-format requests (messages array, model, temperature, etc.)
- Returns OpenAI-compatible responses so Letta thinks it's talking to a standard API

## Auth
- Uses `CLAUDE_CODE_OAUTH_TOKEN` environment variable
- Token can be extracted from macOS Keychain: `security find-generic-password -s "Claude Code-credentials" -w`
- The OAuth token starts with `sk-ant-oat01-...`

## How Letta Will Use It
Letta normally calls Anthropic directly with `ANTHROPIC_API_KEY`. Instead, we configure Letta to use a custom OpenAI-compatible endpoint pointing to this proxy (e.g., `http://localhost:8400/v1`).

## Key Details
- The Agent SDK Python package is `claude-agent-sdk` (pip install claude-agent-sdk)
- The SDK's `query()` function is async and yields messages (AssistantMessage, ResultMessage, etc.)
- Set `allowed_tools=[]` to disable agent tools and get plain chat completions
- Need to handle system prompts, multi-turn conversations, and tool definitions from Letta
- Support streaming would be nice but not required for v1

## Reference Code (Agent SDK usage)
```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

async for message in query(
    prompt="your prompt",
    options=ClaudeAgentOptions(allowed_tools=[]),
):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                response_text += block.text
    elif isinstance(message, ResultMessage):
        if message.result:
            response_text = message.result
```

## Files to Create
1. `proxy.py` - Main FastAPI server
2. `requirements.txt` - Dependencies
3. `README.md` - Setup and usage instructions
4. `start.sh` - Quick start script that extracts OAuth token from keychain and starts server

## Environment
- macOS (x86_64)
- Python 3.x available
- Claude Code v2.1.42 installed, logged in with MAX subscription
- Letta server can be run via Docker or pip
