#!/usr/bin/env bash
set -euo pipefail

# Letta Claude Proxy - Quick Start
# Extracts OAuth token from macOS Keychain and starts the proxy server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extract OAuth token from macOS Keychain if not already set
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "Extracting OAuth token from macOS Keychain..."
    RAW=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null || true)
    if [ -z "$RAW" ]; then
        echo "ERROR: Could not extract token from Keychain."
        echo "Make sure Claude Code is installed and you are logged in."
        exit 1
    fi

    # The keychain entry is JSON — pull the oauth token out
    TOKEN=$(echo "$RAW" | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Handle both flat and nested formats
if isinstance(data, dict):
    token = data.get('claudeAiOauth', {}).get('accessToken', '') or data.get('accessToken', '')
    print(token)
else:
    print('')
" 2>/dev/null || echo "")

    # Fallback: if the raw value already looks like a token, use it directly
    if [ -z "$TOKEN" ] && echo "$RAW" | grep -q "^sk-ant-"; then
        TOKEN="$RAW"
    fi

    if [ -z "$TOKEN" ]; then
        echo "ERROR: Could not parse OAuth token from Keychain entry."
        echo "Raw value starts with: ${RAW:0:20}..."
        echo "Try setting CLAUDE_CODE_OAUTH_TOKEN manually."
        exit 1
    fi

    export CLAUDE_CODE_OAUTH_TOKEN="$TOKEN"
    echo "OAuth token loaded (${TOKEN:0:15}...)"
fi

# Create venv and install dependencies if needed
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

if ! "$PYTHON" -c "import fastapi, uvicorn, claude_agent_sdk" 2>/dev/null; then
    echo "Installing dependencies..."
    "$PIP" install -r "$SCRIPT_DIR/requirements.txt"
fi

echo "Starting Letta Claude Proxy on ${PROXY_HOST:-0.0.0.0}:${PROXY_PORT:-8400}..."
exec "$PYTHON" "$SCRIPT_DIR/proxy.py"
