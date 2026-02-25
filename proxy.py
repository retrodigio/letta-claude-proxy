"""
Letta Claude Proxy - Anthropic Messages API proxy backed by Claude Agent SDK.

Exposes /v1/messages so Letta Server (via LiteLLM) can use a Claude MAX subscription
OAuth token via the Claude Agent SDK with native Anthropic formatting.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PORT = int(os.getenv("PROXY_PORT", "8400"))
MAX_TURNS = int(os.getenv("PROXY_MAX_TURNS", "1"))


CONFIG_PATHS = [
    os.path.expanduser("~/.letta-claude-proxy/config.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
]

class TokenPool:
    """Manages a pool of OAuth tokens with round-robin rotation on rate limits."""

    def __init__(self) -> None:
        self._tokens: list[str] = []
        self._labels: list[str] = []
        self._active_index: int = 0
        self._usage_counts: dict[int, int] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._tokens)

    @property
    def active_label(self) -> str:
        if not self._labels:
            return "none"
        return self._labels[self._active_index]

    def load(self) -> None:
        """Load tokens from all sources: env var, config file, macOS Keychain."""
        tokens_found: list[tuple[str, str]] = []  # (token, label)

        # 1. Environment variable (single token or comma-separated)
        env_tokens = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
        if env_tokens:
            for i, t in enumerate(env_tokens.split(","), 1):
                t = t.strip()
                if t:
                    tokens_found.append((t, f"env-{i}"))

        # 2. Config file (supports both single token and array)
        for config_path in CONFIG_PATHS:
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    # Array of tokens (preferred for multi-account)
                    token_list = config.get("oauth_tokens", [])
                    if isinstance(token_list, list):
                        for i, entry in enumerate(token_list, 1):
                            if isinstance(entry, str):
                                tokens_found.append((entry, f"config-{i}"))
                            elif isinstance(entry, dict):
                                t = entry.get("token", "")
                                label = entry.get("label", f"config-{i}")
                                if t:
                                    tokens_found.append((t, label))

                    # Single token (backward compat)
                    if not token_list:
                        single = config.get("oauth_token", "") or config.get("CLAUDE_CODE_OAUTH_TOKEN", "")
                        if single:
                            tokens_found.append((single, f"config-file"))

                    if tokens_found:
                        logger.info("Loaded tokens from %s", config_path)
                        break  # Use first config found
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Failed to read config %s: %s", config_path, e)

        # 3. macOS Keychain (single token fallback)
        if not tokens_found:
            try:
                import subprocess
                raw = subprocess.check_output(
                    ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if raw:
                    data = json.loads(raw)
                    token = (
                        data.get("claudeAiOauth", {}).get("accessToken", "")
                        or data.get("accessToken", "")
                    )
                    if token:
                        tokens_found.append((token, "keychain"))
                        logger.info("OAuth token loaded from macOS Keychain")
            except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
                pass

        if not tokens_found:
            raise RuntimeError(
                "No OAuth tokens found. Set them via:\n"
                "  1. CLAUDE_CODE_OAUTH_TOKEN env var (comma-separated for multiple)\n"
                '  2. ~/.letta-claude-proxy/config.json: {"oauth_tokens": ["sk-...", "sk-..."]}\n'
                "  3. macOS Keychain (automatic if Claude Code is logged in)"
            )

        self._tokens = [t for t, _ in tokens_found]
        self._labels = [l for _, l in tokens_found]
        self._active_index = 0
        self._usage_counts = {i: 0 for i in range(len(self._tokens))}
        logger.info(
            "Token pool initialized: %d token(s) — [%s]",
            len(self._tokens),
            ", ".join(self._labels),
        )

    def get_active_token(self) -> str:
        """Get the current active token."""
        if not self._tokens:
            raise RuntimeError("Token pool is empty")
        return self._tokens[self._active_index]

    async def record_usage(self) -> None:
        """Record a successful request against the active token."""
        async with self._lock:
            self._usage_counts[self._active_index] = (
                self._usage_counts.get(self._active_index, 0) + 1
            )

    async def rotate_on_limit(self) -> str:
        """Rotate to the next token in the pool (round-robin)."""
        async with self._lock:
            prev = self._labels[self._active_index]
            self._active_index = (self._active_index + 1) % len(self._tokens)
            logger.warning(
                "Token '%s' hit rate limit — rotated to '%s' (index %d)",
                prev,
                self._labels[self._active_index],
                self._active_index,
            )
            return self._labels[self._active_index]

    def status(self) -> dict:
        """Return pool status for the health endpoint."""
        tokens_status = []
        for i in range(len(self._tokens)):
            tokens_status.append({
                "label": self._labels[i],
                "active": i == self._active_index,
                "requests": self._usage_counts.get(i, 0),
                "token_prefix": self._tokens[i][:15] + "..." if self._tokens[i] else "",
            })
        return {
            "pool_size": len(self._tokens),
            "active_token": self._labels[self._active_index] if self._tokens else None,
            "tokens": tokens_status,
        }


# Global token pool
token_pool = TokenPool()


# ---------------------------------------------------------------------------
# Request / Response schemas (Anthropic Messages API)
# ---------------------------------------------------------------------------


class AnthropicToolDef(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict | None = None


class AnthropicMessage(BaseModel):
    role: str
    content: Any  # str | list[dict]


class MessagesRequest(BaseModel):
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    messages: list[AnthropicMessage]
    system: Any | None = None  # str | list[dict]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    tools: list[AnthropicToolDef] | None = None
    tool_choice: dict | None = None
    stop_sequences: list[str] | None = None
    metadata: dict | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: dict | None = None


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str = DEFAULT_MODEL
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(
    messages: list[AnthropicMessage],
    system: Any | None,
    tools: list[AnthropicToolDef] | None,
) -> tuple[str | None, str]:
    """Convert Anthropic-format messages into a system prompt and user-facing prompt."""
    system_parts: list[str] = []
    conversation_parts: list[str] = []

    # Extract system prompt from the top-level system parameter
    if system:
        if isinstance(system, str):
            system_parts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append(block.get("text", ""))

    for msg in messages:
        content_text = _extract_content_text(msg.content)

        if msg.role == "user":
            # Check if content contains tool_result blocks
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_id = item.get("tool_use_id", "unknown")
                        result_content = item.get("content", "")
                        if isinstance(result_content, list):
                            result_text = " ".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        else:
                            result_text = str(result_content)
                        conversation_parts.append(
                            f"[Tool Result (call {tool_id})]\n{result_text}"
                        )
                # Also include any plain text content
                text_parts = []
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    conversation_parts.append(f"[User]\n{' '.join(text_parts)}")
            else:
                conversation_parts.append(f"[User]\n{content_text}")
        elif msg.role == "assistant":
            # Check for tool_use blocks in assistant content
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_name = item.get("name", "unknown")
                        tool_input = json.dumps(item.get("input", {}))
                        tool_id = item.get("id", "unknown")
                        conversation_parts.append(
                            f"[Assistant Tool Call ({tool_id}): {tool_name}]\n{tool_input}"
                        )
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            conversation_parts.append(f"[Assistant]\n{text}")
            else:
                conversation_parts.append(f"[Assistant]\n{content_text}")

    # Include tool definitions in system prompt
    if tools:
        tool_descriptions = _format_tools(tools)
        system_parts.append(
            "The following tools/functions are available. "
            "When you want to call one, respond with a JSON object containing "
            '"tool_calls" with the function name and arguments.\n\n'
            f"{tool_descriptions}"
        )

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

    return system_prompt, prompt


def _extract_content_text(content: Any) -> str:
    """Extract plain text from various content formats."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_result":
                    # Handled separately in _build_prompt
                    pass
                elif item.get("type") == "tool_use":
                    # Handled separately in _build_prompt
                    pass
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _format_tools(tools: list[AnthropicToolDef]) -> str:
    """Format Anthropic tool definitions into a human-readable block."""
    lines = []
    for t in tools:
        params = json.dumps(t.input_schema, indent=2) if t.input_schema else "{}"
        lines.append(
            f"- {t.name}: {t.description or '(no description)'}\n  Parameters: {params}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude Agent SDK call
# ---------------------------------------------------------------------------


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect if an exception indicates a rate limit / quota exceeded."""
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "rate limit", "rate_limit", "429", "quota", "overloaded",
        "too many requests", "capacity", "usage limit", "exceeded",
    ])


async def _call_claude(
    prompt: str,
    system_prompt: str | None,
    model: str,
    max_turns: int = MAX_TURNS,
) -> tuple[str, dict]:
    """Send a prompt to Claude via the Agent SDK with automatic token rotation."""
    # Try with current token, rotate on rate limit
    attempts = token_pool.size
    last_error: Exception | None = None

    for attempt in range(attempts):
        token = token_pool.get_active_token()

        opts = ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=max_turns,
            model=model,
            env={"CLAUDE_CODE_OAUTH_TOKEN": token},
        )
        if system_prompt:
            opts.system_prompt = system_prompt

        response_text = ""
        usage_info: dict = {}

        try:
            async for message in query(prompt=prompt, options=opts):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(message, ResultMessage):
                    if message.result and not response_text:
                        response_text = message.result
                    if message.usage:
                        usage_info = message.usage

            await token_pool.record_usage()
            return response_text, usage_info

        except Exception as exc:
            last_error = exc
            if _is_rate_limit_error(exc):
                new_label = await token_pool.rotate_on_limit()
                logger.info("Retrying with token '%s' (attempt %d/%d)", new_label, attempt + 2, attempts)
                continue
            else:
                logger.exception("Claude Agent SDK query failed (non-rate-limit error)")
                raise

    # Cycled through all tokens, all rate-limited
    raise last_error or RuntimeError("All OAuth tokens are currently rate-limited")


# ---------------------------------------------------------------------------
# Streaming helpers (Anthropic SSE format)
# ---------------------------------------------------------------------------


async def _stream_claude(
    prompt: str,
    system_prompt: str | None,
    model: str,
    message_id: str,
    max_turns: int = MAX_TURNS,
):
    """Yield SSE chunks in Anthropic streaming format."""
    token = token_pool.get_active_token()

    opts = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=max_turns,
        model=model,
        env={"CLAUDE_CODE_OAUTH_TOKEN": token},
    )
    if system_prompt:
        opts.system_prompt = system_prompt

    # message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # content_block_start for index 0 (text)
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

    # ping
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    output_tokens = 0

    try:
        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        delta = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": block.text},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            elif isinstance(message, ResultMessage):
                if message.result:
                    delta = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": message.result},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                if message.usage and isinstance(message.usage, dict):
                    output_tokens = message.usage.get("output_tokens", 0)
    except Exception:
        logger.exception("Streaming query failed")

    # content_block_stop
    block_stop = {"type": "content_block_stop", "index": 0}
    yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

    # message_delta with stop_reason
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Letta Claude Proxy starting on %s:%s", HOST, PORT)
    try:
        token_pool.load()
    except RuntimeError as e:
        logger.warning("Token pool init: %s", e)
    yield
    logger.info("Letta Claude Proxy shutting down")


app = FastAPI(
    title="Letta Claude Proxy",
    description="Anthropic Messages API proxy backed by Claude Agent SDK",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API endpoint."""
    body = await request.json()

    try:
        req = MessagesRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e),
                },
            },
        )

    logger.info(
        "Request: model=%s messages=%d stream=%s",
        req.model,
        len(req.messages),
        req.stream,
    )

    system_prompt, prompt = _build_prompt(req.messages, req.system, req.tools)

    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "No user message content provided",
                },
            },
        )

    model = req.model or DEFAULT_MODEL

    if req.stream:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        return StreamingResponse(
            _stream_claude(prompt, system_prompt, model, message_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response_text, usage_info = await _call_claude(prompt, system_prompt, model)

    if not response_text:
        response_text = "(no response)"

    input_tokens = usage_info.get("input_tokens", 0) if isinstance(usage_info, dict) else 0
    output_tokens = usage_info.get("output_tokens", 0) if isinstance(usage_info, dict) else 0

    response = MessagesResponse(
        model=model,
        content=[AnthropicContentBlock(type="text", text=response_text)],
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )

    return response.model_dump()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "token_pool": token_pool.status(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "proxy:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
