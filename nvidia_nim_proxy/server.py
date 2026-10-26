
"""Local OpenAI-compatible proxy for NVIDIA NIM chat completions."""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import ParseResult, urlparse

from nvidia_nim_proxy import __version__
from nvidia_nim_proxy.credentials import (
    API_KEY_MODE_CLIENT as API_KEY_MODE_CLIENT,
    API_KEY_MODE_ENV as API_KEY_MODE_ENV,
    API_KEY_MODE_POOL as API_KEY_MODE_POOL,
    CredentialBroker,
    extract_bearer_token as extract_bearer_token,
    fingerprint_secret as fingerprint_secret,
    load_pool_keys,
)
from nvidia_nim_proxy.key_pool import (
    NvidiaKeyPool,
    PoolQueueFullError,
    PoolUnavailableError,
    PoolWaitTimeoutError,
)
from nvidia_nim_proxy.model_catalog import DEFAULT_REFRESH_SECONDS, ModelCatalog
from nvidia_nim_proxy.opencode_sync import OpenCodeModelSync
from nvidia_nim_proxy.sanitizer import ProviderContext, sanitize_chat_completion_body
from nvidia_nim_proxy.scheduler import (
    QueueFullError,
    QueueWaitTimeoutError,
    RequestScheduler,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_UPSTREAM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 1200
DEFAULT_MAX_CONCURRENT_PER_KEY = 1
DEFAULT_MAX_QUEUE_PER_KEY = 4
DEFAULT_MAX_TOTAL_QUEUED = 32
DEFAULT_QUEUE_WAIT_SECONDS = 180
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 60
DEFAULT_MAX_5XX_FAILOVERS = 1
MAX_REQUEST_BYTES = 10 * 1024 * 1024
STREAM_CHUNK_SIZE = 8192
STREAM_DIAGNOSTIC_SCAN_BYTES = 8 * 1024
STREAM_DIAGNOSTIC_SCAN_EVENTS = 8
STREAM_KEEPALIVE_SECONDS = 10.0
HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)
TOOL_CALL_LEAK_MARKERS = (b"<tool_call", b"</tool_call", b"&lt;tool_call", b"&lt;/tool_call")
TOOL_CALL_TEXT_DIAGNOSTIC = (
    "Provider/model compatibility issue: NVIDIA NIM returned tool-call markup as normal "
    "assistant text instead of real OpenAI-compatible tool_calls. The proxy did not execute "
    "that text. Use this model for normal chat, or switch ZCode to a provider/model that "
    "supports real OpenAI tool calls for agentic file and command workflows."
)

logger = logging.getLogger("zcode-nim-proxy")


class ProxyConfig:
    """Runtime settings for the local proxy."""

    def __init__(
        self,
        upstream_base_url: str,
        api_key: str | None,
        tool_call_text_mode: str = "diagnostic",
        api_key_mode: str = API_KEY_MODE_ENV,
        upstream_timeout_seconds: int = DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
        local_client_key: str | None = None,
        pool_keys: tuple[str, ...] = (),
        max_concurrent_per_key: int = DEFAULT_MAX_CONCURRENT_PER_KEY,
        max_queue_per_key: int = DEFAULT_MAX_QUEUE_PER_KEY,
        max_total_queued: int = DEFAULT_MAX_TOTAL_QUEUED,
        queue_wait_seconds: int = DEFAULT_QUEUE_WAIT_SECONDS,
        rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
        max_5xx_failovers: int = DEFAULT_MAX_5XX_FAILOVERS,
        model_catalog: ModelCatalog | None = None,
    ) -> None:
        normalized_upstream_base_url = upstream_base_url.rstrip("/")
        validate_upstream_base_url(normalized_upstream_base_url)
        self.upstream_base_url = normalized_upstream_base_url
        self.api_key = api_key
        self.tool_call_text_mode = tool_call_text_mode
        self.api_key_mode = api_key_mode
        self.upstream_timeout_seconds = upstream_timeout_seconds
        self.model_catalog = model_catalog
        self.credential_broker = CredentialBroker(
            mode=api_key_mode,
            env_api_key=api_key,
            local_client_key=local_client_key,
        )
        self.request_scheduler = RequestScheduler(
            max_concurrent_per_key=max_concurrent_per_key,
            max_queue_per_key=max_queue_per_key,
            max_total_queued=max_total_queued,
            queue_wait_seconds=queue_wait_seconds,
        )
        self.key_pool = (
            NvidiaKeyPool(
                pool_keys,
                max_concurrent_per_key=max_concurrent_per_key,
                max_total_queued=max_total_queued,
                queue_wait_seconds=queue_wait_seconds,
                default_cooldown_seconds=rate_limit_cooldown_seconds,
                max_5xx_failovers=max_5xx_failovers,
            )
            if api_key_mode == API_KEY_MODE_POOL
            else None
        )


@dataclass(frozen=True)
class UpstreamRequest:
    """Prepared request details for NVIDIA NIM."""

    netloc: str
    path: str
    payload: bytes
    headers: dict[str, str]
    stream: bool
    use_tls: bool
    api_key_fingerprint: str


@dataclass(frozen=True)
class UpstreamAttempt:
    """An opened NVIDIA response and its owning connection."""

    connection: http.client.HTTPConnection
    response: http.client.HTTPResponse
    request: UpstreamRequest
    elapsed_ms: int


@dataclass(frozen=True)
