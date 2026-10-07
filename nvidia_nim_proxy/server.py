```python
"""Local OpenAI-compatible proxy for NVIDIA NIM chat completions."""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import logging
import os
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
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 600
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
TOOL_CALL
