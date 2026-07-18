import re
from abc import ABC, abstractmethod
from typing import Generator

# ── Standardized Error Codes ────────────────────────────────────────
AUTH_ERROR = "AUTH_ERROR"
RATE_LIMIT = "RATE_LIMIT"
NETWORK_ERROR = "NETWORK_ERROR"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
PROVIDER_UNAVAILABLE = "UNAVAILABLE"
TIMEOUT = "TIMEOUT"
CONTEXT_LENGTH_EXCEEDED = "CONTEXT_TOO_LONG"
UNKNOWN = "UNKNOWN"

# Transient errors worth retrying once
_TRANSIENT_CODES = {RATE_LIMIT, NETWORK_ERROR, TIMEOUT, PROVIDER_UNAVAILABLE}
# Errors that should never be retried
_FATAL_CODES = {AUTH_ERROR, MODEL_NOT_FOUND, CONTEXT_LENGTH_EXCEEDED}


class ProviderError(Exception):
    def __init__(self, message: str, code: str = UNKNOWN, provider: str = ""):
        self.code = code
        self.provider = provider
        prefix = f"[{code}]"
        if provider:
            prefix += f" {provider}"
        super().__init__(f"{prefix}: {message}")

    @property
    def is_transient(self) -> bool:
        return self.code in _TRANSIENT_CODES

    @property
    def is_fatal(self) -> bool:
        return self.code in _FATAL_CODES


_SENSITIVE_PATTERNS = re.compile(
    r'(?:sk-[a-zA-Z0-9]{20,}|api[_-]?key[=:]\s*["\']?[a-zA-Z0-9_\-]{16,}|key["\']?\s*:\s*["\'][a-zA-Z0-9_\-]{16,}|bearer\s+[a-zA-Z0-9_\-]{20,})',
    re.IGNORECASE,
)


def sanitize_error(msg: str) -> str:
    """Strip API keys and other sensitive patterns from error messages."""
    return _SENSITIVE_PATTERNS.sub("***REDACTED***", str(msg))


def classify_error(exc: Exception, provider_name: str = "") -> ProviderError:
    """Wrap any exception into a classified ProviderError."""
    raw = str(exc)
    safe = sanitize_error(raw)
    low = raw.lower()

    if any(t in low for t in ("401", "unauthorized", "403", "forbidden", "api key", "auth")):
        code = AUTH_ERROR
    elif any(t in low for t in ("429", "rate limit", "too many requests", "quota")):
        code = RATE_LIMIT
    elif any(t in low for t in ("timeout", "timed out")):
        code = TIMEOUT
    elif any(t in low for t in ("connection", "dns", "econnrefused", "econnreset", "network", "resolve")):
        code = NETWORK_ERROR
    elif any(t in low for t in ("model not found", "not found", "404")):
        code = MODEL_NOT_FOUND
    elif any(t in low for t in ("context length", "too many tokens", "maximum context", "token limit")):
        code = CONTEXT_LENGTH_EXCEEDED
    elif any(t in low for t in ("not available", "not running", "refused", "unavailable", "503")):
        code = PROVIDER_UNAVAILABLE

    return ProviderError(safe, code=code, provider=provider_name)


class LLMProvider(ABC):
    """Abstract Base Class for all LLM Providers (Local or Cloud)."""

    def __init__(self, model_id: str, api_key: str = None, base_url: str = None):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def _do_chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        """Actual chat implementation — subclasses override this."""
        pass

    def chat_stream(self, messages: list, stream: bool = True) -> Generator[str, None, None]:
        """Wrapper with error classification and retry."""
        try:
            yield from self._do_chat_stream(messages, stream)
        except ProviderError:
            raise
        except Exception as e:
            raise classify_error(e, self.__class__.__name__) from e

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured correctly and available."""
        pass
