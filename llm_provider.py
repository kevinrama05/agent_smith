"""
llm_provider.py — Agent Smith's LLM provider abstraction (Section 4.6).

Three layers, each with one job:

  LLMProvider        -- talks to ONE provider's API (OpenAI-compatible
                         chat/completions shape, which covers Groq,
                         OpenRouter, Together, Fireworks, Mistral, etc.)
  KeyRotator         -- cycles through that provider's API keys, skipping
                         ones currently rate-limited
  ProviderRouter      -- tries providers in priority order, falling back to
                         the next one when a provider's keys are all
                         exhausted or it's otherwise unavailable

This is what your agent loop calls -- it never talks to requests/HTTP
directly, and never knows which provider actually served a given step.

Config-driven, not hardcoded: providers and their keys come from a JSON
file (provider names, base URLs, model names) plus environment variables
(the actual secret key values) -- see provider_template.json and the
load_provider_router() docstring below. No API key is ever written in code.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMProviderError(RuntimeError):
    """Base class for all provider-related failures."""


class RateLimitError(LLMProviderError):
    """The provider rejected the request due to rate limiting / quota."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds, from the Retry-After header if sent


class TransientProviderError(LLMProviderError):
    """A retryable failure: timeout, 5xx, connection reset, etc."""


class AllKeysExhaustedError(LLMProviderError):
    """Every API key for this provider is currently rate-limited or has
    failed too many times in a row."""


class AllProvidersExhaustedError(LLMProviderError):
    """Every configured provider failed. This is the one that should
    ultimately surface to the agent loop / StepMetrics.error."""


# ---------------------------------------------------------------------------
# Response shape -- everything StepMetrics needs, in one object
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    text: str                     # -> StepMetrics.llm_output
    input_tokens: int             # -> StepMetrics.input_tokens
    output_tokens: int            # -> StepMetrics.output_tokens
    request_time_ms: float        # -> StepMetrics.request_time_ms
    api_url: str                  # -> StepMetrics.api_url
    model_name: str               # -> StepMetrics.model_name
    retries: int = 0              # -> StepMetrics.retries
    raw: Optional[dict] = None    # full provider payload, for debugging


# ---------------------------------------------------------------------------
# Key rotation
# ---------------------------------------------------------------------------


@dataclass
class _KeyState:
    key: str
    cooldown_until: float = 0.0   # time.time() timestamp; 0 = never rate-limited


class KeyRotator:
    """Round-robins through a provider's API keys, skipping any currently
    in cooldown after a rate-limit response."""

    def __init__(self, keys: List[str], cooldown_seconds: float = 60.0):
        if not keys:
            raise ValueError("KeyRotator requires at least one API key.")
        self._states = [_KeyState(key=k) for k in keys]
        self._cooldown_seconds = cooldown_seconds
        self._next_index = 0

    def get_key(self) -> str:
        """Returns the next available (not-in-cooldown) key, round-robin.
        Raises AllKeysExhaustedError if every key is currently cooling down."""
        now = time.time()
        n = len(self._states)
        for offset in range(n):
            idx = (self._next_index + offset) % n
            state = self._states[idx]
            if state.cooldown_until <= now:
                self._next_index = (idx + 1) % n
                return state.key
        raise AllKeysExhaustedError(
            f"All {n} API key(s) are currently rate-limited. "
            f"Next available in "
            f"{min(s.cooldown_until for s in self._states) - now:.0f}s."
        )

    def mark_rate_limited(self, key: str) -> None:
        for state in self._states:
            if state.key == key:
                state.cooldown_until = time.time() + self._cooldown_seconds
                return

    def has_available_key(self) -> bool:
        now = time.time()
        return any(s.cooldown_until <= now for s in self._states)


# ---------------------------------------------------------------------------
# LLMProvider interface
# ---------------------------------------------------------------------------


class LLMProvider:
    """Base interface every concrete provider implements. The agent loop
    (and ProviderRouter) only ever depends on this, never on a specific
    provider's SDK or wire format -- this is the "sufficiently abstract"
    requirement from Section 4.6.1."""

    name: str

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> LLMResponse:
        raise NotImplementedError


# Reasoning models (gpt-oss, o-series, ...) spend hidden "reasoning" tokens
# that are billed as completion_tokens. At the provider's default effort they
# can eat the whole per-step max_tokens and return EMPTY content, burning the
# MBPP output budget (1500) without producing a single line of code. Asking
# for low effort keeps reasoning short and leaves room for the actual answer.
_REASONING_MODEL_MARKERS = ("gpt-oss", "o1", "o3", "o4-mini")


def _is_reasoning_model(model: str) -> bool:
    name = model.lower().rsplit("/", 1)[-1]
    return any(name.startswith(m) for m in _REASONING_MODEL_MARKERS)


def _default_reasoning_effort(model: str) -> Optional[str]:
    override = os.environ.get("LLM_REASONING_EFFORT")
    if override is not None:
        return override or None  # empty string disables it explicitly
    return "low" if _is_reasoning_model(model) else None


def _truncate_at_stop(text: str, stop: Optional[List[str]]) -> str:
    """Client-side equivalent of the API's `stop` parameter."""
    for marker in stop or []:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text


class OpenAICompatibleProvider(LLMProvider):
    """Works for any provider exposing an OpenAI-style
    POST {base_url}/chat/completions endpoint -- which covers Groq,
    OpenRouter, Together AI, Fireworks AI, Mistral AI, and most others
    on the subject's example list. One class, many providers: the
    abstraction the subject is actually asking for.

    Handles per-provider key rotation and retries internally; raises
    AllKeysExhaustedError only once every key has been tried and failed,
    which is the signal ProviderRouter uses to fall back to the next
    provider.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_keys: List[str],
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 60.0,
        max_retries_per_key: int = 2,
        cooldown_seconds: float = 60.0,
        max_rate_limit_wait_seconds: float = 30.0,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.key_rotator = KeyRotator(api_keys, cooldown_seconds=cooldown_seconds)
        self.extra_headers = extra_headers or {}
        self.timeout_seconds = timeout_seconds
        self.max_retries_per_key = max_retries_per_key
        self.max_rate_limit_wait_seconds = max_rate_limit_wait_seconds
        # Set to False the first time the provider rejects `reasoning_effort`,
        # so later steps don't keep paying for a failed request.
        self._send_reasoning_effort = True

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> LLMResponse:
        total_retries = 0
        num_keys_tried = 0
        max_keys_to_try = len(self.key_rotator._states)

        while num_keys_tried < max_keys_to_try:
            try:
                key = self.key_rotator.get_key()
            except AllKeysExhaustedError:
                raise
            num_keys_tried += 1

            for attempt in range(self.max_retries_per_key):
                try:
                    return self._single_request(
                        key, messages, model, stop, max_tokens, temperature, total_retries
                    )
                except RateLimitError as e:
                    total_retries += 1
                    wait = e.retry_after if e.retry_after is not None else self._backoff_delay(attempt + 1)
                    # Per-minute token/request limits (e.g. Groq's free tier)
                    # reset within seconds. If there's no other key to rotate
                    # to, waiting briefly beats failing the whole task.
                    others_available = any(
                        st.key != key and st.cooldown_until <= time.time()
                        for st in self.key_rotator._states
                    )
                    if (not others_available and wait <= self.max_rate_limit_wait_seconds
                            and attempt + 1 < self.max_retries_per_key):
                        time.sleep(wait + random.uniform(0, 0.5))
                        continue
                    self.key_rotator.mark_rate_limited(key)
                    break  # move to the next key
                except TransientProviderError:
                    total_retries += 1
                    if attempt + 1 >= self.max_retries_per_key:
                        break  # give this key up, try the next one
                    time.sleep(self._backoff_delay(attempt))

        raise AllKeysExhaustedError(
            f"Provider '{self.name}': all keys exhausted after {total_retries} retries."
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        base = 2 ** attempt
        jitter = random.uniform(0, 0.5)
        return base + jitter

    def _single_request(
        self,
        key: str,
        messages: List[Dict[str, str]],
        model: str,
        stop: Optional[List[str]],
        max_tokens: int,
        temperature: float,
        retries_so_far: int,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Reasoning models draft their answer in hidden reasoning first, and the
        # API applies `stop` to that reasoning too: the model writes
        # "<end_code>" while thinking and generation halts before any visible
        # content is produced. Apply stop sequences client-side for them.
        client_side_stop = _is_reasoning_model(model)
        if stop and not client_side_stop:
            payload["stop"] = stop
        reasoning_effort = _default_reasoning_effort(model) if self._send_reasoning_effort else None
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        except requests.exceptions.Timeout as e:
            raise TransientProviderError(f"Request to {self.name} timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise TransientProviderError(f"Connection to {self.name} failed: {e}") from e
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 429:
            try:
                retry_after: Optional[float] = float(resp.headers.get("retry-after", ""))
            except ValueError:
                retry_after = None
            raise RateLimitError(f"{self.name} rate-limited this key (HTTP 429).", retry_after)
        if 500 <= resp.status_code < 600:
            raise TransientProviderError(f"{self.name} returned HTTP {resp.status_code}.")
        if resp.status_code == 400 and "reasoning_effort" in payload and "reasoning" in resp.text.lower():
            # This provider/model doesn't accept the parameter -- drop it and retry once.
            self._send_reasoning_effort = False
            return self._single_request(key, messages, model, stop, max_tokens, temperature, retries_so_far)
        if resp.status_code != 200:
            # 4xx other than 429 is not retryable -- bad request, bad key, etc.
            raise LLMProviderError(
                f"{self.name} returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
            if client_side_stop:
                text = _truncate_at_stop(text, stop)
        except (KeyError, IndexError) as e:
            raise LLMProviderError(f"Unexpected response shape from {self.name}: {data}") from e

        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            request_time_ms=elapsed_ms,
            api_url=self.base_url,
            model_name=model,
            retries=retries_so_far,
            raw=data,
        )


# ---------------------------------------------------------------------------
# Multi-provider fallback routing
# ---------------------------------------------------------------------------


@dataclass
class RoutedProvider:
    provider: LLMProvider
    model: str  # the model name to request from this specific provider


class ProviderRouter:
    """Tries providers in order; falls back to the next one when a
    provider's keys are all exhausted. This is what the agent loop
    actually calls -- it has zero awareness of Groq vs OpenRouter vs
    anything else."""

    def __init__(self, routed_providers: List[RoutedProvider]):
        if not routed_providers:
            raise ValueError("ProviderRouter requires at least one provider.")
        self._routed = routed_providers

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> LLMResponse:
        errors = []
        for routed in self._routed:
            try:
                return routed.provider.generate(
                    messages=messages,
                    model=routed.model,
                    stop=stop,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except AllKeysExhaustedError as e:
                errors.append(f"{routed.provider.name}: {e}")
                continue
            except LLMProviderError as e:
                # Non-retryable provider-side error (bad model name, bad
                # request shape, etc.) -- also fall through to the next
                # provider rather than failing the whole task outright.
                errors.append(f"{routed.provider.name}: {e}")
                continue

        raise AllProvidersExhaustedError(
            "All configured providers failed:\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class ProviderConfigEntry(BaseModel):
    """One provider's config, as it appears in providers.json.
    api_key_env_vars lists ENVIRONMENT VARIABLE NAMES, never raw keys --
    e.g. ["GROQ_KEY_1", "GROQ_KEY_2"], resolved via os.environ at load time.
    """

    name: str
    base_url: str
    model: str
    api_key_env_vars: List[str]
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 60.0
    max_retries_per_key: int = 2
    cooldown_seconds: float = 60.0


class RouterConfig(BaseModel):
    providers: List[ProviderConfigEntry]


def load_provider_router(config_path: str) -> ProviderRouter:
    """Build a ProviderRouter from a JSON config file. Provider order in
    the file is fallback priority order (first = tried first).

    Raises ValueError if a referenced environment variable is unset --
    fail fast at startup rather than mysteriously mid-task.
    """
    with open(config_path) as f:
        raw = json.load(f)
    config = RouterConfig(**raw)

    routed: List[RoutedProvider] = []
    for entry in config.providers:
        keys = []
        for env_var in entry.api_key_env_vars:
            value = os.environ.get(env_var)
            if not value:
                raise ValueError(
                    f"Provider '{entry.name}' references env var '{env_var}', "
                    f"which is not set. Check your .env file."
                )
            keys.append(value)

        provider = OpenAICompatibleProvider(
            name=entry.name,
            base_url=entry.base_url,
            api_keys=keys,
            extra_headers=entry.extra_headers,
            timeout_seconds=entry.timeout_seconds,
            max_retries_per_key=entry.max_retries_per_key,
            cooldown_seconds=entry.cooldown_seconds,
        )
        routed.append(RoutedProvider(provider=provider, model=entry.model))

    return ProviderRouter(routed)