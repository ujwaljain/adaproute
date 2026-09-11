"""Translate runner settings for OpenRouter or a Chat Completions API."""

import os
import re
from urllib.parse import urlsplit

OPENROUTER = {"type": "openrouter", "base_url": "https://openrouter.ai/api/v1",
              "api_key_env": "OPENROUTER_API_KEY"}


def connection(config):
    """Return API settings, retaining the default for older OpenRouter runs."""
    return config.get("api", OPENROUTER)


def validate_api(config):
    """Validate protocol, HTTPS endpoint, and credential environment variable.

    Configuration stores the name of an environment variable, never a key.
    OpenRouter uses its fixed endpoint; chat_completions accepts another
    HTTPS service implementing the same request and response format.
    """
    api = connection(config)
    if not isinstance(api, dict) or set(api) != {"type", "base_url", "api_key_env"}:
        raise ValueError("API needs type, base_url, and api_key_env")
    if api["type"] not in ("openrouter", "chat_completions"):
        raise ValueError("API type must be openrouter or chat_completions")
    if not isinstance(api["base_url"], str):
        raise ValueError("API base_url must be an HTTPS URL")
    url = urlsplit(api["base_url"])
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or any(c.isspace() for c in api["base_url"])):
        raise ValueError("API base_url must be HTTPS without credentials, query, or fragment")
    if api["type"] == "openrouter" and api["base_url"].rstrip("/") != OPENROUTER["base_url"]:
        raise ValueError("OpenRouter requires its official API endpoint")
    if not isinstance(api["api_key_env"], str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api["api_key_env"]):
        raise ValueError("api_key_env must name an environment variable, not contain a key")


def api_key(config):
    """Read the selected API's key from the environment or raise ValueError."""
    name = connection(config)["api_key_env"]
    key = os.environ.get(name, "").strip()
    if not key:
        raise ValueError(f"Set {name} in the environment")
    return key


def validate_model(config, model):
    """Check model identifiers, routing, and reasoning for the chosen protocol."""
    if connection(config)["type"] == "openrouter":
        if not re.fullmatch(r"[\w.-]+/[\w.:-]+", model["id"]):
            raise ValueError("Expected an explicit OpenRouter model ID")
        if not isinstance(model.get("provider"), str) or not model["provider"].strip():
            raise ValueError("OpenRouter models need an explicit provider slug")
    else:
        if "provider" in model:
            raise ValueError("Provider routing is only supported by OpenRouter")
        reasoning = model.get("reasoning")
        if reasoning is not None and "max_tokens" in reasoning:
            raise ValueError("Chat Completions uses reasoning effort, not a reasoning token budget")


def build_request(config, model, messages):
    """Translate common prompts and settings into the selected API's body."""
    body = {"model": model["id"], "messages": messages, "stream": False,
            "max_tokens": config["max_tokens"]}
    reasoning = model.get("reasoning")
    if connection(config)["type"] == "openrouter":
        body["provider"] = {"only": [model["provider"]], "allow_fallbacks": False,
                            "require_parameters": True}
        if reasoning is not None:
            body["reasoning"] = reasoning
    elif reasoning is not None:
        body["reasoning_effort"] = reasoning["effort"]
    return body


def inspect_models(config, request_json):
    """Read model metadata without making generation requests.

    OpenRouter has public endpoint metadata for routing and parameter checks.
    Other Chat Completions services use authenticated GET /models/{id}; their
    metadata confirms the model ID (including a models/ resource prefix used
    by Google) but may not expose parameter support or
    token limits. Returned metadata is frozen in the run manifest.
    """
    api = connection(config)
    endpoints = {}
    key = api_key(config) if api["type"] == "chat_completions" else None
    for model in config["models"]:
        path = "/models/" + model["id"]
        if api["type"] == "chat_completions":
            metadata = request_json(path, key=key, base_url=api["base_url"])
            if metadata.get("id") not in (model["id"], "models/" + model["id"]):
                raise ValueError(f"Model metadata identity mismatch: {model['id']}")
            endpoints[model["id"]] = metadata
            continue
        data = request_json(path + "/endpoints", base_url=api["base_url"])
        matches = [item for item in data["data"]["endpoints"] if item["tag"] == model["provider"]]
        if len(matches) != 1:
            raise ValueError(f"Provider unavailable or ambiguous: {model['id']} / {model['provider']}")
        endpoint = matches[0]
        required = {"max_tokens"} | ({"reasoning"} if model.get("reasoning") is not None else set())
        if not required <= set(endpoint["supported_parameters"]):
            raise ValueError(f"Provider does not support the requested parameters: {model['id']}")
        limit = endpoint.get("max_completion_tokens")
        if limit is not None and limit < config["max_tokens"]:
            raise ValueError(f"Provider completion limit too low: {model['id']}")
        if not isinstance(endpoint.get("provider_name"), str) or not endpoint["provider_name"]:
            raise ValueError("Missing provider identity")
        endpoints[model["id"]] = endpoint
    return endpoints
