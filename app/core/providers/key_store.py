import json
import os
from cryptography.fernet import Fernet
from app.core.config import BASE_DIR

KEYS_FILE = os.path.join(BASE_DIR, "provider_keys.json")
KEY_FILE = os.path.join(BASE_DIR, ".secret_key")

DEFAULT_PROVIDERS = {
    "openai": {"type": "openai", "base_url": None, "options": {}},
    "anthropic": {"type": "anthropic", "base_url": None, "options": {}},
    "gemini": {"type": "gemini", "base_url": None, "options": {}},
}


def _get_cipher() -> Fernet:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    return Fernet(key)


def _encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    return _get_cipher().encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    try:
        return _get_cipher().decrypt(ciphertext.encode()).decode()
    except Exception:
        return ciphertext


def _load():
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            return json.load(f)
    return {"providers": {}, "model_map": {}}


def _save(data):
    with open(KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_provider(provider: str, provider_type: str, api_key: str, base_url: str = None):
    data = _load()
    providers = data.setdefault("providers", {})
    providers[provider] = {
        "type": provider_type,
        "api_key": _encrypt(api_key) if api_key else "",
        "base_url": base_url,
        "models": providers.get(provider, {}).get("models", []),
    }
    _save(data)


def remove_provider(provider: str) -> bool:
    data = _load()
    providers = data.get("providers", {})
    if provider not in providers:
        return False
    model_map = data.get("model_map", {})
    for model in providers[provider].get("models", []):
        model_map.pop(model, None)
    del providers[provider]
    _save(data)
    return True


def get_provider_info(provider: str) -> dict | None:
    data = _load()
    return data.get("providers", {}).get(provider)


def get_key(provider: str) -> str | None:
    info = get_provider_info(provider)
    if not info:
        return None
    return _decrypt(info.get("api_key", "")) or None


def get_base_url(provider: str) -> str | None:
    info = get_provider_info(provider)
    return info.get("base_url") if info else None


def get_provider_type(provider: str) -> str | None:
    info = get_provider_info(provider)
    return info.get("type") if info else None


def save_key(provider: str, model_id: str, api_key: str = None):
    data = _load()
    providers = data.setdefault("providers", {})
    if provider not in providers:
        defaults = DEFAULT_PROVIDERS.get(provider, {"type": "openai_compatible", "base_url": None})
        providers[provider] = {**defaults, "api_key": _encrypt(api_key) if api_key else "", "models": []}
    else:
        if api_key:
            providers[provider]["api_key"] = _encrypt(api_key)
    if model_id not in providers[provider]["models"]:
        providers[provider]["models"].append(model_id)
    data.setdefault("model_map", {})[model_id] = provider
    _save(data)


def get_provider_for_model(model_id: str) -> str | None:
    data = _load()
    return data.get("model_map", {}).get(model_id)


def list_cloud_models() -> list[dict]:
    data = _load()
    result = []
    for provider, info in data.get("providers", {}).items():
        for m in info.get("models", []):
            result.append({"model": m, "provider": provider})
    return result


def remove_model(model_id: str) -> bool:
    data = _load()
    model_map = data.get("model_map", {})
    provider = model_map.pop(model_id, None)
    if provider:
        providers = data.get("providers", {})
        if provider in providers:
            models = providers[provider]["models"]
            if model_id in models:
                models.remove(model_id)
        _save(data)
        return True
    return False


def list_providers() -> list[dict]:
    data = _load()
    providers = data.get("providers", {})
    return [{"name": k, "type": v.get("type", "unknown"), "base_url": v.get("base_url"), "models": v.get("models", [])} for k, v in providers.items()] + [{"name": "ollama", "type": "ollama", "base_url": None, "models": []}]


def detect_models(provider_type: str, api_key: str, base_url: str = None) -> list[str]:
    try:
        if provider_type == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            return [m.id for m in client.models.list() if m.id.startswith(("gpt-", "o"))]

        elif provider_type == "anthropic":
            import httpx
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            r = httpx.get("https://api.anthropic.com/v1/models", headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]

        elif provider_type == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            models = client.models.list()
            return [m.name.replace("models/", "") for m in models
                    if m.supported_actions and "generateContent" in m.supported_actions]

        elif provider_type == "openai_compatible":
            if not base_url:
                return []
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            return [m.id for m in client.models.list()]

    except Exception as e:
        raise Exception(f"Model detection failed: {e}")
    return []
