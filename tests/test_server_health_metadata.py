import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"


def load_server(tmp_path, monkeypatch):
    for key in [
        "HERMES_AUTH_PROVIDER",
        "HERMES_AUTH_MODEL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "HERMES_PROVIDER",
        "HERMES_MODEL",
        "GATEWAY_ENABLED",
        "WORKER_MODE",
        "LEGION_WORKER_MODE",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    spec = importlib.util.spec_from_file_location("server_health_metadata_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["server_health_metadata_under_test"] = module
    spec.loader.exec_module(module)
    return module


def decode_json_response(response):
    return json.loads(response.body.decode("utf-8"))


def test_health_exposes_configured_provider_model_and_openai_oauth_identity(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openai-codex", "default": "gpt-5.5"},
    }))
    (hermes_home / "auth.json").write_text(json.dumps({
        "providers": {
            "openai-codex": {
                "account": {"email": "devilan1204@gmail.com"},
                "tokens": {"refresh_token": "must-not-leak"},
            }
        },
        "credential_pool": {
            "openai-codex": [{
                "email": "pool@example.com",
                "tokens": {"refresh_token": "also-must-not-leak"},
            }]
        },
    }))

    response = asyncio.run(server.route_health(None))
    payload = decode_json_response(response)

    assert payload["provider"] == "openai-codex"
    assert payload["model"] == "gpt-5.5"
    assert payload["main_provider"] == "openai-codex"
    assert payload["main_model"] == "gpt-5.5"
    assert payload["system_gateway_status"] == payload["gateway"]
    assert payload["openai_oauth_google_id"] == "devilan1204@gmail.com"
    assert "refresh_token" not in json.dumps(payload)
    assert "must-not-leak" not in json.dumps(payload)


def test_health_metadata_falls_back_to_env_without_leaking_auth_tokens(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_AUTH_PROVIDER", "openai-codex")
    monkeypatch.setenv("HERMES_AUTH_MODEL", "gpt-5.5")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "providers": {"openai-codex": {"tokens": {"refresh_token": "secret-refresh"}}}
    }))

    payload = decode_json_response(asyncio.run(server.route_health(None)))

    assert payload["provider"] == "openai-codex"
    assert payload["model"] == "gpt-5.5"
    assert payload["openai_oauth_google_id"] is None
    assert "secret-refresh" not in json.dumps(payload)
