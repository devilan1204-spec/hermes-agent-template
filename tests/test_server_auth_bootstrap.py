import base64
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"


def load_server(tmp_path, monkeypatch):
    for key in [
        "HERMES_AUTH_JSON_BOOTSTRAP",
        "HERMES_AUTH_JSON_B64",
        "HERMES_AUTH_PROVIDER",
        "HERMES_AUTH_MODEL",
        "HERMES_AUTH_BOOTSTRAP_MODE",
        "LLM_MODEL",
        "TELEGRAM_BOT_TOKEN",
        "WORKER_MODE",
        "GATEWAY_ENABLED",
        "TELEGRAM_GATEWAY_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    spec = importlib.util.spec_from_file_location("server_auth_bootstrap_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["server_auth_bootstrap_under_test"] = module
    spec.loader.exec_module(module)
    return module


def auth_payload(provider="openai-codex", token="refresh-test"):
    return {
        "version": 2,
        "active_provider": provider,
        "providers": {
            provider: {
                "tokens": {"refresh_token": token, "access_token": "access-test"},
                "auth_mode": "oauth_device",
            }
        },
    }


def test_auth_bootstrap_writes_auth_json_and_configures_provider_model(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    payload = auth_payload()
    env = {
        "HERMES_AUTH_JSON_BOOTSTRAP": json.dumps(payload),
        "HERMES_AUTH_PROVIDER": "openai-codex",
        "HERMES_AUTH_MODEL": "gpt-5.5",
    }

    result = server._apply_auth_json_bootstrap(env)

    assert result["applied"] is True
    auth_path = tmp_path / ".hermes" / "auth.json"
    stored = json.loads(auth_path.read_text())
    assert stored["providers"]["openai-codex"]["tokens"]["refresh_token"] == "refresh-test"
    assert oct(auth_path.stat().st_mode & 0o777) == "0o600"
    config = yaml.safe_load((tmp_path / ".hermes" / "config.yaml").read_text())
    assert config["model"]["provider"] == "openai-codex"
    assert config["model"]["default"] == "gpt-5.5"
    assert server.is_config_complete(env) is True


def test_auth_bootstrap_accepts_base64_payload(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    encoded = base64.b64encode(json.dumps(auth_payload("qwen-oauth")).encode()).decode()

    result = server._apply_auth_json_bootstrap({
        "HERMES_AUTH_JSON_B64": encoded,
        "HERMES_AUTH_PROVIDER": "qwen-oauth",
        "HERMES_AUTH_MODEL": "qwen3-coder-plus",
    })

    assert result["applied"] is True
    stored = json.loads((tmp_path / ".hermes" / "auth.json").read_text())
    assert "qwen-oauth" in stored["providers"]
    config = yaml.safe_load((tmp_path / ".hermes" / "config.yaml").read_text())
    assert config["model"]["provider"] == "qwen-oauth"


def test_auth_bootstrap_merge_preserves_existing_provider(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    auth_path = tmp_path / ".hermes" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(auth_payload("xai-oauth", "old-refresh")))

    result = server._apply_auth_json_bootstrap({
        "HERMES_AUTH_JSON_BOOTSTRAP": json.dumps(auth_payload("openai-codex", "new-refresh")),
        "HERMES_AUTH_PROVIDER": "openai-codex",
        "HERMES_AUTH_MODEL": "gpt-5.5",
    })

    assert result["applied"] is True
    stored = json.loads(auth_path.read_text())
    assert stored["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "old-refresh"
    assert stored["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"


def test_auth_bootstrap_missing_mode_does_not_overwrite_existing(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    auth_path = tmp_path / ".hermes" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(auth_payload("openai-codex", "keep-refresh")))

    result = server._apply_auth_json_bootstrap({
        "HERMES_AUTH_BOOTSTRAP_MODE": "missing",
        "HERMES_AUTH_JSON_BOOTSTRAP": json.dumps(auth_payload("openai-codex", "replace-refresh")),
        "HERMES_AUTH_PROVIDER": "openai-codex",
        "HERMES_AUTH_MODEL": "gpt-5.5",
    })

    assert result == {"applied": False, "reason": "auth_json_exists"}
    stored = json.loads(auth_path.read_text())
    assert stored["providers"]["openai-codex"]["tokens"]["refresh_token"] == "keep-refresh"
