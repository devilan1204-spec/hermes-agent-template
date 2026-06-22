import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"


def load_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    spec = importlib.util.spec_from_file_location("server_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["server_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_worker_mode_disables_gateway_readiness(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("WORKER_MODE", "true")

    assert server.gateway_enabled() is False
    assert server.is_config_complete({"LLM_MODEL": "gpt-5.5", "ANTHROPIC_API_KEY": "sk-ant-test"}) is False


def test_gateway_enabled_false_disables_gateway_readiness(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("GATEWAY_ENABLED", "false")

    assert server.gateway_enabled() is False
    assert server.is_config_complete({"LLM_MODEL": "gpt-5.5", "ANTHROPIC_API_KEY": "sk-ant-test"}) is False


def test_telegram_gateway_enabled_false_disables_gateway_readiness(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("TELEGRAM_GATEWAY_ENABLED", "false")

    assert server.gateway_enabled() is False
    assert server.is_config_complete({"LLM_MODEL": "gpt-5.5", "ANTHROPIC_API_KEY": "sk-ant-test"}) is False
