import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"


def load_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    spec = importlib.util.spec_from_file_location("server_legion_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["server_legion_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_legion_bus_stays_disabled_without_transport(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.delenv("COMMAND_TRANSPORT", raising=False)
    monkeypatch.delenv("LEGION_COMMAND_SOURCE", raising=False)

    assert server.legion_bus.should_start() is False
    assert server.legion_bus.status()["enabled"] is False


def test_legion_bus_enables_for_worker_http_redis_postgres(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("COMMAND_TRANSPORT", "http_redis_postgres")
    monkeypatch.setenv("WORKER_MODE", "true")
    monkeypatch.setenv("POSTGRES_SCHEMA", "legion_2_dev")
    monkeypatch.setenv("REDIS_KEY_PREFIX", "legion_2_dev")
    monkeypatch.setenv("LEGION_AGENT_ID", "2군단-개발팀")

    status = server.legion_bus.status()

    assert server.legion_bus.should_start() is True
    assert status["transport"] == "http_redis_postgres"
    assert status["postgres_schema"] == "legion_2_dev"
    assert status["redis_key_prefix"] == "legion_2_dev"
    assert status["agent_id"] == "2군단-개발팀"


@pytest.mark.parametrize("identifier", ["legion_2_dev", "legion_4_creative_worker", "abc_123"])
def test_quote_ident_accepts_safe_schema_names(tmp_path, monkeypatch, identifier):
    server = load_server(tmp_path, monkeypatch)

    assert server._quote_ident(identifier) == f'"{identifier}"'


@pytest.mark.parametrize("identifier", ["legion-2", "legion_2;drop", "2legion", "", "public.foo"])
def test_quote_ident_rejects_unsafe_schema_names(tmp_path, monkeypatch, identifier):
    server = load_server(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        server._quote_ident(identifier)


def test_task_prompt_contains_instruction_and_worker_context(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("LEGION_ID", "legion-2")
    monkeypatch.setenv("LEGION_AGENT_ID", "dev-worker")

    prompt = server._task_prompt({
        "work_order_id": "wo-1",
        "task_id": "task-1",
        "task_type": "smoke",
        "instruction": "상태 확인해",
    })

    assert "Legion: legion-2" in prompt
    assert "Agent: dev-worker" in prompt
    assert "Task: task-1" in prompt
    assert "상태 확인해" in prompt
