import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "server.py"


def load_server(tmp_path, monkeypatch):
    for key in [
        "LEGION_COMMAND_BUS_ENABLED",
        "COMMAND_TRANSPORT",
        "LEGION_COMMAND_SOURCE",
        "LEGION_REPORT_TRANSPORT",
        "WORKER_MODE",
        "LEGION_WORKER_MODE",
        "TELEGRAM_GATEWAY_ENABLED",
        "TELEGRAM_BOT_TOKEN",
        "DATABASE_URL",
        "POSTGRES_SCHEMA",
        "LEGION_POSTGRES_SCHEMA",
        "DATABASE_SCHEMA",
        "REDIS_URL",
        "REDIS_KEY_PREFIX",
        "LEGION_REDIS_PREFIX",
        "R2_PREFIX",
        "R2_BUCKET",
        "R2_BUCKET_NAME",
        "CLOUDFLARE_R2_BUCKET",
        "S3_BUCKET",
        "R2_ENDPOINT_URL",
        "R2_ENDPOINT",
        "S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "R2_ACCOUNT_ID",
        "CLOUDFLARE_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
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
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("POSTGRES_SCHEMA", "legion_2_dev")
    monkeypatch.setenv("REDIS_KEY_PREFIX", "legion_2_dev")
    monkeypatch.setenv("LEGION_AGENT_ID", "2군단-개발팀")

    status = server.legion_bus.status()

    assert server.legion_bus.should_start() is True
    assert status["transport"] == "http_redis_postgres"
    assert status["postgres_schema"] == "legion_2_dev"
    assert status["redis_key_prefix"] == "legion_2_dev"
    assert status["agent_id"] == "2군단-개발팀"
    assert status["transport_configured"] is True


def test_legion_bus_does_not_start_for_http_only_without_database(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("COMMAND_TRANSPORT", "http")
    monkeypatch.setenv("WORKER_MODE", "true")

    status = server.legion_bus.status()

    assert server.legion_bus.should_start() is False
    assert status["transport_configured"] is True
    assert status["database_configured"] is False


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


def test_artifact_storage_does_not_fake_r2_key_without_r2_config(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("POSTGRES_SCHEMA", "legion_2_dev")

    r2_key, size_bytes, metadata = server.legion_bus._store_artifact_payload(
        "artifact-1",
        {"work_order_id": "wo-1", "task_id": "task-1"},
        server.TASK_DONE_STATE,
        "작업 완료",
        None,
    )

    assert r2_key == ""
    assert size_bytes > 0
    assert metadata["storage"] == "postgres_inline"
    assert metadata["output"] == "작업 완료"


def test_r2_config_accepts_r2_endpoint_env_name(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("R2_ENDPOINT", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    cfg = server.legion_bus._r2_config()

    assert cfg["endpoint_url"] == "https://example.r2.cloudflarestorage.com"
    assert cfg["bucket"] == "bucket"


class DummyRequest:
    def __init__(self, headers=None, cookies=None):
        self.headers = headers or {}
        self.cookies = cookies or {}


def test_legion_status_requires_auth(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)

    response = asyncio.run(server.api_legion_status(DummyRequest()))

    assert response.status_code == 401
    assert json.loads(response.body)["error"] == "Unauthorized"


def test_legion_status_accepts_bearer_token(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("COMMAND_API_TOKEN", "shared-secret")

    response = asyncio.run(server.api_legion_status(DummyRequest(headers={"authorization": "Bearer shared-secret"})))

    assert response.status_code == 200
    body = json.loads(response.body)
    assert "agent_id" in body
    assert "last_error" in body


def test_health_omits_private_command_bus_details(tmp_path, monkeypatch):
    server = load_server(tmp_path, monkeypatch)
    monkeypatch.setenv("COMMAND_TRANSPORT", "http_redis_postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("POSTGRES_SCHEMA", "legion_private_schema")
    server.legion_bus.last_error = "private failure detail"
    server.legion_bus.last_task_id = "task-secret"

    response = asyncio.run(server.route_health(DummyRequest()))
    body = json.loads(response.body)
    command_bus = body["command_bus"]

    assert response.status_code == 200
    assert "postgres_schema" not in command_bus
    assert "redis_key_prefix" not in command_bus
    assert "agent_id" not in command_bus
    assert "last_error" not in command_bus
    assert "last_task_id" not in command_bus
    assert command_bus["database_configured"] is True
    assert command_bus["redis_configured"] is True
