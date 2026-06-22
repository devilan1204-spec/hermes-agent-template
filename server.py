"""
Hermes Agent — Railway admin server.

Responsibilities:
  - Admin UI / setup wizard at /setup (Starlette + Jinja, cookie-auth guarded)
  - Management API at /setup/api/* (config, status, logs, gateway, pairing)
  - Reverse proxy at / and /* → native Hermes dashboard (hermes_cli/web_server, on 127.0.0.1:9119)
  - Managed subprocesses: `hermes gateway` (agent) and `hermes dashboard` (native UI)
  - Cookie-based session auth at /login (HMAC-signed, 7-day expiry, httponly)

Auth model: Basic Auth was dropped in favor of cookies because the Hermes React
SPA's plain fetch() calls do not reliably include basic-auth creds across browsers,
and basic-auth's per-directory protection space forced separate prompts for
/setup and /. Cookies auto-include on every same-origin request, so both the
setup UI and the proxied dashboard work with a single login. The cookie signing
secret is regenerated on every process start, so any ADMIN_PASSWORD change on
Railway (which triggers a redeploy) invalidates all existing sessions.

First-visit behavior: if no provider+model config exists, GET / redirects to /setup.
Once configured, / proxies to the Hermes dashboard. A small "← Setup" widget is
injected into every proxied HTML response so users can always return to the wizard.
"""

# PEP 563 lazy annotations: keeps function/parameter type hints as strings so
# they're never evaluated at import. Avoids the startup DeprecationWarning from
# annotating against websockets.WebSocketClientProtocol (renamed in websockets
# >= 14), and is forward-compatible regardless of the installed websockets
# version. Safe here — nothing in this module introspects annotations at runtime.
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
import signal
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
import websockets.exceptions
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route, WebSocketRoute
from starlette.templating import Jinja2Templates
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
ENV_FILE = Path(HERMES_HOME) / ".env"
PAIRING_DIR = Path(HERMES_HOME) / "pairing"
PAIRING_TTL = 3600

# Native Hermes dashboard — runs on loopback, fronted by our reverse proxy.
HERMES_DASHBOARD_HOST = "127.0.0.1"
HERMES_DASHBOARD_PORT = int(os.environ.get("HERMES_DASHBOARD_PORT", "9119"))
HERMES_DASHBOARD_URL = f"http://{HERMES_DASHBOARD_HOST}:{HERMES_DASHBOARD_PORT}"
# Process-efficiency mode for Railway/Telegram-first deployments: the native
# browser dashboard (and its embedded TUI/node sidecars) is useful for admin, but
# it is not required for Telegram message handling. Start it lazily on first web
# request by default instead of burning ~400MB RSS 24/7. Set
# HERMES_DASHBOARD_AUTOSTART=1 to restore eager boot behavior.
HERMES_DASHBOARD_AUTOSTART = os.environ.get("HERMES_DASHBOARD_AUTOSTART", "0").lower() in ("1", "true", "yes", "on")

# Mirror dashboard-ref-only/auth_proxy.py: strip only `host` (httpx sets it)
# and `transfer-encoding` (httpx recomputes it from the body). Keep everything
# else — notably `authorization`, because the SPA uses Bearer tokens against
# hermes's own /api/env/reveal and OAuth endpoints, and keep `cookie` since
# some hermes endpoints read it. Aggressive stripping was masking requests in
# ways that produced spurious 401s.
HOP_BY_HOP = {"host", "transfer-encoding"}

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"[server] Admin credentials — username: {ADMIN_USERNAME}  password: {ADMIN_PASSWORD}", flush=True)
else:
    print(f"[server] Admin username: {ADMIN_USERNAME}", flush=True)

# ── Env var registry ──────────────────────────────────────────────────────────
# (key, label, category, is_secret)
ENV_VARS = [
    ("LLM_MODEL",               "Model",                    "model",     False),
    ("HERMES_AUTH_PROVIDER",     "OAuth Provider",           "model",     False),
    ("HERMES_AUTH_MODEL",        "OAuth Model",              "model",     False),
    ("HERMES_AUTH_BOOTSTRAP_MODE", "Auth Bootstrap Mode",     "model",     False),
    ("HERMES_AUTH_JSON_BOOTSTRAP", "Auth JSON Bootstrap",     "auth",      True),
    ("HERMES_AUTH_JSON_B64",     "Auth JSON Bootstrap (B64)", "auth",      True),
    ("OPENROUTER_API_KEY",       "OpenRouter",               "provider",  True),
    ("DEEPSEEK_API_KEY",         "DeepSeek",                 "provider",  True),
    ("DASHSCOPE_API_KEY",        "Qwen Cloud (DashScope)",   "provider",  True),
    ("GLM_API_KEY",              "GLM / Z.AI",               "provider",  True),
    ("KIMI_API_KEY",             "Kimi",                     "provider",  True),
    ("MINIMAX_API_KEY",          "MiniMax",                  "provider",  True),
    ("HF_TOKEN",                 "Hugging Face",             "provider",  True),
    # Added in v2026.4.23+ (hermes v0.11.0+). All plain API-key auth — hermes
    # auto-routes by env-var presence, no extra config needed on our side.
    # OAuth-based providers (xAI Grok SuperGrok, Gemini CLI, Qwen OAuth, Claude Code)
    # are set up via the dashboard's Keys tab or HERMES_AUTH_JSON_BOOTSTRAP.
    ("NVIDIA_API_KEY",           "NVIDIA NIM",               "provider",  True),
    ("ARCEEAI_API_KEY",          "Arcee AI",                 "provider",  True),
    ("STEPFUN_API_KEY",          "Step Plan",                "provider",  True),
    ("GEMINI_API_KEY",           "Google AI Studio",         "provider",  True),
    ("NOVITA_API_KEY",           "NovitaAI",                 "provider",  True),
    ("FIREWORKS_API_KEY",        "Fireworks AI",             "provider",  True),
    ("ANTHROPIC_API_KEY",        "Anthropic (Claude)",       "provider",  True),
    ("XAI_API_KEY",              "xAI",                      "provider",  True),
    ("AWS_ACCESS_KEY_ID",        "AWS Access Key ID",        "provider",  True),
    ("AWS_SECRET_ACCESS_KEY",    "AWS Secret Access Key",    "bedrock",   True),
    ("AWS_DEFAULT_REGION",       "AWS Region",               "bedrock",   False),
    ("COPILOT_GITHUB_TOKEN",     "GitHub Copilot",           "provider",  True),
    ("GMI_API_KEY",              "GMI Cloud",                "provider",  True),
    ("OPENCODE_ZEN_API_KEY",     "OpenCode Zen",             "provider",  True),
    ("OPENCODE_GO_API_KEY",      "OpenCode Go",              "provider",  True),
    ("KILOCODE_API_KEY",         "Kilo Code",                "provider",  True),
    ("OLLAMA_API_KEY",           "Ollama Cloud",             "provider",  True),
    ("AZURE_FOUNDRY_API_KEY",    "Azure Foundry key",        "provider",  True),
    ("AZURE_FOUNDRY_BASE_URL",   "Azure Foundry URL",        "azure",     False),
    # Custom OpenAI-compatible endpoint — one slot; more via Hermes dashboard.
    # Only the API key is in category "provider" so PROVIDER_KEYS / is_config_complete
    # only trigger when an actual key is present, not just a base URL.
    ("CUSTOM_PROVIDER_API_KEY",  "Custom Provider key",      "provider",  True),
    ("CUSTOM_PROVIDER_BASE_URL", "Custom Provider base URL", "custom",    False),
    ("CUSTOM_PROVIDER_NAME",     "Custom Provider name",     "custom",    False),
    ("PARALLEL_API_KEY",         "Parallel (search)",        "tool",      True),
    ("FIRECRAWL_API_KEY",        "Firecrawl (scrape)",       "tool",      True),
    ("TAVILY_API_KEY",           "Tavily (search)",          "tool",      True),
    ("FAL_KEY",                  "FAL (image gen)",          "tool",      True),
    ("BROWSERBASE_API_KEY",      "Browserbase key",          "tool",      True),
    ("BROWSERBASE_PROJECT_ID",   "Browserbase project",      "tool",      False),
    ("GITHUB_TOKEN",             "GitHub token",             "tool",      True),
    ("VOICE_TOOLS_OPENAI_KEY",   "OpenAI (voice/TTS)",       "tool",      True),
    ("HONCHO_API_KEY",           "Honcho (memory)",          "tool",      True),
    ("TELEGRAM_BOT_TOKEN",       "Bot Token",                "telegram",  True),
    ("TELEGRAM_ALLOWED_USERS",   "Allowed User IDs",         "telegram",  False),
    ("DISCORD_BOT_TOKEN",        "Bot Token",                "discord",   True),
    ("DISCORD_ALLOWED_USERS",    "Allowed User IDs",         "discord",   False),
    ("SLACK_BOT_TOKEN",          "Bot Token (xoxb-...)",     "slack",     True),
    ("SLACK_APP_TOKEN",          "App Token (xapp-...)",     "slack",     True),
    ("WHATSAPP_ENABLED",         "Enable WhatsApp",          "whatsapp",  False),
    ("EMAIL_ADDRESS",            "Email Address",            "email",     False),
    ("EMAIL_PASSWORD",           "Email Password",           "email",     True),
    ("EMAIL_IMAP_HOST",          "IMAP Host",                "email",     False),
    ("EMAIL_SMTP_HOST",          "SMTP Host",                "email",     False),
    ("MATTERMOST_URL",           "Server URL",               "mattermost",False),
    ("MATTERMOST_TOKEN",         "Bot Token",                "mattermost",True),
    ("MATRIX_HOMESERVER",        "Homeserver URL",           "matrix",    False),
    ("MATRIX_ACCESS_TOKEN",      "Access Token",             "matrix",    True),
    ("MATRIX_USER_ID",           "User ID",                  "matrix",    False),
    ("GATEWAY_ALLOW_ALL_USERS",  "Allow all users",          "gateway",   False),
    ("ADMIN_USERNAME",           "Admin username",           "admin",     False),
    ("ADMIN_PASSWORD",           "Admin password",           "admin",     True),
]

SECRET_KEYS  = {k for k, _, _, s in ENV_VARS if s}
PROVIDER_KEYS = [k for k, _, c, _ in ENV_VARS if c == "provider"]
CHANNEL_MAP  = {
    "Telegram":    "TELEGRAM_BOT_TOKEN",
    "Discord":     "DISCORD_BOT_TOKEN",
    "Slack":       "SLACK_BOT_TOKEN",
    "WhatsApp":    "WHATSAPP_ENABLED",
    "Email":       "EMAIL_ADDRESS",
    "Mattermost":  "MATTERMOST_TOKEN",
    "Matrix":      "MATRIX_ACCESS_TOKEN",
}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def gateway_enabled() -> bool:
    """Whether this deployment should run `hermes gateway`."""
    if _truthy(os.environ.get("WORKER_MODE")):
        return False
    if _falsey(os.environ.get("GATEWAY_ENABLED")):
        return False
    if _falsey(os.environ.get("TELEGRAM_GATEWAY_ENABLED")):
        return False
    return True


# ── .env helpers ──────────────────────────────────────────────────────────────
def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def _effective_env() -> dict[str, str]:
    """Railway/runtime env plus persisted .env, with .env taking precedence."""
    env = dict(os.environ)
    env.update(read_env(ENV_FILE))
    return env


def _auth_path() -> Path:
    return Path(HERMES_HOME) / "auth.json"


def _write_auth_json(data: dict) -> None:
    auth_path = _auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        auth_path.chmod(0o600)
    except Exception:
        pass


def _decode_auth_json_bootstrap(env: dict[str, str] | None = None) -> dict:
    """Decode raw or base64 auth.json bootstrap payload from env."""
    env = env or _effective_env()
    raw = (env.get("HERMES_AUTH_JSON_BOOTSTRAP") or "").strip()
    raw_b64 = (env.get("HERMES_AUTH_JSON_B64") or "").strip()
    if raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Hermes auth bootstrap payload must be a JSON object")
    providers = data.get("providers")
    if providers is not None and not isinstance(providers, dict):
        raise ValueError("Hermes auth bootstrap providers must be a JSON object")
    return data


def _apply_oauth_provider_config(provider: str, model: str = "") -> None:
    """Persist model/provider selection for OAuth-backed providers."""
    if not provider:
        return
    import yaml

    config_path = Path(HERMES_HOME) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            pass
    merged = dict(existing)
    merged_model = dict(merged.get("model") if isinstance(merged.get("model"), dict) else {})
    if model:
        merged_model["default"] = model
    merged_model["provider"] = provider
    merged["model"] = merged_model
    merged.setdefault("data_dir", HERMES_HOME)
    with config_path.open("w") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)


def _apply_auth_json_bootstrap(env: dict[str, str] | None = None) -> dict:
    """Apply fleet auth bootstrap from Railway env.

    Env contract:
      - HERMES_AUTH_JSON_BOOTSTRAP: raw auth.json content, or
      - HERMES_AUTH_JSON_B64: base64-encoded auth.json content
      - HERMES_AUTH_BOOTSTRAP_MODE: missing|merge|replace|force (default: merge)
      - HERMES_AUTH_PROVIDER/HERMES_AUTH_MODEL: optional provider/model to persist
    """
    env = env or _effective_env()
    incoming = _decode_auth_json_bootstrap(env)
    if not incoming:
        return {"applied": False, "reason": "no_bootstrap"}

    mode = (env.get("HERMES_AUTH_BOOTSTRAP_MODE") or "merge").strip().lower()
    if mode == "force":
        mode = "replace"
    if mode not in {"missing", "merge", "replace"}:
        raise ValueError("HERMES_AUTH_BOOTSTRAP_MODE must be one of: missing, merge, replace, force")

    existing = _read_auth_json()
    if mode == "missing" and existing:
        provider = env.get("HERMES_AUTH_PROVIDER") or existing.get("active_provider") or incoming.get("active_provider") or ""
        model = env.get("HERMES_AUTH_MODEL") or env.get("LLM_MODEL") or ""
        if provider and _oauth_provider_has_tokens(provider):
            _apply_oauth_provider_config(provider, model)
        return {"applied": False, "reason": "auth_json_exists"}

    if mode == "replace" or not existing:
        merged = dict(incoming)
    else:
        merged = dict(existing)
        providers = dict(merged.get("providers") if isinstance(merged.get("providers"), dict) else {})
        providers.update(incoming.get("providers") if isinstance(incoming.get("providers"), dict) else {})
        if providers:
            merged["providers"] = providers
        for key in ("active_provider", "version"):
            if incoming.get(key) is not None:
                merged[key] = incoming[key]

    merged["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_auth_json(merged)

    provider = env.get("HERMES_AUTH_PROVIDER") or merged.get("active_provider") or incoming.get("active_provider") or ""
    model = env.get("HERMES_AUTH_MODEL") or env.get("LLM_MODEL") or ""
    if provider:
        _apply_oauth_provider_config(str(provider), str(model))
    return {"applied": True, "mode": mode, "provider": provider, "has_model": bool(model)}


def write_config_yaml(data: dict[str, str]) -> None:
    """Write config.yaml — deep-merge template defaults with any existing user/cron-managed sections.

    Previously this overwrote ``$HERMES_HOME/config.yaml`` with a hardcoded template
    body on every boot, silently erasing user-managed top-level keys. The most
    common casualty is ``mcp_servers`` — Hermes reads downstream MCP servers
    *only* from this file (see ``hermes_cli/mcp_config.py:_get_mcp_servers``), so
    the wipe broke ``hermes mcp add/test/list`` state across every container
    restart and required hand-restoration after each redeploy.

    The fix: load the existing file if any, apply the deployment-managed keys
    (``model.default``, ``model.provider``, ``terminal``, ``agent``, ``data_dir``)
    on top, and write the merged result. Unknown top-level keys (``mcp_servers``,
    custom skill config, etc.) are preserved verbatim.
    """
    import yaml  # hermes-agent already pulls pyyaml; deferred import keeps cold start light

    model = data.get("LLM_MODEL") or data.get("HERMES_AUTH_MODEL") or ""
    config_path = Path(HERMES_HOME) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if config_path.exists():
        try:
            with config_path.open() as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except (yaml.YAMLError, OSError):
            # Treat unparseable as absent — we'll overwrite with template defaults.
            existing = {}

    merged = dict(existing)

    # Deployment-managed (always authoritative — these reflect the runtime env).
    merged_model = dict(merged.get("model") if isinstance(merged.get("model"), dict) else {})
    # If LLM_MODEL is absent, preserve the dashboard-selected model from
    # config.yaml. OAuth-based setups (openai-codex/qwen/nous/xai) commonly do
    # not set LLM_MODEL in .env; overwriting model.default with "" on boot makes
    # the gateway look unconfigured after a redeploy.
    if model:
        merged_model["default"] = model
    elif not merged_model.get("default"):
        merged_model["default"] = ""
    # Only force provider="auto" when a known API key is configured. If no
    # API key is set, the user likely configured an OAuth provider (xai-oauth,
    # qwen-oauth, etc.) via the dashboard's model picker — preserve that value
    # so a container restart doesn't revert it to "auto" and break their session.
    # Explicit OAuth bootstrap provider wins when tokens are present. This is the
    # fleet-auth path used by Railway legions: auth.json is delivered through env,
    # while provider/model selection must be persisted in config.yaml for Hermes.
    auth_provider = str(data.get("HERMES_AUTH_PROVIDER") or "").strip()
    if auth_provider and _oauth_provider_has_tokens(auth_provider):
        merged_model["provider"] = auth_provider
    elif any(data.get(k) for k in PROVIDER_KEYS):
        merged_model["provider"] = "auto"
    merged["model"] = merged_model

    merged_terminal = dict(merged.get("terminal") if isinstance(merged.get("terminal"), dict) else {})
    merged_terminal["backend"] = "local"
    merged_terminal["timeout"] = 60
    merged_terminal["cwd"] = "/tmp"
    merged["terminal"] = merged_terminal

    merged_agent = dict(merged.get("agent") if isinstance(merged.get("agent"), dict) else {})
    merged_agent.setdefault("max_iterations", 50)
    merged["agent"] = merged_agent

    merged["data_dir"] = HERMES_HOME

    # Custom OpenAI-compatible endpoint — write custom_providers block when configured,
    # remove it when not (safe on Railway where users don't hand-edit config.yaml).
    custom_base_url = data.get("CUSTOM_PROVIDER_BASE_URL", "").strip()
    if custom_base_url:
        raw_name = data.get("CUSTOM_PROVIDER_NAME", "").strip() or custom_base_url
        # Sanitise to a valid hermes provider name (lowercase alphanumeric + hyphens).
        sanitized_name = re.sub(r"[^a-z0-9-]", "-", raw_name.lower()).strip("-") or "custom"
        merged["custom_providers"] = [{
            "name": sanitized_name,
            "base_url": custom_base_url,
            "key_env": "CUSTOM_PROVIDER_API_KEY",
        }]
    else:
        merged.pop("custom_providers", None)

    with config_path.open("w") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)


def write_env(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cat_order = ["model", "provider", "auth", "bedrock", "azure", "custom", "tool",
                 "telegram", "discord", "slack", "whatsapp",
                 "email", "mattermost", "matrix", "gateway", "admin"]
    cat_labels = {
        "model": "Model", "provider": "Providers", "auth": "Hermes Auth Bootstrap",
        "bedrock": "AWS Bedrock", "azure": "Azure Foundry",
        "custom": "Custom Endpoint", "tool": "Tools",
        "telegram": "Telegram", "discord": "Discord", "slack": "Slack",
        "whatsapp": "WhatsApp", "email": "Email",
        "mattermost": "Mattermost", "matrix": "Matrix", "gateway": "Gateway",
        "admin": "Admin",
    }
    key_cat = {k: c for k, _, c, _ in ENV_VARS}
    grouped: dict[str, list[str]] = {c: [] for c in cat_order}
    grouped["other"] = []

    for k, v in data.items():
        if not v:
            continue
        cat = key_cat.get(k, "other")
        grouped.setdefault(cat, []).append(f"{k}={v}")

    lines: list[str] = []
    for cat in cat_order:
        entries = sorted(grouped.get(cat, []))
        if entries:
            lines.append(f"# {cat_labels.get(cat, cat)}")
            lines.extend(entries)
            lines.append("")
    if grouped["other"]:
        lines.append("# Other")
        lines.extend(sorted(grouped["other"]))
        lines.append("")

    path.write_text("\n".join(lines))


# ── xAI Grok SuperGrok OAuth (Device Code — RFC 8628) ───────────────────────
# xAI's OIDC discovery at https://auth.x.ai/.well-known/openid-configuration
# declares device_authorization_endpoint, so Device Code flow works without
# any redirect URL. The client_id matches hermes's own Grok CLI credential.
_XAI_CLIENT_ID   = "b1a00492-073a-47ea-816f-4c329264a828"
_XAI_SCOPE       = "openid profile email offline_access grok-cli:access api:access"
_XAI_DEVICE_URL  = "https://auth.x.ai/oauth2/device/code"
_XAI_TOKEN_URL   = "https://auth.x.ai/oauth2/token"
_XAI_GRANT_TYPE  = "urn:ietf:params:oauth:grant-type:device_code"

_xai_oauth_state: dict | None = None  # one auth at a time (single-user deployment)


def _read_auth_json() -> dict:
    """Best-effort read of Hermes OAuth credential store."""
    auth_path = Path(HERMES_HOME) / "auth.json"
    if not auth_path.exists():
        return {}
    try:
        data = json.loads(auth_path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _oauth_provider_has_tokens(provider: str) -> bool:
    """True when auth.json contains usable-looking tokens for an OAuth provider.

    Railway deployments often use Hermes OAuth providers such as openai-codex,
    qwen-oauth, nous, or xai-oauth instead of API-key env vars. The setup
    server must treat those credentials as a configured provider; otherwise a
    redeploy sees "provider/model not configured" and never starts the gateway.
    """
    if not provider:
        return False
    entry = _read_auth_json().get("providers", {}).get(provider, {})
    if not isinstance(entry, dict):
        return False
    tokens = entry.get("tokens") if isinstance(entry.get("tokens"), dict) else entry
    return any(tokens.get(k) for k in ("refresh_token", "access_token", "id_token"))


def _has_xai_oauth_tokens() -> bool:
    """True when auth.json contains a valid xAI OAuth refresh token."""
    return _oauth_provider_has_tokens("xai-oauth")


def _configured_model_from_yaml() -> tuple[str, str]:
    """Return (model, provider) from config.yaml, if present."""
    try:
        import yaml
        config_path = Path(HERMES_HOME) / "config.yaml"
        loaded = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        model_cfg = loaded.get("model", {}) if isinstance(loaded, dict) else {}
        if not isinstance(model_cfg, dict):
            return "", ""
        return str(model_cfg.get("default") or ""), str(model_cfg.get("provider") or "")
    except Exception:
        return "", ""


def _save_xai_auth_json(tokens: dict) -> None:
    """Write xAI OAuth tokens to auth.json in hermes's expected format."""
    auth_path = Path(HERMES_HOME) / "auth.json"
    existing: dict = {}
    if auth_path.exists():
        try:
            existing = json.loads(auth_path.read_text())
        except Exception:
            pass
    if not isinstance(existing, dict):
        existing = {}

    providers = existing.setdefault("providers", {})
    providers["xai-oauth"] = {
        "tokens": tokens,
        "auth_mode": "oauth_device",
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "discovery": {
            "authorization_endpoint": "https://auth.x.ai/oauth2/authorize",
            "token_endpoint": _XAI_TOKEN_URL,
        },
        "redirect_uri": "",
    }
    existing["active_provider"] = "xai-oauth"
    existing["version"] = 2
    existing["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    auth_path.write_text(json.dumps(existing, indent=2) + "\n")
    try:
        auth_path.chmod(0o600)
    except Exception:
        pass


def _apply_xai_oauth_config(model: str) -> None:
    """Write config.yaml with provider=xai-oauth and the chosen model."""
    import yaml
    config_path = Path(HERMES_HOME) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if config_path.exists():
        try:
            with config_path.open() as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            pass

    merged = dict(existing)
    merged_model = dict(merged.get("model") if isinstance(merged.get("model"), dict) else {})
    if model:
        merged_model["default"] = model
    merged_model["provider"] = "xai-oauth"
    merged["model"] = merged_model

    merged_terminal = dict(merged.get("terminal") if isinstance(merged.get("terminal"), dict) else {})
    merged_terminal.setdefault("backend", "local")
    merged_terminal.setdefault("timeout", 60)
    merged_terminal.setdefault("cwd", "/tmp")
    merged["terminal"] = merged_terminal

    merged_agent = dict(merged.get("agent") if isinstance(merged.get("agent"), dict) else {})
    merged_agent.setdefault("max_iterations", 50)
    merged["agent"] = merged_agent
    merged["data_dir"] = HERMES_HOME

    with config_path.open("w") as f:
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=False)

    # Persist LLM_MODEL and track the per-provider model so the setup UI can
    # display it alongside the xAI entry in the "Configured Providers" list.
    if model:
        existing_env = read_env(ENV_FILE)
        existing_env["LLM_MODEL"] = model
        existing_env["_MODEL_XAI_OAUTH"] = model
        write_env(ENV_FILE, existing_env)


async def _poll_xai_device_auth(state: dict) -> None:
    """Background task: poll xAI token endpoint until authorized or expired."""
    client = get_http_client()
    while time.time() < state["expires_at"]:
        await asyncio.sleep(state["interval"])
        try:
            resp = await client.post(
                _XAI_TOKEN_URL,
                data={
                    "grant_type": _XAI_GRANT_TYPE,
                    "device_code": state["device_code"],
                    "client_id": _XAI_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=httpx.Timeout(15.0),
            )
        except Exception as e:
            print(f"[xai-oauth] poll error: {e!r}", flush=True)
            continue

        if resp.status_code == 200:
            try:
                tokens = resp.json()
            except Exception:
                state["status"] = "error"
                state["error"] = "Invalid token response from xAI"
                return
            _save_xai_auth_json(tokens)
            _apply_xai_oauth_config(state.get("model", ""))
            state["status"] = "authorized"
            print("[xai-oauth] authorized — restarting gateway", flush=True)
            asyncio.create_task(gw.restart())
            return

        try:
            err_data = resp.json()
        except Exception:
            err_data = {}
        error = err_data.get("error", "")

        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            state["interval"] = min(state["interval"] + 5, 30)
        else:
            state["status"] = "error"
            state["error"] = err_data.get("error_description", error) or error or "Unknown error"
            print(f"[xai-oauth] failed: {error}", flush=True)
            return

    state["status"] = "expired"
    print("[xai-oauth] device code expired", flush=True)


async def api_oauth_xai_delete(request: Request) -> Response:
    global _xai_oauth_state
    if err := guard(request):
        return err
    auth_path = Path(HERMES_HOME) / "auth.json"
    if auth_path.exists():
        try:
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            data.get("providers", {}).pop("xai-oauth", None)
            if data.get("active_provider") == "xai-oauth":
                data.pop("active_provider", None)
            auth_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
    env = read_env(ENV_FILE)
    env.pop("_MODEL_XAI_OAUTH", None)
    write_env(ENV_FILE, env)
    _xai_oauth_state = None
    return JSONResponse({"ok": True})


async def api_oauth_xai_start(request: Request) -> Response:
    global _xai_oauth_state
    if err := guard(request):
        return err

    try:
        body = await request.json()
    except Exception:
        body = {}
    model = str(body.get("model", "")).strip()

    client = get_http_client()
    try:
        resp = await client.post(
            _XAI_DEVICE_URL,
            data={"client_id": _XAI_CLIENT_ID, "scope": _XAI_SCOPE},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=httpx.Timeout(15.0),
        )
    except Exception as e:
        return JSONResponse({"error": f"Could not reach xAI: {e}"}, status_code=502)

    if resp.status_code != 200:
        return JSONResponse(
            {"error": f"xAI returned {resp.status_code}: {resp.text[:200]}"},
            status_code=502,
        )

    try:
        data = resp.json()
    except Exception:
        return JSONResponse({"error": "Invalid response from xAI"}, status_code=502)

    _xai_oauth_state = {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data.get("verification_uri_complete") or data["verification_uri"],
        "expires_at": time.time() + data.get("expires_in", 900),
        "interval": max(data.get("interval", 5), 5),
        "status": "pending",
        "model": model,
    }
    asyncio.create_task(_poll_xai_device_auth(_xai_oauth_state))

    return JSONResponse({
        "user_code": data["user_code"],
        "verification_uri": _xai_oauth_state["verification_uri"],
        "expires_in": data.get("expires_in", 900),
    })


async def api_oauth_xai_status(request: Request) -> Response:
    if err := guard(request):
        return err
    if _xai_oauth_state is None:
        # No active flow — check if a previous session left valid tokens.
        if _has_xai_oauth_tokens():
            return JSONResponse({"status": "authorized"})
        return JSONResponse({"status": "none"})
    return JSONResponse({
        "status": _xai_oauth_state["status"],
        "error": _xai_oauth_state.get("error", ""),
    })


def is_config_complete(data: dict[str, str] | None = None) -> bool:
    """Single source of truth for 'ready to run the gateway'.

    Used by: GET / redirect, auto_start on boot, admin API status.
    """
    if not gateway_enabled():
        return False
    if data is None:
        data = _effective_env()
    yaml_model, yaml_provider = _configured_model_from_yaml()
    model = data.get("LLM_MODEL") or data.get("HERMES_AUTH_MODEL") or yaml_model
    provider = data.get("HERMES_AUTH_PROVIDER") or yaml_provider
    has_model = bool(model)
    has_api_key_provider = any(data.get(k) for k in PROVIDER_KEYS)
    # OAuth providers keep credentials in auth.json rather than provider API-key
    # env vars. Accept the provider selected in config.yaml when it has stored
    # tokens; this keeps Railway redeploys from leaving the gateway stopped.
    has_oauth_provider = _oauth_provider_has_tokens(provider)
    has_provider = has_api_key_provider or has_oauth_provider or _has_xai_oauth_tokens()
    return has_model and has_provider


def mask(data: dict[str, str]) -> dict[str, str]:
    return {
        k: (v[:8] + "***" if len(v) > 8 else "***") if k in SECRET_KEYS and v else v
        for k, v in data.items()
    }


def unmask(new: dict[str, str], existing: dict[str, str]) -> dict[str, str]:
    return {
        k: (existing.get(k, "") if k in SECRET_KEYS and v.endswith("***") else v)
        for k, v in new.items()
    }


# ── Auth (cookie-based) ───────────────────────────────────────────────────────
# We use HMAC-signed cookies instead of HTTP Basic Auth because:
#   1. Basic auth's per-directory protection space means browsers cache creds
#      for /setup/* separately from /*, forcing re-prompt on navigation.
#   2. Browser behavior for sending Basic auth on XHR/fetch is inconsistent;
#      the Hermes React SPA's plain fetch() calls don't reliably include it,
#      causing every proxied API call to 401.
# Cookies are auto-included on every same-origin request (navigation + XHR)
# so both the setup UI and the proxied Hermes dashboard work with one login.
#
# The SECRET is regenerated on every process start. That means any ADMIN_PASSWORD
# change via Railway → redeploy → all existing cookies invalidate → users re-login.
import hashlib as _hashlib
import hmac as _hmac
from urllib.parse import quote as _url_quote, urlparse as _urlparse

COOKIE_NAME = "hermes_auth"
COOKIE_MAX_AGE = 7 * 86400  # 7 days
COOKIE_SECRET = secrets.token_bytes(32)

# Public paths — no auth required. Everything else is behind the cookie gate.
PUBLIC_PATHS = {"/health", "/login", "/logout"}


def _make_auth_token() -> str:
    """Build a cookie value: `<expires>.<hmac-sha256>`."""
    expires = str(int(time.time()) + COOKIE_MAX_AGE)
    sig = _hmac.new(COOKIE_SECRET, expires.encode(), _hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"


def _verify_auth_token(token: str) -> bool:
    try:
        expires_s, sig = token.rsplit(".", 1)
        if int(expires_s) < time.time():
            return False
        expected = _hmac.new(COOKIE_SECRET, expires_s.encode(), _hashlib.sha256).hexdigest()
        return _hmac.compare_digest(sig, expected)
    except Exception:
        return False


def _is_authenticated(request: Request) -> bool:
    return _verify_auth_token(request.cookies.get(COOKIE_NAME, ""))


def _safe_return_to(value: str) -> str:
    """Reject open-redirect attempts — only allow same-origin relative paths."""
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    # Strip any scheme/netloc that slipped through.
    p = _urlparse(value)
    if p.scheme or p.netloc:
        return "/"
    return value


def guard(request: Request) -> Response | None:
    """Enforce auth on protected routes.

    - HTML navigation: 302 to /login?returnTo=<path>
    - API / XHR: 401 JSON (so the SPA's fetch() can surface it cleanly)
    """
    if _is_authenticated(request):
        return None
    accept = request.headers.get("accept", "").lower()
    wants_html = "text/html" in accept
    if wants_html:
        rt = request.url.path
        if request.url.query:
            rt = f"{rt}?{request.url.query}"
        return RedirectResponse(f"/login?returnTo={_url_quote(rt)}", status_code=302)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Agent — Sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0f14;color:#c9d1d9;font-family:'IBM Plex Sans',sans-serif;
  min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#14181f;border:1px solid #252d3d;border-radius:12px;padding:36px 32px;width:100%;max-width:380px;
  box-shadow:0 20px 40px rgba(0,0,0,0.4)}
.brand{text-align:center;margin-bottom:28px}
.brand-logo{display:inline-flex;align-items:center;gap:10px;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:18px;color:#6272ff}
.brand-logo span{color:#6b7688;font-weight:400}
.brand-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7688;margin-top:8px;letter-spacing:1.5px;text-transform:uppercase}
label{display:block;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7688;
  letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px;margin-top:16px}
input{width:100%;background:#0d0f14;border:1px solid #252d3d;border-radius:6px;color:#c9d1d9;
  font-family:'IBM Plex Mono',monospace;font-size:13px;padding:9px 11px;outline:none;transition:border-color .15s}
input:focus{border-color:#6272ff}
button{width:100%;margin-top:24px;background:#6272ff;border:1px solid #6272ff;border-radius:6px;color:#fff;
  font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:500;padding:10px;cursor:pointer;
  transition:background .15s,border-color .15s}
button:hover{background:#7b8fff;border-color:#7b8fff}
.err{background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.3);border-radius:6px;
  color:#f85149;font-family:'IBM Plex Mono',monospace;font-size:12px;padding:8px 12px;margin-bottom:14px;text-align:center}
.footnote{margin-top:18px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6b7688;text-align:center;line-height:1.6}
</style></head>
<body>
<div class="card">
  <div class="brand">
    <div class="brand-logo">hermes<span>/admin</span></div>
    <div class="brand-sub">Sign in to continue</div>
  </div>
  __ERROR__
  <form method="POST" action="/login">
    <input type="hidden" name="returnTo" value="__RETURN_TO__">
    <label for="username">Username</label>
    <input id="username" name="username" type="text" autocomplete="username" autofocus required>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
  </form>
  <p class="footnote">Credentials are the <code>ADMIN_USERNAME</code> and <code>ADMIN_PASSWORD</code><br>Railway service variables.</p>
</div>
</body></html>"""


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


async def page_login(request: Request) -> Response:
    """GET /login — render the sign-in form."""
    # Already signed in? Bounce to returnTo (or /).
    if _is_authenticated(request):
        return RedirectResponse(_safe_return_to(request.query_params.get("returnTo", "/")), status_code=302)
    rt = _safe_return_to(request.query_params.get("returnTo", "/"))
    error_html = ('<div class="err">Invalid username or password</div>'
                  if request.query_params.get("error") else "")
    html = (LOGIN_PAGE_HTML
            .replace("__ERROR__", error_html)
            .replace("__RETURN_TO__", _html_escape(rt)))
    return HTMLResponse(html)


async def login_post(request: Request) -> Response:
    """POST /login — validate creds and set the auth cookie."""
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))
    return_to = _safe_return_to(str(form.get("returnTo", "/")))

    valid_user = _hmac.compare_digest(username, ADMIN_USERNAME)
    valid_pw = _hmac.compare_digest(password, ADMIN_PASSWORD)
    if valid_user and valid_pw:
        resp = RedirectResponse(return_to, status_code=302)
        resp.set_cookie(
            COOKIE_NAME,
            _make_auth_token(),
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp
    return RedirectResponse(f"/login?returnTo={_url_quote(return_to)}&error=1", status_code=302)


async def logout(request: Request) -> Response:
    """GET /logout — clear cookie and bounce to login."""
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ── Gateway manager ───────────────────────────────────────────────────────────
# Auto-respawn tuning. When the gateway exits without us asking it to — an
# in-band `/restart` (inside a container hermes exits 75 expecting a supervisor
# to bring it back; verified it takes the exit-75 path, NOT a detached
# self-restart, when /run/.containerenv or /.dockerenv exists), a crash, or an
# OOM kill — server.py is that supervisor and must restart it. Nothing else
# will, and /health stays 200, so the bot would otherwise sit silently dead.
# A crash-loop guard stops us hammering a gateway that genuinely can't stay up
# (e.g. a bad provider key / model).
RESPAWN_WINDOW_S   = 120     # rolling window (s) for counting unexpected exits
RESPAWN_MAX_IN_WIN = 5       # give up auto-restart after this many exits in window
RESPAWN_BASE_DELAY = 2.0     # first backoff (seconds)
RESPAWN_MAX_DELAY  = 30.0    # backoff cap


class Gateway:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=500)
        self.started_at: float | None = None
        self.restarts = 0
        # True while a deliberate stop()/restart()/reset is in flight, so the
        # exiting process's _drain() doesn't fire an auto-respawn that races the
        # intentional lifecycle.
        self._stopping = False
        # Monotonic timestamps of recent unexpected exits (crash-loop guard).
        self._recent_exits: list[float] = []

    async def start(self, *, reset_budget: bool = True):
        if self.proc and self.proc.returncode is None:
            return
        # A manual Start/Restart (or boot) grants a fresh crash-loop budget; the
        # auto-respawn path passes reset_budget=False so repeated crashes keep
        # accumulating toward the give-up threshold.
        if reset_budget:
            self._recent_exits.clear()
        self.state = "starting"
        self._stopping = False
        try:
            # .env values take priority over Railway env vars.
            # We build the env this way so hermes's own dotenv loading
            # (which reads the same file) doesn't shadow our values.
            env = {**os.environ, "HERMES_HOME": HERMES_HOME}
            env.update(read_env(ENV_FILE))
            model = env.get("LLM_MODEL") or env.get("HERMES_AUTH_MODEL") or ""
            provider_key = next((env.get(k, "") for k in PROVIDER_KEYS if env.get(k)), "")
            auth_provider = env.get("HERMES_AUTH_PROVIDER", "")
            print(f"[gateway] model={model or '⚠ NOT SET'} | provider_key={'set' if provider_key else '⚠ NOT SET'} | auth_provider={auth_provider or '⚠ NOT SET'}", flush=True)
            # Write config.yaml so hermes picks up the model/provider (env vars alone aren't always enough)
            write_config_yaml(env)
            self.proc = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.started_at = time.time()
            asyncio.create_task(self._drain(self.proc))
        except Exception as e:
            self.state = "error"
            self.logs.append(f"[error] Failed to start: {e}")

    async def stop(self):
        self._stopping = True
        if not self.proc or self.proc.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.state = "stopped"
        self.started_at = None

    async def restart(self):
        await self.stop()
        self.restarts += 1
        await self.start()

    async def _drain(self, proc: asyncio.subprocess.Process):
        assert proc.stdout
        async for raw in proc.stdout:
            line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
            self.logs.append(line)
        rc = proc.returncode
        # Ignore the drain of a process we've already replaced (e.g. via restart()).
        if proc is not self.proc:
            return
        # A deliberate stop()/restart()/reset owns its own lifecycle — don't respawn.
        if self._stopping:
            return
        # Unexpected exit: in-band `/restart` (exit 75), a crash, or an OOM kill.
        # On Railway nothing else brings the gateway back, so we supervise it.
        self.state = "error"
        self.logs.append(f"[gateway] exited (code {rc}) — supervising restart")
        asyncio.create_task(self._supervise_respawn(proc.pid))

    async def _supervise_respawn(self, dead_pid: int | None):
        # Crash-loop guard: count unexpected exits inside a rolling window and
        # give up (rather than hammer) once they exceed the threshold.
        now = time.monotonic()
        self._recent_exits = [t for t in self._recent_exits if now - t < RESPAWN_WINDOW_S]
        self._recent_exits.append(now)
        if len(self._recent_exits) > RESPAWN_MAX_IN_WIN:
            self.state = "crashed"
            self.logs.append(
                f"[gateway] crash-looping ({len(self._recent_exits)} exits in "
                f"{RESPAWN_WINDOW_S}s) — giving up auto-restart. Fix the provider/"
                f"model in the admin UI, then Start/Restart the gateway."
            )
            return
        delay = min(RESPAWN_BASE_DELAY * 2 ** (len(self._recent_exits) - 1), RESPAWN_MAX_DELAY)
        self.logs.append(f"[gateway] restarting in {int(delay)}s (attempt {len(self._recent_exits)})")
        await asyncio.sleep(delay)
        # Re-check the deliberate-lifecycle conditions AFTER the backoff sleep: a
        # Stop, Reset, or shutdown issued during the wait must win over the respawn.
        if self._stopping:
            self.logs.append("[gateway] restart cancelled (stopped/reconfigured)")
            return
        if self.proc and self.proc.returncode is None:
            return  # a manual Start already brought a live gateway back
        if not is_config_complete():
            self.state = "stopped"
            self.logs.append("[gateway] restart skipped — provider/model not configured")
            return
        # Clear a pid file left stale by a hard crash (SIGKILL/OOM skips hermes'
        # atexit cleanup) so the respawn's own O_EXCL pid claim can't bail with
        # "PID file race lost". Scoped to the pid we just buried — never disturbs
        # a live gateway's lock.
        self._clear_stale_pidfile(dead_pid)
        self.restarts += 1
        await self.start(reset_budget=False)

    def _clear_stale_pidfile(self, dead_pid: int | None) -> None:
        if dead_pid is None:
            return
        pid_file = Path(HERMES_HOME) / "gateway.pid"
        try:
            rec = json.loads(pid_file.read_text())
        except Exception:
            return
        if rec.get("pid") == dead_pid:
            try:
                pid_file.unlink()
                self.logs.append(f"[gateway] cleared stale pid file (pid {dead_pid})")
            except OSError:
                pass

    def status(self) -> dict:
        uptime = int(time.time() - self.started_at) if self.started_at and self.state == "running" else None
        return {
            "state":    self.state,
            "pid":      self.proc.pid if self.proc and self.proc.returncode is None else None,
            "uptime":   uptime,
            "restarts": self.restarts,
        }


gw = Gateway()
cfg_lock = asyncio.Lock()


# ── Legion command bus ────────────────────────────────────────────────────────
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TASK_PENDING_STATES = ("pending", "queued", "ready", "new")
TASK_RUNNING_STATE = "running"
TASK_DONE_STATE = "completed"
TASK_FAILED_STATE = "failed"


def _env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _legion_transport() -> str:
    return _env_value("COMMAND_TRANSPORT", "LEGION_COMMAND_SOURCE", "LEGION_REPORT_TRANSPORT")


def _legion_transport_declared() -> bool:
    transport = _legion_transport().lower()
    return bool(transport and ("postgres" in transport or "redis" in transport or "http" in transport))


def _legion_worker_bus_configured() -> bool:
    """True only when this process can actually poll/execute Postgres tasks."""
    transport = _legion_transport().lower()
    return "postgres" in transport and bool(os.environ.get("DATABASE_URL"))


def _legion_disabled_reason() -> str:
    if _falsey(os.environ.get("LEGION_COMMAND_BUS_ENABLED")):
        return "LEGION_COMMAND_BUS_ENABLED=false"
    if not _legion_transport_declared():
        return "command transport not configured"
    if "postgres" not in _legion_transport().lower():
        return "postgres transport required for worker polling"
    if not os.environ.get("DATABASE_URL"):
        return "DATABASE_URL not configured"
    if not (
        _truthy(os.environ.get("WORKER_MODE"))
        or _truthy(os.environ.get("LEGION_WORKER_MODE"))
        or _falsey(os.environ.get("TELEGRAM_GATEWAY_ENABLED"))
        or not bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
    ):
        return "gateway service is not in worker polling mode"
    return ""


def _quote_ident(name: str) -> str:
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _legion_agent_id() -> str:
    return _env_value("LEGION_AGENT_ID", "AGENT_ID", "RAILWAY_SERVICE_NAME") or "hermes-worker"


def _legion_id() -> str:
    return _env_value("LEGION_ID") or _legion_agent_id().split("-", 1)[0] or "legion"


def _legion_schema() -> str:
    explicit = _env_value("POSTGRES_SCHEMA", "LEGION_POSTGRES_SCHEMA", "DATABASE_SCHEMA")
    if explicit:
        return explicit
    agent = _legion_agent_id().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", agent).strip("_")
    return cleaned if cleaned.startswith("legion_") else f"legion_{cleaned or 'worker'}"


def _legion_redis_prefix() -> str:
    return _env_value("REDIS_KEY_PREFIX", "LEGION_REDIS_PREFIX") or _legion_schema()


def _task_prompt(task: dict) -> str:
    return "\n".join([
        "You are a Railway-hosted Hermes legion worker.",
        f"Legion: {_legion_id()}",
        f"Agent: {_legion_agent_id()}",
        f"Work order: {task.get('work_order_id') or ''}",
        f"Task: {task.get('task_id') or ''}",
        f"Type: {task.get('task_type') or 'general'}",
        "",
        "Execute the instruction. Return a concise Korean operational report with concrete outputs, blockers, and verification evidence.",
        "",
        "Instruction:",
        str(task.get("instruction") or "").strip(),
    ])


class LegionCommandBus:
    """Postgres-backed worker loop for Railway legion services.

    Railway service links and env vars are not enough by themselves: a worker has
    to heartbeat, claim tasks, execute them, and write results back. This bus is
    deliberately conservative: Postgres is the source of truth, Redis is only an
    optional wake-up/status side channel, and all SQL identifiers are allowlisted.
    """

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.state = "stopped"
        self.last_error = ""
        self.last_task_id = ""
        self.last_heartbeat_at: float | None = None
        self.processed = 0
        self.failed = 0

    def should_start(self) -> bool:
        if _falsey(os.environ.get("LEGION_COMMAND_BUS_ENABLED")):
            return False
        if not _legion_worker_bus_configured():
            return False
        # Commander services may still run a gateway. They can use the enqueue
        # API, but should not consume worker tasks unless explicitly requested.
        return (
            _truthy(os.environ.get("WORKER_MODE"))
            or _truthy(os.environ.get("LEGION_WORKER_MODE"))
            or _falsey(os.environ.get("TELEGRAM_GATEWAY_ENABLED"))
            or not bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        )

    async def start(self) -> None:
        if self.task and not self.task.done():
            return
        if not self.should_start():
            self.state = "disabled"
            return
        self.state = "starting"
        self.task = asyncio.create_task(self._run(), name="legion-command-bus")

    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.state = "stopped"

    def status(self) -> dict:
        return {
            "enabled": self.should_start(),
            "polling_enabled": self.should_start(),
            "state": self.state,
            "disabled_reason": "" if self.should_start() else _legion_disabled_reason(),
            "transport": _legion_transport(),
            "agent_id": _legion_agent_id(),
            "legion_id": _legion_id(),
            "postgres_schema": _legion_schema(),
            "redis_key_prefix": _legion_redis_prefix(),
            "transport_configured": _legion_transport_declared(),
            "database_configured": bool(os.environ.get("DATABASE_URL")),
            "redis_configured": bool(os.environ.get("REDIS_URL")),
            "last_error": self.last_error,
            "last_task_id": self.last_task_id,
            "last_heartbeat_at": self.last_heartbeat_at,
            "processed": self.processed,
            "failed": self.failed,
        }

    async def _run(self) -> None:
        self.state = "running"
        poll_seconds = max(2, int(os.environ.get("LEGION_POLL_SECONDS", "10")))
        while True:
            try:
                await asyncio.to_thread(self._ensure_schema)
                await asyncio.to_thread(self._heartbeat, "idle")
                task = await asyncio.to_thread(self._claim_next_task)
                if task:
                    await self._execute_task(task)
                else:
                    await self._publish_redis_status("idle")
                    await asyncio.sleep(poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state = "error"
                self.last_error = repr(exc)
                print(f"[legion-bus] error: {exc!r}", flush=True)
                await asyncio.sleep(min(30, poll_seconds * 2))
                self.state = "running"

    def _connect(self):
        import psycopg
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL is required for legion command bus")
        return psycopg.connect(url, autocommit=True)

    def _ensure_schema(self) -> None:
        self._ensure_schema_name(_legion_schema())

    def _ensure_schema_name(self, schema_name: str) -> None:
        schema = _quote_ident(schema_name)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.work_orders (
                    work_order_id text PRIMARY KEY,
                    legion_id text,
                    commander_agent_id text,
                    title text,
                    objective text,
                    priority text DEFAULT 'normal',
                    status text DEFAULT 'pending',
                    quality_gate_required boolean DEFAULT false,
                    created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.tasks (
                    task_id text PRIMARY KEY,
                    work_order_id text,
                    legion_id text,
                    agent_id text,
                    task_type text,
                    instruction text,
                    status text DEFAULT 'pending',
                    priority text DEFAULT 'normal',
                    redis_queue_key text,
                    created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.artifacts (
                    artifact_id text PRIMARY KEY,
                    work_order_id text,
                    task_id text,
                    legion_id text,
                    agent_id text,
                    artifact_type text,
                    r2_key text,
                    title text,
                    mime_type text,
                    size_bytes bigint,
                    metadata jsonb,
                    created_at timestamptz DEFAULT now()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema}.agent_status (
                    agent_id text PRIMARY KEY,
                    legion_id text,
                    role text,
                    status text,
                    redis_key_prefix text,
                    postgres_schema text,
                    r2_prefix text,
                    last_heartbeat_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now()
                )
            """)

    def _heartbeat(self, status: str) -> None:
        schema = _quote_ident(_legion_schema())
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.agent_status
                    (agent_id, legion_id, role, status, redis_key_prefix, postgres_schema, r2_prefix, last_heartbeat_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (agent_id) DO UPDATE SET
                    legion_id=EXCLUDED.legion_id,
                    role=EXCLUDED.role,
                    status=EXCLUDED.status,
                    redis_key_prefix=EXCLUDED.redis_key_prefix,
                    postgres_schema=EXCLUDED.postgres_schema,
                    r2_prefix=EXCLUDED.r2_prefix,
                    last_heartbeat_at=now(),
                    updated_at=now()
                """,
                (_legion_agent_id(), _legion_id(), os.environ.get("LEGION_ROLE", "worker"), status, _legion_redis_prefix(), _legion_schema(), os.environ.get("R2_PREFIX", "")),
            )
        self.last_heartbeat_at = time.time()

    def _claim_next_task(self) -> dict | None:
        schema = _quote_ident(_legion_schema())
        with self._connect() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT task_id, work_order_id, legion_id, agent_id, task_type, instruction, status, priority
                        FROM {schema}.tasks
                        WHERE status = ANY(%s)
                          AND (agent_id IS NULL OR agent_id = '' OR agent_id = %s)
                        ORDER BY
                          CASE priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                          created_at
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                        """,
                        (list(TASK_PENDING_STATES), _legion_agent_id()),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    task_id = row[0]
                    cur.execute(
                        f"UPDATE {schema}.tasks SET status=%s, agent_id=%s, updated_at=now() WHERE task_id=%s",
                        (TASK_RUNNING_STATE, _legion_agent_id(), task_id),
                    )
                    if row[1]:
                        cur.execute(
                            f"UPDATE {schema}.work_orders SET status=%s, updated_at=now() WHERE work_order_id=%s",
                            (TASK_RUNNING_STATE, row[1]),
                        )
                    return {
                        "task_id": row[0],
                        "work_order_id": row[1],
                        "legion_id": row[2],
                        "agent_id": row[3],
                        "task_type": row[4],
                        "instruction": row[5],
                        "status": row[6],
                        "priority": row[7],
                    }

    async def _execute_task(self, task: dict) -> None:
        task_id = str(task.get("task_id") or "")
        self.last_task_id = task_id
        await asyncio.to_thread(self._heartbeat, "busy")
        await self._publish_redis_status("busy", task_id=task_id)
        timeout = max(60, int(os.environ.get("LEGION_TASK_TIMEOUT_SECONDS", "900")))
        env = {**os.environ, "HERMES_HOME": HERMES_HOME}
        env.update(read_env(ENV_FILE))
        prompt = _task_prompt(task)
        try:
            proc = await asyncio.create_subprocess_exec(
                "hermes", "chat", "-q", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = raw.decode(errors="replace")[-20000:]
            if proc.returncode == 0:
                await asyncio.to_thread(self._finish_task, task, TASK_DONE_STATE, output, None)
                self.processed += 1
            else:
                await asyncio.to_thread(self._finish_task, task, TASK_FAILED_STATE, output, f"hermes exited {proc.returncode}")
                self.failed += 1
        except Exception as exc:
            await asyncio.to_thread(self._finish_task, task, TASK_FAILED_STATE, "", repr(exc))
            self.failed += 1
        finally:
            await asyncio.to_thread(self._heartbeat, "idle")
            await self._publish_redis_status("idle")

    def _r2_config(self) -> dict[str, str]:
        account_id = _env_value("R2_ACCOUNT_ID", "CLOUDFLARE_ACCOUNT_ID")
        endpoint = _env_value("R2_ENDPOINT_URL", "R2_ENDPOINT", "S3_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3")
        if not endpoint and account_id:
            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        return {
            "bucket": _env_value("R2_BUCKET", "R2_BUCKET_NAME", "CLOUDFLARE_R2_BUCKET", "S3_BUCKET"),
            "endpoint_url": endpoint,
            "access_key_id": _env_value("R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
            "secret_access_key": _env_value("R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            "region_name": _env_value("R2_REGION", "AWS_DEFAULT_REGION") or "auto",
        }

    def _store_artifact_payload(self, artifact_id: str, task: dict, status: str, output: str, error: str | None) -> tuple[str, int, dict]:
        payload = {
            "artifact_id": artifact_id,
            "work_order_id": task.get("work_order_id"),
            "task_id": task.get("task_id"),
            "legion_id": _legion_id(),
            "agent_id": _legion_agent_id(),
            "status": status,
            "error": error,
            "output": output,
            "created_at": int(time.time()),
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode()
        prefix = os.environ.get("R2_PREFIX", _legion_schema()).strip("/")
        object_key = f"{prefix}/artifacts/{artifact_id}.json" if prefix else f"artifacts/{artifact_id}.json"
        metadata = {
            "status": status,
            "error": error,
            "storage": "postgres_inline",
            "output": output,
        }
        cfg = self._r2_config()
        if all(cfg.get(k) for k in ("bucket", "endpoint_url", "access_key_id", "secret_access_key")):
            try:
                import boto3
                from botocore.config import Config

                client = boto3.client(
                    "s3",
                    endpoint_url=cfg["endpoint_url"],
                    aws_access_key_id=cfg["access_key_id"],
                    aws_secret_access_key=cfg["secret_access_key"],
                    region_name=cfg["region_name"],
                    config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2}),
                )
                client.put_object(
                    Bucket=cfg["bucket"],
                    Key=object_key,
                    Body=body,
                    ContentType="application/json; charset=utf-8",
                )
                metadata = {
                    "status": status,
                    "error": error,
                    "storage": "r2",
                    "bucket": cfg["bucket"],
                    "output_preview": output[:4000],
                }
                return object_key, len(body), metadata
            except Exception as exc:
                metadata["r2_upload_error"] = repr(exc)
                metadata["artifact_storage_degraded"] = True
                self.last_error = repr(exc)
        return "", len(body), metadata

    def _finish_task(self, task: dict, status: str, output: str, error: str | None) -> None:
        schema = _quote_ident(_legion_schema())
        artifact_id = f"artifact-{uuid.uuid4().hex}"
        r2_key, size_bytes, metadata = self._store_artifact_payload(artifact_id, task, status, output, error)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {schema}.tasks SET status=%s, updated_at=now() WHERE task_id=%s",
                (status, task.get("task_id")),
            )
            if task.get("work_order_id"):
                cur.execute(
                    f"UPDATE {schema}.work_orders SET status=%s, updated_at=now() WHERE work_order_id=%s",
                    (status, task.get("work_order_id")),
                )
            cur.execute(
                f"""
                INSERT INTO {schema}.artifacts
                    (artifact_id, work_order_id, task_id, legion_id, agent_id, artifact_type, r2_key, title, mime_type, size_bytes, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                """,
                (
                    artifact_id,
                    task.get("work_order_id"),
                    task.get("task_id"),
                    _legion_id(),
                    _legion_agent_id(),
                    "hermes_task_result",
                    r2_key,
                    f"Task result {task.get('task_id')}",
                    "application/json",
                    size_bytes,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        if error:
            self.last_error = error

    async def _publish_redis_status(self, status: str, *, task_id: str = "") -> None:
        if not os.environ.get("REDIS_URL"):
            return
        try:
            await asyncio.to_thread(self._publish_redis_status_sync, status, task_id)
        except Exception as exc:
            self.last_error = repr(exc)

    def _publish_redis_status_sync(self, status: str, task_id: str = "") -> None:
        import redis
        client = redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=2, socket_timeout=2)
        key = f"{_legion_redis_prefix()}:agent:{_legion_agent_id()}:status"
        payload = {
            "agent_id": _legion_agent_id(),
            "legion_id": _legion_id(),
            "status": status,
            "task_id": task_id,
            "postgres_schema": _legion_schema(),
            "updated_at": str(int(time.time())),
        }
        client.hset(key, mapping=payload)
        client.expire(key, 300)


legion_bus = LegionCommandBus()


def _legion_api_authorized(request: Request) -> bool:
    shared_secret = os.environ.get("LEGION_SHARED_SECRET") or os.environ.get("COMMAND_API_TOKEN")
    auth = request.headers.get("authorization", "")
    return _is_authenticated(request) or bool(shared_secret and _hmac.compare_digest(auth, f"Bearer {shared_secret}"))


async def api_legion_status(request: Request):
    if not _legion_api_authorized(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(legion_bus.status())


async def api_legion_enqueue(request: Request):
    # Internal API: allow either admin cookie or a shared bearer token.
    if not _legion_api_authorized(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        target_schema = body.get("postgres_schema") or body.get("schema") or _legion_schema()
        _quote_ident(target_schema)
        work_order_id = body.get("work_order_id") or f"wo-{uuid.uuid4().hex}"
        task_id = body.get("task_id") or f"task-{uuid.uuid4().hex}"
        title = body.get("title") or "Commander work order"
        instruction = body.get("instruction") or body.get("objective") or ""
        if not instruction.strip():
            return JSONResponse({"error": "instruction is required"}, status_code=400)
        await asyncio.to_thread(_enqueue_task_sync, target_schema, work_order_id, task_id, title, instruction, body)
        return JSONResponse({"ok": True, "postgres_schema": target_schema, "work_order_id": work_order_id, "task_id": task_id})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def _enqueue_task_sync(schema_name: str, work_order_id: str, task_id: str, title: str, instruction: str, body: dict) -> None:
    schema = _quote_ident(schema_name)
    legion_bus._ensure_schema_name(schema_name)
    with legion_bus._connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {schema}.work_orders
                (work_order_id, legion_id, commander_agent_id, title, objective, priority, status, quality_gate_required, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, now(), now())
            ON CONFLICT (work_order_id) DO UPDATE SET
                title=EXCLUDED.title,
                objective=EXCLUDED.objective,
                priority=EXCLUDED.priority,
                status='pending',
                updated_at=now()
            """,
            (work_order_id, body.get("legion_id") or _legion_id(), _legion_agent_id(), title, body.get("objective") or instruction, body.get("priority") or "normal", bool(body.get("quality_gate_required", False))),
        )
        cur.execute(
            f"""
            INSERT INTO {schema}.tasks
                (task_id, work_order_id, legion_id, agent_id, task_type, instruction, status, priority, redis_queue_key, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, %s, now(), now())
            ON CONFLICT (task_id) DO UPDATE SET
                instruction=EXCLUDED.instruction,
                status='pending',
                priority=EXCLUDED.priority,
                updated_at=now()
            """,
            (task_id, work_order_id, body.get("legion_id") or _legion_id(), body.get("agent_id") or "", body.get("task_type") or "general", instruction, body.get("priority") or "normal", body.get("redis_queue_key") or f"{schema_name}:tasks"),
        )


# ── Hermes dashboard subprocess ───────────────────────────────────────────────
class Dashboard:
    """Manages the `hermes dashboard` subprocess (native Hermes web UI).

    Bound to loopback only — we expose it to the public internet through our
    reverse proxy on $PORT, where edge basic auth guards every request.
    The dashboard is independent of the gateway: it reads config files
    directly and tolerates a stopped gateway.

    All subprocess output is streamed to our stdout (→ Railway logs) with a
    `[dashboard]` prefix AND retained in a ring buffer for diagnostics.
    Unexpected exits are explicitly logged with their return code.
    """

    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.logs: deque[str] = deque(maxlen=300)
        self._drain_task: asyncio.Task | None = None

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        try:
            self.proc = await asyncio.create_subprocess_exec(
                "hermes", "dashboard",
                "--host", HERMES_DASHBOARD_HOST,
                "--port", str(HERMES_DASHBOARD_PORT),
                "--no-open",
                # --skip-build: the Dockerfile pre-builds the React dashboard
                # into hermes_cli/web_dist/ at image time. This flag tells
                # hermes to trust that dist and skip its npm build check,
                # which would otherwise add ~30s to first startup (hermes >= v2026.5.16).
                "--skip-build",
                # NOTE: the embedded Chat tab (/api/pty + /api/ws + /api/events)
                # is unconditionally enabled as of hermes v2026.6.5 — the old
                # `--tui` flag was REMOVED from the dashboard subcommand. Passing
                # it now aborts startup with "unrecognized arguments: --tui",
                # which kills this subprocess and 503s the reverse proxy. The
                # Dockerfile still pre-builds ui-tui/dist/ (via HERMES_TUI_DIR)
                # so the PTY child spawns instantly on first chat connect.
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            print(f"[dashboard] spawned pid={self.proc.pid} → {HERMES_DASHBOARD_URL}", flush=True)
            self._drain_task = asyncio.create_task(self._drain())
        except Exception as e:
            print(f"[dashboard] FAILED to spawn: {e!r}", flush=True)

    def running(self) -> bool:
        return bool(self.proc and self.proc.returncode is None)

    async def _drain(self):
        """Stream subprocess output to Railway logs (prefixed) and a ring buffer."""
        assert self.proc and self.proc.stdout
        try:
            async for raw in self.proc.stdout:
                line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
                self.logs.append(line)
                print(f"[dashboard] {line}", flush=True)
        except Exception as e:
            print(f"[dashboard] drain error: {e!r}", flush=True)
        finally:
            rc = self.proc.returncode if self.proc else None
            if rc is not None and rc != 0:
                print(f"[dashboard] EXITED with code {rc} — reverse proxy will return 503 until restart", flush=True)
            elif rc == 0:
                print(f"[dashboard] exited cleanly (code 0)", flush=True)

    async def stop(self):
        if not self.proc or self.proc.returncode is not None:
            return
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


dash = Dashboard()

# Shared async HTTP client for the reverse proxy. Created lazily so we pick up
# the running event loop, torn down in lifespan.
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=False,
        )
    return _http_client


# ── Route handlers ────────────────────────────────────────────────────────────
async def page_index(request: Request):
    if err := guard(request): return err
    return templates.TemplateResponse(request, "index.html")


def _gateway_pidfile_live() -> bool:
    """Best-effort liveness check for a gateway started by any supervisor.

    Railway has two possible gateway supervisors during upgrades: this admin
    server and the native dashboard sidecar. Health should reflect the actual
    bot process, not only this server's in-memory gw.state.
    """
    try:
        rec = json.loads((Path(HERMES_HOME) / "gateway.pid").read_text())
        pid = int(rec.get("pid") or 0)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


async def route_health(request: Request):
    gateway_state = "running" if gateway_enabled() and (gw.state == "running" or _gateway_pidfile_live()) else gw.state
    bus_enabled = legion_bus.should_start()
    return JSONResponse({
        "status": "ok",
        "gateway": gateway_state,
        "gateway_enabled": gateway_enabled(),
        "worker_mode": _truthy(os.environ.get("WORKER_MODE")),
        "transport": _legion_transport(),
        "command_bus": {
            "enabled": bus_enabled,
            "polling_enabled": bus_enabled,
            "state": legion_bus.state,
            "disabled_reason": "" if bus_enabled else _legion_disabled_reason(),
            "transport_configured": _legion_transport_declared(),
            "database_configured": bool(os.environ.get("DATABASE_URL")),
            "redis_configured": bool(os.environ.get("REDIS_URL")),
        },
    })


async def api_config_get(request: Request):
    if err := guard(request): return err
    async with cfg_lock:
        data = read_env(ENV_FILE)
    defs = [{"key": k, "label": l, "category": c, "secret": s} for k, l, c, s in ENV_VARS]
    return JSONResponse({"vars": mask(data), "defs": defs})


async def api_config_put(request: Request):
    if err := guard(request): return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    try:
        restart = body.pop("_restart", False)
        new_vars = body.get("vars", {})
        async with cfg_lock:
            existing = read_env(ENV_FILE)
            merged = unmask(new_vars, existing)
            for k, v in existing.items():
                if k not in merged:
                    merged[k] = v
            write_env(ENV_FILE, merged)
            write_config_yaml(merged)
        if restart:
            asyncio.create_task(gw.restart())
        return JSONResponse({"ok": True, "restarting": restart})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_status(request: Request):
    if err := guard(request): return err
    data = read_env(ENV_FILE)
    providers = {
        k.replace("_API_KEY","").replace("_TOKEN","").replace("HF_","HuggingFace ").replace("_"," ").title():
        {"configured": bool(data.get(k))}
        for k in PROVIDER_KEYS
    }
    channels = {
        name: {"configured": bool(v := data.get(key,"")) and v.lower() not in ("false","0","no")}
        for name, key in CHANNEL_MAP.items()
    }
    gateway_status = gw.status()
    if gateway_status.get("state") != "running" and _gateway_pidfile_live():
        gateway_status["state"] = "running"
        try:
            gateway_status["pid"] = int(json.loads((Path(HERMES_HOME) / "gateway.pid").read_text()).get("pid") or 0)
        except Exception:
            pass
    return JSONResponse({"gateway": gateway_status, "providers": providers, "channels": channels})


async def api_logs(request: Request):
    if err := guard(request): return err
    return JSONResponse({"lines": list(gw.logs)})


async def api_gw_start(request: Request):
    if err := guard(request): return err
    if not gateway_enabled():
        return JSONResponse({"ok": False, "error": "Gateway disabled for this deployment"}, status_code=409)
    asyncio.create_task(gw.start())
    return JSONResponse({"ok": True})


async def api_gw_stop(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.stop())
    return JSONResponse({"ok": True})


async def api_gw_restart(request: Request):
    if err := guard(request): return err
    if not gateway_enabled():
        return JSONResponse({"ok": False, "error": "Gateway disabled for this deployment"}, status_code=409)
    asyncio.create_task(gw.restart())
    return JSONResponse({"ok": True})


async def api_config_reset(request: Request):
    if err := guard(request): return err
    asyncio.create_task(gw.stop())
    async with cfg_lock:
        if ENV_FILE.exists():
            ENV_FILE.unlink()
        write_config_yaml({})
    return JSONResponse({"ok": True})


# ── Pairing ───────────────────────────────────────────────────────────────────
# Pending-request file format (hermes >= v0.15 / v2026.5.29.x, gateway/pairing.py):
# each `{platform}-pending.json` entry is keyed by a random opaque `entry_id`
# (secrets.token_hex), and the user-facing pairing code is stored only as a
# salted hash ({hash, salt, user_id, user_name, created_at}) — the plaintext
# code is never on disk. Our admin-approval flow is code-agnostic: the dashboard
# is already cookie-authed, so we approve by moving an entry from pending →
# approved keyed off that `entry_id` (round-tripped from the pending list as
# `code`), reading `user_id`/`user_name` straight from the entry. We must NOT
# uppercase that key — entry_ids are lowercase hex, and uppercasing them was
# what silently broke approve/deny on the v0.15 upgrade. Older plaintext-keyed
# entries still work here because we treat the key as an opaque handle.
def _pjson(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _wjson(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    try: os.chmod(path, 0o600)
    except OSError: pass


def _platforms(suffix: str) -> list[str]:
    if not PAIRING_DIR.exists(): return []
    return [f.stem.rsplit(f"-{suffix}", 1)[0] for f in PAIRING_DIR.glob(f"*-{suffix}.json")]


async def api_pairing_pending(request: Request):
    if err := guard(request): return err
    now = time.time()
    out = []
    for p in _platforms("pending"):
        for code, info in _pjson(PAIRING_DIR / f"{p}-pending.json").items():
            if now - info.get("created_at", now) <= PAIRING_TTL:
                out.append({"platform": p, "code": code,
                            "user_id": info.get("user_id",""), "user_name": info.get("user_name",""),
                            "age_minutes": int((now - info.get("created_at", now)) / 60)})
    return JSONResponse({"pending": out})


async def api_pairing_approve(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, code = body.get("platform",""), body.get("code","").strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)
    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _pjson(pending_path)
    if code not in pending:
        return JSONResponse({"error": "Code not found"}, status_code=404)
    entry = pending.pop(code)
    user_id = (entry.get("user_id") or "").strip() if isinstance(entry, dict) else ""
    if not user_id:
        # Malformed/legacy entry without a user_id — leave it in pending (we
        # haven't written the pop yet) rather than silently discarding it.
        return JSONResponse({"error": "Pending entry has no user_id"}, status_code=422)
    _wjson(pending_path, pending)
    approved = _pjson(PAIRING_DIR / f"{platform}-approved.json")
    approved[user_id] = {"user_name": entry.get("user_name",""), "approved_at": time.time()}
    _wjson(PAIRING_DIR / f"{platform}-approved.json", approved)
    return JSONResponse({"ok": True})


async def api_pairing_deny(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, code = body.get("platform",""), body.get("code","").strip()
    p = PAIRING_DIR / f"{platform}-pending.json"
    pending = _pjson(p)
    if code in pending:
        del pending[code]
        _wjson(p, pending)
    return JSONResponse({"ok": True})


async def api_pairing_approved(request: Request):
    if err := guard(request): return err
    out = []
    for p in _platforms("approved"):
        for uid, info in _pjson(PAIRING_DIR / f"{p}-approved.json").items():
            out.append({"platform": p, "user_id": uid,
                        "user_name": info.get("user_name",""), "approved_at": info.get("approved_at",0)})
    return JSONResponse({"approved": out})


async def api_pairing_revoke(request: Request):
    if err := guard(request): return err
    try: body = await request.json()
    except Exception: return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    platform, uid = body.get("platform",""), body.get("user_id","")
    if not platform or not uid:
        return JSONResponse({"error": "platform and user_id required"}, status_code=400)
    p = PAIRING_DIR / f"{platform}-approved.json"
    approved = _pjson(p)
    if uid in approved:
        del approved[uid]
        _wjson(p, approved)
    return JSONResponse({"ok": True})


# ── Reverse proxy → Hermes dashboard ──────────────────────────────────────────
_WIDGET_LINK_STYLE = (
    "background:rgba(20,24,31,0.92);backdrop-filter:blur(8px);"
    "border:1px solid #252d3d;border-radius:6px;padding:6px 12px;"
    "color:#c9d1d9;text-decoration:none;display:inline-flex;"
    "align-items:center;gap:6px;"
)
BACK_TO_SETUP_WIDGET = (
    '<div id="hermes-back-widget" style="position:fixed;bottom:14px;right:14px;'
    'z-index:99999;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
    'font-size:11px;display:flex;gap:8px;">'
    f'<a href="/setup" style="{_WIDGET_LINK_STYLE}">← Setup</a>'
    f'<a href="/logout" style="{_WIDGET_LINK_STYLE}">Sign out</a>'
    '</div>'
)

DASHBOARD_UNAVAILABLE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Dashboard starting…</title>
<style>body{background:#0d0f14;color:#c9d1d9;font-family:ui-monospace,Menlo,monospace;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{max-width:480px;padding:32px;border:1px solid #252d3d;border-radius:12px;
background:#14181f;text-align:center}
h1{font-size:16px;color:#d29922;margin:0 0 12px;font-weight:600}
p{font-size:13px;color:#6b7688;line-height:1.6;margin:0 0 16px}
a{color:#6272ff;text-decoration:none;border:1px solid #252d3d;border-radius:6px;
padding:7px 14px;font-size:12px;display:inline-block}
a:hover{border-color:#6272ff}</style></head>
<body><div class="card">
<h1>⚠ Hermes dashboard unavailable</h1>
<p>The native Hermes dashboard is not responding on port %d.<br>
It may still be starting up, or it may have crashed.</p>
<p>Try refreshing in a few seconds, or head back to setup.</p>
<a href="/setup">← Back to Setup</a>
</div>
<script>setTimeout(()=>location.reload(),4000);</script>
</body></html>""" % HERMES_DASHBOARD_PORT


async def _proxy_to_dashboard(request: Request) -> Response:
    """Forward an authenticated request to the Hermes dashboard subprocess.

    Assumes edge auth (basic auth middleware) has already validated the caller.
    HTTP-only: the native Hermes dashboard does not use WebSockets.
    """
    if not dash.running():
        await dash.start()
    client = get_http_client()
    target = f"{HERMES_DASHBOARD_URL}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    req_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    body = await request.body()

    try:
        upstream = await client.request(
            request.method,
            target,
            headers=req_headers,
            content=body,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return HTMLResponse(DASHBOARD_UNAVAILABLE_HTML, status_code=503)
    except httpx.RequestError as e:
        print(f"[proxy] upstream error for {request.method} {request.url.path}: {e}", flush=True)
        return HTMLResponse(DASHBOARD_UNAVAILABLE_HTML, status_code=502)

    # Surface non-2xx responses from hermes into Railway logs so we can
    # diagnose 401/500s without needing browser DevTools access.
    if upstream.status_code >= 400:
        body_snip = upstream.content[:200].decode("utf-8", errors="replace")
        print(
            f"[proxy] {request.method} {request.url.path} -> {upstream.status_code} "
            f"body={body_snip!r}",
            flush=True,
        )

    # Strip hop-by-hop and length/encoding headers — Starlette recomputes them.
    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in HOP_BY_HOP
        and k.lower() not in ("content-encoding", "content-length")
    }

    content = upstream.content
    content_type = upstream.headers.get("content-type", "").lower()

    # Inject the "← Setup" widget into HTML pages so users can always return.
    if "text/html" in content_type and b"</body>" in content:
        try:
            text = content.decode("utf-8", errors="replace")
            text = text.replace("</body>", BACK_TO_SETUP_WIDGET + "</body>", 1)
            content = text.encode("utf-8")
        except Exception:
            pass  # on any error, fall back to raw upstream content

    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=resp_headers,
    )


async def route_root(request: Request) -> Response:
    """GET /: first-visit smart redirect, otherwise proxy to the dashboard.

    - Unconfigured + bare GET `/` → bounce to `/setup` so new users land on
      the wizard instead of a half-empty dashboard.
    - Sidebar / in-app links pass `?force=1` to opt out of that redirect —
      users who explicitly want the dashboard (e.g. to set providers via
      the Keys tab) can still reach it without saving config first.
    - Non-GET (SPA API calls, etc.) always proxy through.
    """
    if err := guard(request): return err
    if (request.method == "GET"
            and request.query_params.get("force") != "1"
            and not is_config_complete()):
        return RedirectResponse("/setup", status_code=302)
    return await _proxy_to_dashboard(request)


async def route_proxy(request: Request) -> Response:
    """Catch-all: forward any unmatched path to the Hermes dashboard."""
    if err := guard(request): return err
    return await _proxy_to_dashboard(request)


async def route_setup_404(request: Request) -> Response:
    """Typos under /setup/* should 404 here — not fall through to the proxy."""
    if err := guard(request): return err
    return Response("Not Found", status_code=404, media_type="text/plain")


# ── App lifecycle ─────────────────────────────────────────────────────────────
async def auto_start():
    if not gateway_enabled():
        print("[server] Gateway disabled by WORKER_MODE/GATEWAY_ENABLED — running admin/dashboard only.", flush=True)
        return
    if is_config_complete():
        asyncio.create_task(gw.start())
    else:
        print("[server] Config incomplete — gateway not started. Configure provider + model in the admin UI.", flush=True)


@asynccontextmanager
async def lifespan(app):
    # The gateway is the critical path for Telegram. The dashboard is heavy and
    # optional, so in Railway/Telegram deployments we start it lazily on first
    # authenticated web request unless HERMES_DASHBOARD_AUTOSTART=1 is set.
    if HERMES_DASHBOARD_AUTOSTART:
        asyncio.create_task(dash.start())
    else:
        print("[dashboard] lazy mode enabled — not starting dashboard until first web request", flush=True)
    try:
        auth_bootstrap = _apply_auth_json_bootstrap()
        if auth_bootstrap.get("applied"):
            print(f"[auth-bootstrap] applied mode={auth_bootstrap.get('mode')} provider={auth_bootstrap.get('provider') or 'unknown'}", flush=True)
    except Exception as exc:
        print(f"[auth-bootstrap] failed: {exc!r}", flush=True)
    await auto_start()
    await legion_bus.start()
    try:
        yield
    finally:
        await asyncio.gather(
            gw.stop(),
            dash.stop(),
            legion_bus.stop(),
            return_exceptions=True,
        )
        global _http_client
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


# ── WebSocket reverse proxy ──────────────────────────────────────────────────
# The hermes dashboard exposes several WebSocket endpoints when started with
# --tui. The browser SPA opens these and they must flow through our reverse
# proxy. /api/pub is opened only by the PTY child against loopback and is
# intentionally NOT proxied — exposing it would let an authed user spam events
# into channels. It lives at /api/pub (not under /api/plugins/), so the plugin
# prefix route below does not match it.
#
#   /api/pty                  binary stream — embedded TUI keystrokes/output
#   /api/ws                   JSON-RPC      — gateway sidecar driving Chat metadata
#   /api/events               text frames   — dashboard subscriber for /api/pub fan-out
#   /api/plugins/<name>/...   plugin-contributed sockets. Mounted by hermes
#                             under /api/plugins/<name>/ (web_server.
#                             _mount_plugin_api_routes), e.g. kanban's
#                             /api/plugins/kanban/events live task feed. Added
#                             in v0.15 — without a proxy route Starlette 403s
#                             the upgrade and the SPA retries in a tight loop.
#
# Auth model (matches the HTTP proxy):
#   * Edge: our HMAC cookie via _is_authenticated. WebSocket inherits .cookies
#     from starlette HTTPConnection so the same helper works unchanged.
#   * Upstream: hermes's own ?token=<_SESSION_TOKEN> query param. The SPA
#     fetches that token via /api/auth/session-token and includes it in the
#     WS URL, so we just forward path + query verbatim.
PROXIED_WS_PATHS = ("/api/pty", "/api/ws", "/api/events", "/api/plugins/*")


async def _ws_pump_client_to_upstream(
    client: WebSocket,
    upstream: websockets.WebSocketClientProtocol,
) -> None:
    """Forward client → upstream until the client side disconnects.

    Handles both binary (PTY bytes) and text (JSON-RPC) frames.
    """
    try:
        while True:
            msg = await client.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data is not None:
                await upstream.send(data)
                continue
            text = msg.get("text")
            if text is not None:
                await upstream.send(text)
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        return
    except Exception as e:
        print(f"[ws-proxy] client→upstream error on {client.url.path}: {e!r}", flush=True)
        return


async def _ws_pump_upstream_to_client(
    upstream: websockets.WebSocketClientProtocol,
    client: WebSocket,
) -> None:
    """Forward upstream → client until upstream closes."""
    try:
        async for msg in upstream:
            if isinstance(msg, bytes):
                await client.send_bytes(msg)
            else:
                await client.send_text(msg)
    except (websockets.exceptions.ConnectionClosed, WebSocketDisconnect):
        return
    except Exception as e:
        print(f"[ws-proxy] upstream→client error on {client.url.path}: {e!r}", flush=True)
        return


async def ws_proxy(websocket: WebSocket) -> None:
    """Reverse-proxy a single WebSocket from browser → hermes dashboard.

    Order matters: connect upstream BEFORE accepting the client. If hermes
    is wedged or rejects the upgrade, we close the client with a meaningful
    code instead of accepting and then dropping silently.

    Connection lifecycle:
      1. Verify edge cookie auth → 4401 close on failure
      2. Open upstream WS with bounded open_timeout → 1011 on failure
      3. Accept client
      4. Spawn two pump tasks (bidirectional byte forwarding)
      5. When either direction ends (client navigates away, upstream PTY
         exits, etc.), cancel the other task and close both sockets
    """
    # 1. Edge auth.
    if not _is_authenticated(websocket):
        # Close before accept — browser sees the handshake fail (expected
        # for unauthenticated calls).
        await websocket.close(code=4401)
        return

    # 2. Build upstream URL preserving the SPA's path + query (the query
    #    contains the hermes session token + channel id).
    path = websocket.url.path
    qs = websocket.url.query
    upstream_url = f"ws://{HERMES_DASHBOARD_HOST}:{HERMES_DASHBOARD_PORT}{path}"
    if qs:
        upstream_url = f"{upstream_url}?{qs}"

    try:
        if not dash.running():
            await dash.start()
        upstream = await websockets.connect(
            upstream_url,
            open_timeout=5,
            # Don't forward client cookies/headers — hermes WS auth is
            # purely token-based via the URL, and forwarding random
            # headers risks future upstream surprises.
        )
    except (asyncio.TimeoutError, OSError, websockets.exceptions.WebSocketException) as e:
        # Hermes dashboard down, restarting, or rejected the upgrade
        # (e.g. bad/missing session token).
        print(f"[ws-proxy] upstream connect failed for {path}: {e!r}", flush=True)
        # 1011 = internal error; client SPA will surface a generic close.
        await websocket.close(code=1011)
        return

    # 3. Both sides ready — accept and start pumping.
    await websocket.accept()

    pump_in = asyncio.create_task(_ws_pump_client_to_upstream(websocket, upstream))
    pump_out = asyncio.create_task(_ws_pump_upstream_to_client(upstream, websocket))

    try:
        # First side to finish wins; cancel the other.
        done, pending = await asyncio.wait(
            (pump_in, pump_out),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        # websockets.connect() outside `async with` doesn't auto-close;
        # do it explicitly. Same for the client side if still open.
        try:
            await upstream.close()
        except Exception:
            pass
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass


ANY_METHOD = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

routes = [
    # Public — no auth required.
    Route("/health",                            route_health),
    Route("/legion/api/status",                 api_legion_status),
    Route("/legion/api/tasks",                  api_legion_enqueue,  methods=["POST"]),
    Route("/login",                             page_login,          methods=["GET"]),
    Route("/login",                             login_post,          methods=["POST"]),
    Route("/logout",                            logout),

    # Our setup wizard + management API, all under /setup/* (cookie-auth guarded).
    Route("/setup",                             page_index),
    Route("/setup/",                            page_index),
    Route("/setup/api/config",                  api_config_get,      methods=["GET"]),
    Route("/setup/api/config",                  api_config_put,      methods=["PUT"]),
    Route("/setup/api/status",                  api_status),
    Route("/setup/api/logs",                    api_logs),
    Route("/setup/api/gateway/start",           api_gw_start,        methods=["POST"]),
    Route("/setup/api/gateway/stop",            api_gw_stop,         methods=["POST"]),
    Route("/setup/api/gateway/restart",         api_gw_restart,      methods=["POST"]),
    Route("/setup/api/config/reset",            api_config_reset,    methods=["POST"]),
    Route("/setup/api/pairing/pending",         api_pairing_pending),
    Route("/setup/api/pairing/approve",         api_pairing_approve, methods=["POST"]),
    Route("/setup/api/pairing/deny",            api_pairing_deny,    methods=["POST"]),
    Route("/setup/api/pairing/approved",        api_pairing_approved),
    Route("/setup/api/pairing/revoke",          api_pairing_revoke,  methods=["POST"]),
    Route("/setup/api/oauth/xai/start",         api_oauth_xai_start,  methods=["POST"]),
    Route("/setup/api/oauth/xai/status",        api_oauth_xai_status),
    Route("/setup/api/oauth/xai",               api_oauth_xai_delete, methods=["DELETE"]),

    # /setup/* typos return a real 404 — not a silent proxy fallthrough.
    Route("/setup/{path:path}",                 route_setup_404,     methods=ANY_METHOD),

    # Reverse-proxy hermes's dashboard WebSockets (Chat tab + sidecar).
    # WebSocketRoute is matched independently of HTTP routes, so order
    # relative to the catch-all HTTP `Route("/{path:path}", ...)` below
    # doesn't matter — but listing them as a group keeps the surface
    # area auditable. Only paths in PROXIED_WS_PATHS are forwarded;
    # /api/pub is intentionally omitted (not under /api/plugins/, so the
    # prefix route below does not match it).
    WebSocketRoute("/api/pty",                  ws_proxy),
    WebSocketRoute("/api/ws",                   ws_proxy),
    WebSocketRoute("/api/events",               ws_proxy),
    # Plugin-contributed sockets, mounted by hermes under /api/plugins/<name>/
    # (e.g. kanban's /api/plugins/kanban/events). Prefix-matched so new plugin
    # WS endpoints in future hermes releases proxy without re-touching this list.
    WebSocketRoute("/api/plugins/{path:path}",  ws_proxy),

    # Root: redirect to /setup if unconfigured, otherwise proxy the dashboard.
    Route("/",                                  route_root,          methods=ANY_METHOD),

    # Catch-all: everything else proxies to the Hermes dashboard subprocess.
    Route("/{path:path}",                       route_proxy,         methods=ANY_METHOD),
]

# No middleware — auth is enforced per-handler via guard(). This keeps /health
# and /login truly unauthenticated without middleware gymnastics.
app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def _shutdown():
        loop.create_task(gw.stop())
        loop.create_task(dash.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    loop.run_until_complete(server.serve())
