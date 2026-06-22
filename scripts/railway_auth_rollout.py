#!/usr/bin/env python3
"""Roll out Hermes auth.json bootstrap vars to Railway legion services.

Examples:
  python scripts/railway_auth_rollout.py \
    --auth-json-file /data/.hermes/auth.json \
    --provider openai-codex --model gpt-5.5 --selector workers --redeploy

  python scripts/railway_auth_rollout.py \
    --auth-json-file /data/.hermes/auth.json \
    --provider openai-codex --model gpt-5.5 --include '3군단|4군단-개발'

Security:
  - Never prints auth.json or token values.
  - Uses `railway variable set KEY --stdin --skip-deploys` so secrets do not
    appear in process arguments or shell history.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Service:
    legion: str
    name: str
    project_id: str
    service_id: str
    role: str  # commander|worker|database


SERVICES = [
    Service("작전본부", "총사령관", "a5acdd0b-faae-40d8-8293-372d4b91489f", "1b8097c2-074a-4082-95bf-59f0c2d46806", "commander"),
    Service("작전본부", "작전본부-운영참모", "a5acdd0b-faae-40d8-8293-372d4b91489f", "a7c0e609-7ec4-4570-adbd-5da96a52ed33", "worker"),
    Service("2군단", "2군단장", "3b928697-2b5a-477f-a600-322fc639dded", "5fa23c6d-1365-4cde-bf9e-87a50367d163", "commander"),
    Service("2군단", "KB데이터팀", "3b928697-2b5a-477f-a600-322fc639dded", "a4280d1b-cca3-4d6c-9699-6d7f2b20b034", "worker"),
    Service("2군단", "QA운영팀", "3b928697-2b5a-477f-a600-322fc639dded", "98c521d8-c69c-48da-805e-917eab9fa4dd", "worker"),
    Service("2군단", "개발팀", "3b928697-2b5a-477f-a600-322fc639dded", "d03b1f49-d492-4899-8f82-df1225ed6ac6", "worker"),
    Service("2군단", "기획전략팀", "3b928697-2b5a-477f-a600-322fc639dded", "3625117c-7122-4e25-ad09-1b352182c5bb", "worker"),
    Service("2군단", "디자인팀", "3b928697-2b5a-477f-a600-322fc639dded", "4e2a518b-a793-46a8-8a37-8266276aebed", "worker"),
    Service("2군단", "마케팅팀", "3b928697-2b5a-477f-a600-322fc639dded", "c47f7488-d61e-4fba-8923-7a7a5d0914b4", "worker"),
    Service("3군단", "3군단장", "bcb28c32-cbb4-4eac-a8cb-6ee1ecd68808", "0ea2ccd5-59bd-48cb-a75b-102441b3ef80", "commander"),
    Service("3군단", "3군단-기획및리뷰팀", "bcb28c32-cbb4-4eac-a8cb-6ee1ecd68808", "1817e293-9ef0-48f7-aba1-82ce80691165", "worker"),
    Service("3군단", "3군단-개발팀-A", "bcb28c32-cbb4-4eac-a8cb-6ee1ecd68808", "fb54d586-b319-4555-9d89-6aba184db906", "worker"),
    Service("3군단", "3군단-개발팀-B", "bcb28c32-cbb4-4eac-a8cb-6ee1ecd68808", "0b749d89-d421-49e2-ac4a-4f2ca163c58b", "worker"),
    Service("4군단", "4군단장", "c1d1951d-7ce7-4b1c-8635-a0f4a1aaba47", "1fbd93e8-5651-4b5a-9461-a855ff4ef812", "commander"),
    Service("4군단", "4군단-설계및PM팀", "c1d1951d-7ce7-4b1c-8635-a0f4a1aaba47", "fc48fde4-92aa-4a1c-8305-36327677bdc4", "worker"),
    Service("4군단", "4군단-개발검증팀", "c1d1951d-7ce7-4b1c-8635-a0f4a1aaba47", "9f4b0701-6ead-443a-96f6-27cc1d19d0c4", "worker"),
    Service("4군단", "4군단-작전통제실", "c1d1951d-7ce7-4b1c-8635-a0f4a1aaba47", "b83e1bb3-a868-417f-8ce1-9eac9a86fbe4", "worker"),
    Service("4군단", "4군단-크리에이티브제작워커", "c1d1951d-7ce7-4b1c-8635-a0f4a1aaba47", "d22cb380-e132-4b95-8b74-6e9b2c86a5d6", "worker"),
]


def railway_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("RAILWAY_"):
            env.pop(key)
    return env


def run(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=input_text, text=True, capture_output=True, env=railway_env(), check=check)


def set_var(service: Service, key: str, value: str, dry_run: bool) -> None:
    if dry_run:
        return
    run([
        "railway", "variable", "set", key,
        "--stdin",
        "--project", service.project_id,
        "--environment", "production",
        "--service", service.service_id,
        "--skip-deploys",
        "--json",
    ], input_text=value)


def redeploy(service: Service, dry_run: bool) -> None:
    if dry_run:
        return
    run([
        "railway", "service", "redeploy",
        "--project", service.project_id,
        "--environment", "production",
        "--service", service.service_id,
        "--yes",
        "--json",
    ])


def select_services(args: argparse.Namespace) -> list[Service]:
    role = {"workers": "worker", "commanders": "commander"}.get(args.selector, args.selector)
    selected = [s for s in SERVICES if args.selector == "all" or s.role == role]
    if args.include:
        pat = re.compile(args.include)
        selected = [s for s in selected if pat.search(f"{s.legion}/{s.name}/{s.service_id}")]
    if args.exclude:
        pat = re.compile(args.exclude)
        selected = [s for s in selected if not pat.search(f"{s.legion}/{s.name}/{s.service_id}")]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll out Hermes auth bootstrap to Railway legion services")
    parser.add_argument("--auth-json-file", required=True, help="Path to source Hermes auth.json")
    parser.add_argument("--provider", default="openai-codex", help="Provider to activate, e.g. openai-codex")
    parser.add_argument("--model", default="gpt-5.5", help="Model to persist in config.yaml")
    parser.add_argument("--mode", default="merge", choices=["missing", "merge", "replace", "force"], help="Bootstrap mode")
    parser.add_argument("--selector", default="workers", choices=["all", "workers", "commanders"], help="Target role set")
    parser.add_argument("--include", help="Regex matched against legion/name/service_id")
    parser.add_argument("--exclude", help="Regex matched against legion/name/service_id")
    parser.add_argument("--redeploy", action="store_true", help="Redeploy targets after setting variables")
    parser.add_argument("--dry-run", action="store_true", help="Print target plan only; do not write variables")
    args = parser.parse_args()

    auth_path = Path(args.auth_json_file).expanduser()
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("providers"), dict):
        raise SystemExit("auth-json-file must be a Hermes auth.json object with providers")
    if args.provider not in payload["providers"]:
        raise SystemExit(f"provider {args.provider!r} not found in {auth_path}")

    encoded = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    targets = select_services(args)
    if not targets:
        raise SystemExit("No services matched selector/include/exclude")

    print(f"Auth rollout plan: targets={len(targets)} provider={args.provider} model={args.model} mode={args.mode} auth_b64_len={len(encoded)} redeploy={args.redeploy} dry_run={args.dry_run}")
    for svc in targets:
        print(f"- {svc.legion}/{svc.name} [{svc.role}] {svc.service_id}")

    for svc in targets:
        set_var(svc, "HERMES_AUTH_JSON_B64", encoded, args.dry_run)
        set_var(svc, "HERMES_AUTH_PROVIDER", args.provider, args.dry_run)
        set_var(svc, "HERMES_AUTH_MODEL", args.model, args.dry_run)
        set_var(svc, "HERMES_AUTH_BOOTSTRAP_MODE", args.mode, args.dry_run)
        print(f"configured {svc.legion}/{svc.name}")
        if args.redeploy:
            redeploy(svc, args.dry_run)
            print(f"redeploy requested {svc.legion}/{svc.name}")

    print("Done. Secrets were not printed. If --redeploy was omitted, redeploy/restart targets before expecting runtime auth.json/config.yaml changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
