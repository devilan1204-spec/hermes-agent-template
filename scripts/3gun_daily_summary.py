#!/usr/bin/env python3
"""Daily 3군단 Railway/Hermes operation summary for Telegram.

No secrets are read or emitted. Intended for Hermes cron no_agent mode.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SERVICES: dict[str, dict[str, Any]] = {
    '3군단장': {
        'url': 'https://3-production-2160.up.railway.app/health',
        'role': 'commander',
        'expect_gateway': 'running',
        'expect_gateway_enabled': True,
        'expect_worker_mode': False,
        'expect_command_bus': None,
    },
    '3군단-기획및리뷰팀': {
        'url': 'https://3-production-ed30.up.railway.app/health',
        'role': 'worker',
        'expect_gateway': 'stopped',
        'expect_gateway_enabled': False,
        'expect_worker_mode': True,
        'expect_command_bus': 'running',
    },
    '3군단-개발팀-A': {
        'url': 'https://3-a-production.up.railway.app/health',
        'role': 'worker',
        'expect_gateway': 'stopped',
        'expect_gateway_enabled': False,
        'expect_worker_mode': True,
        'expect_command_bus': 'running',
    },
    '3군단-개발팀-B': {
        'url': 'https://3-b-production.up.railway.app/health',
        'role': 'worker',
        'expect_gateway': 'stopped',
        'expect_gateway_enabled': False,
        'expect_worker_mode': True,
        'expect_command_bus': 'running',
    },
}

APP_SERVICES = set(SERVICES)
INFRA_SERVICES = {'Redis', 'Postgres'}
REPORT_SERVICE_NAMES = APP_SERVICES | INFRA_SERVICES
STATE_PATH = Path('/data/.hermes/cron/3gun_health_watchdog_state.json')
JOBS_PATH = Path('/data/.hermes/cron/jobs.json')
HEARTBEAT_PATH = Path('/data/.hermes/marsos_status.json')
TEXT_REPORT_LIMIT = 180

SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|token|secret|password|authorization)(["\'\s:=]+)([^"\'\s,}]{6,})'),
    re.compile(r'(?i)bearer\s+[a-z0-9._~+/=-]{10,}'),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_text(text: str, limit: int = TEXT_REPORT_LIMIT) -> str:
    compact = re.sub(r'\s+', ' ', text.replace('\r', ' ').replace('\n', ' ')).strip()
    for pattern in SECRET_PATTERNS:
        compact = pattern.sub(lambda m: f'{m.group(1) if len(m.groups()) >= 1 else "secret"}{m.group(2) if len(m.groups()) >= 2 else "="}[REDACTED]', compact)
    return compact[:limit]


def _bad_status(data: dict[str, Any]) -> str | None:
    status = data.get('status')
    if isinstance(status, str) and status.lower() in {'error', 'failed', 'fail', 'down', 'unhealthy'}:
        return f'status={status!r}'
    if data.get('ok') is False:
        return 'ok=false'
    return None


def validate_health_data(data: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, str]:
    if bad := _bad_status(data):
        return False, bad

    role = spec['role']
    expected_gateway = spec.get('expect_gateway')
    if expected_gateway is not None and data.get('gateway') != expected_gateway:
        return False, f'{role}: gateway={data.get("gateway")!r}, expected {expected_gateway!r}'

    expected_gateway_enabled = spec.get('expect_gateway_enabled')
    if expected_gateway_enabled is not None and data.get('gateway_enabled') is not expected_gateway_enabled:
        return False, f'{role}: gateway_enabled={data.get("gateway_enabled")!r}, expected {expected_gateway_enabled!r}'

    expected_worker_mode = spec.get('expect_worker_mode')
    if expected_worker_mode is not None and data.get('worker_mode') is not expected_worker_mode:
        return False, f'{role}: worker_mode={data.get("worker_mode")!r}, expected {expected_worker_mode!r}'

    expected_command_bus = spec.get('expect_command_bus')
    command_bus = data.get('command_bus') if isinstance(data.get('command_bus'), dict) else {}
    if expected_command_bus == 'running' and (command_bus.get('state') != 'running' or command_bus.get('polling_enabled') is not True):
        return False, f'{role}: command_bus={command_bus.get("state")!r}, polling={command_bus.get("polling_enabled")!r}'

    return True, f'{role} role healthy'


def run_json(cmd: list[str], timeout: int = 60) -> Any:
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f'{cmd[0]} exit {cp.returncode}: {sanitize_text(cp.stderr)}')
    return json.loads(cp.stdout)


def railway_services() -> list[dict[str, Any]]:
    data = run_json(['railway', 'status', '--json'], timeout=90)
    rows = []
    for env_edge in data.get('environments', {}).get('edges', []):
        env = env_edge.get('node', {})
        if env.get('name') != 'production':
            continue
        for edge in env.get('serviceInstances', {}).get('edges', []):
            n = edge.get('node', {})
            dep = n.get('latestDeployment') or {}
            meta = dep.get('meta') or {}
            sm = meta.get('serviceManifest') or {}
            deploy = sm.get('deploy') or {}
            build = sm.get('build') or {}
            source = n.get('source') or {}
            domains = [d.get('domain') for d in (n.get('domains', {}).get('serviceDomains') or []) if d.get('domain')]
            rows.append({
                'name': n.get('serviceName'),
                'status': dep.get('status'),
                'instances': [i.get('status') for i in dep.get('instances', [])],
                'deployment_id': dep.get('id'),
                'created_at': dep.get('createdAt'),
                'source_type': 'repo' if source.get('repo') else ('image' if source.get('image') else None),
                'domains': domains,
                'runtime': meta.get('runtime'),
                'builder': build.get('builder'),
                'healthcheck': deploy.get('healthcheckPath'),
                'replicas': deploy.get('numReplicas'),
                'message': meta.get('cliMessage') or meta.get('commitMessage'),
            })
    return sorted(rows, key=lambda r: r.get('name') or '')


def health(name: str, url: str) -> dict[str, Any]:
    started = time.time()
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read(2000).decode(errors='replace')
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {
                    'ok': False,
                    'http': resp.status,
                    'latency_ms': round((time.time() - started) * 1000),
                    'error': f'invalid_json_root: {type(data).__name__}',
                    'data': {},
                }
            return {'ok': resp.status == 200, 'http': resp.status, 'latency_ms': round((time.time() - started) * 1000), 'data': data}
    except Exception as exc:
        return {'ok': False, 'http': None, 'latency_ms': None, 'error': sanitize_text(repr(exc)), 'data': {}}


def cron_summary() -> list[str]:
    try:
        if not JOBS_PATH.exists():
            return ['jobs.json 없음']
        data = json.loads(JOBS_PATH.read_text())
        lines = []
        for job in data.get('jobs', []):
            lines.append(f"{job.get('name')}: enabled={job.get('enabled')}, last_status={job.get('last_status')}")
        return lines or ['등록 job 없음']
    except Exception as exc:
        return [f'cron 상태 읽기 실패: {type(exc).__name__}']


def disk_summary() -> str:
    try:
        total, used, free = shutil.disk_usage('/data')
        used_pct = round((used / total) * 100) if total else 0
        if used_pct >= 85:
            return f'경고: /data 사용률 {used_pct}%'
        return '정상'
    except Exception as exc:
        return f'저장소 확인 실패: {type(exc).__name__}'


def main() -> None:
    out = ['📊 3군단 일일 운영 요약', f'- 시각(UTC): {now_iso()}', '']

    try:
        rows = railway_services()
        report_rows = [r for r in rows if r.get('name') in REPORT_SERVICE_NAMES]
        out.append('Railway 서비스:')
        for r in report_rows:
            name = r['name']
            instances = r.get('instances') or []
            instances_ok = bool(instances) and all(x == 'RUNNING' for x in instances)
            icon = '✅' if r['status'] == 'SUCCESS' and instances_ok else '⚠️'
            src = r.get('source_type') or '-'
            domain_state = 'present' if r.get('domains') else '-'
            out.append(f"- {icon} {name}: {r['status']}/{','.join(instances) or 'unknown'} · source={src} · domain={domain_state}")

        app_rows = [r for r in rows if r.get('name') in APP_SERVICES]
        app_counts = {name: 0 for name in APP_SERVICES}
        for r in app_rows:
            app_counts[r.get('name')] = app_counts.get(r.get('name'), 0) + 1
        missing_services = sorted(name for name, count in app_counts.items() if count == 0)
        duplicate_services = sorted(name for name, count in app_counts.items() if count > 1)
        complete_unique_rows = [r for r in app_rows if app_counts.get(r.get('name')) == 1]
        messages = {r.get('message') for r in complete_unique_rows}
        source_types = {r.get('source_type') for r in complete_unique_rows}
        healthchecks = {r.get('healthcheck') for r in complete_unique_rows}
        missing_metadata = [
            r.get('name') or 'unknown'
            for r in complete_unique_rows
            if not r.get('message') or not r.get('source_type') or not r.get('healthcheck')
        ]
        drift = (
            bool(missing_services)
            or bool(duplicate_services)
            or bool(missing_metadata)
            or len(messages) > 1
            or len(source_types) > 1
            or len(healthchecks) > 1
        )
        out.append(f"- 배포 drift: {'감지' if drift else '없음'}")
        if missing_services:
            out.append(f"- 누락 app 서비스: {', '.join(missing_services)}")
        if duplicate_services:
            out.append(f"- 중복 app 서비스 row: {', '.join(duplicate_services)}")
        if missing_metadata:
            out.append(f"- metadata 불완전: {', '.join(sorted(set(missing_metadata)))}")
    except Exception as exc:
        out.append(f'Railway 상태 확인 실패: {sanitize_text(repr(exc))}')

    out.append('')
    out.append('Role-aware health:')
    any_failed = False
    for name, spec in SERVICES.items():
        r = health(name, spec['url'])
        data = r.get('data') or {}
        cb = data.get('command_bus') if isinstance(data.get('command_bus'), dict) else {}
        body_ok, reason = validate_health_data(data, spec)
        expected_ok = bool(r['ok']) and body_ok
        any_failed = any_failed or not expected_ok
        icon = '✅' if expected_ok else '❌'
        out.append(f"- {icon} {name} [{spec['role']}]: HTTP {r.get('http')} {r.get('latency_ms')}ms · gateway={data.get('gateway')} · worker={data.get('worker_mode')} · bus={cb.get('state')} · {reason}")

    out.append('')
    out.append('Hermes cron:')
    for line in cron_summary():
        out.append(f'- {line}')

    out.append('')
    out.append(f'저장소: {disk_summary()}')

    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            out.append(f"watchdog state: any_failed={state.get('any_failed')} checked_at={state.get('checked_at')}")
        except Exception as exc:
            out.append(f'watchdog state 읽기 실패: {type(exc).__name__}')
    if HEARTBEAT_PATH.exists():
        out.append('heartbeat: 존재')

    out.append('')
    if any_failed:
        out.append('권장 조치: 실패 서비스 logs/variables 확인 후 redeploy 검토')
    else:
        out.append('판정: 3군단 role-aware 운영 상태 정상')

    print('\n'.join(out))


if __name__ == '__main__':
    main()
