#!/usr/bin/env python3
"""3군단 Railway/Hermes role-aware health watchdog.

Prints a Telegram-deliverable report only when:
- first run baseline is created,
- one or more services fail expected role-aware health,
- service status changes,
- all services recover after a failure.

Also refreshes /data/.hermes/marsos_status.json on every run.
Empty stdout means silent success for cron no_agent mode.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fcntl

STATE_PATH = Path('/data/.hermes/cron/3gun_health_watchdog_state.json')
LOCK_PATH = STATE_PATH.with_suffix('.lock')
HEARTBEAT_PATH = Path(os.environ.get('MARSOS_HEARTBEAT_PATH', '/data/.hermes/marsos_status.json'))
TIMEOUT_SECONDS = 10
BODY_READ_LIMIT = 1200
BODY_REPORT_LIMIT = 220
STEADY_FAILURE_REMINDER_EVERY = 12  # hourly reminders at the current 5m cron interval

SERVICES: dict[str, dict[str, Any]] = {
    '3군단장': {
        'url': 'https://3-production-2160.up.railway.app/health',
        'role': 'commander',
        'expect_gateway': 'running',
        'expect_gateway_enabled': True,
        'expect_worker_mode': False,
        'expect_command_bus': None,  # commander currently has command bus intentionally disabled
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


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def state_lock():
    """Serialize state load/save across cron and manual invocations."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open('w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_state() -> tuple[dict[str, Any], str | None]:
    if not STATE_PATH.exists():
        return {}, None
    try:
        data = json.loads(STATE_PATH.read_text())
        if not isinstance(data, dict):
            return {}, f'state file root is {type(data).__name__}, expected object'
        return data, None
    except Exception as exc:
        return {}, f'{type(exc).__name__}: {exc}'


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp_path.open('w', encoding='utf-8') as tmp_file:
        tmp_file.write(text)
        tmp_file.write('\n')
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    os.replace(tmp_path, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state)


SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|token|secret|password|authorization)(["\'\s:=]+)([^"\'\s,}]{6,})'),
    re.compile(r'(?i)bearer\s+[a-z0-9._~+/=-]{10,}'),
]


def sanitize_body(body: str, limit: int = BODY_REPORT_LIMIT) -> str:
    text = body.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: f'{m.group(1) if len(m.groups()) >= 1 else "secret"}{m.group(2) if len(m.groups()) >= 2 else "="}[REDACTED]', text)
    return text[:limit]


def _bad_status(data: dict[str, Any]) -> str | None:
    status = data.get('status')
    if isinstance(status, str) and status.lower() in {'error', 'failed', 'fail', 'down', 'unhealthy'}:
        return f'status={status!r}'
    if data.get('ok') is False:
        return 'ok=false'
    return None


def validate_health_body(body: str, service: dict[str, Any]) -> tuple[bool, str, str, dict[str, Any]]:
    """Validate /health response against expected commander/worker role."""
    snippet = sanitize_body(body)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, f'invalid_json: {exc.msg}', snippet, {}
    if not isinstance(data, dict):
        return False, f'invalid_json_root: {type(data).__name__}', snippet, {}

    if bad := _bad_status(data):
        return False, bad, snippet, data

    role = service['role']
    expected_gateway = service.get('expect_gateway')
    if expected_gateway is not None and data.get('gateway') != expected_gateway:
        return False, f'{role}: gateway={data.get("gateway")!r}, expected {expected_gateway!r}', snippet, data

    expected_gateway_enabled = service.get('expect_gateway_enabled')
    if expected_gateway_enabled is not None and data.get('gateway_enabled') is not expected_gateway_enabled:
        return False, f'{role}: gateway_enabled={data.get("gateway_enabled")!r}, expected {expected_gateway_enabled!r}', snippet, data

    expected_worker_mode = service.get('expect_worker_mode')
    if expected_worker_mode is not None and data.get('worker_mode') is not expected_worker_mode:
        return False, f'{role}: worker_mode={data.get("worker_mode")!r}, expected {expected_worker_mode!r}', snippet, data

    expected_command_bus = service.get('expect_command_bus')
    command_bus = data.get('command_bus') if isinstance(data.get('command_bus'), dict) else {}
    if expected_command_bus == 'running':
        if command_bus.get('state') != 'running' or command_bus.get('polling_enabled') is not True:
            return False, f'{role}: command_bus={command_bus.get("state")!r}, polling={command_bus.get("polling_enabled")!r}', snippet, data

    return True, f'{role} role healthy', snippet, data


def check_url(name: str, service: dict[str, Any]) -> dict[str, Any]:
    url = service['url']
    started = time.time()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Hermes-3gun-watchdog/2.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = resp.read(BODY_READ_LIMIT).decode('utf-8', errors='replace')
            latency_ms = round((time.time() - started) * 1000)
            body_ok, reason, snippet, parsed = validate_health_body(body, service)
            ok = 200 <= resp.status < 300 and body_ok
            return {
                'name': name,
                'role': service['role'],
                'url': url,
                'ok': ok,
                'status': resp.status,
                'latency_ms': latency_ms,
                'reason': reason,
                'body': snippet,
                'health': {
                    'status': parsed.get('status'),
                    'gateway': parsed.get('gateway'),
                    'gateway_enabled': parsed.get('gateway_enabled'),
                    'worker_mode': parsed.get('worker_mode'),
                    'transport': parsed.get('transport'),
                    'command_bus': parsed.get('command_bus'),
                } if parsed else {},
                'error': None,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(BODY_READ_LIMIT).decode('utf-8', errors='replace') if exc.fp else ''
        return {
            'name': name,
            'role': service['role'],
            'url': url,
            'ok': False,
            'status': exc.code,
            'latency_ms': round((time.time() - started) * 1000),
            'reason': f'http_error: {exc.reason}',
            'body': sanitize_body(body),
            'health': {},
            'error': None,
        }
    except Exception as exc:
        return {
            'name': name,
            'role': service['role'],
            'url': url,
            'ok': False,
            'status': None,
            'latency_ms': None,
            'reason': 'request_exception',
            'body': '',
            'health': {},
            'error': repr(exc),
        }


def railway_status_summary() -> list[str]:
    try:
        cp = subprocess.run(['railway', 'service', 'list'], text=True, capture_output=True, timeout=35)
        if cp.returncode != 0:
            return [f'Railway service list 실패: exit {cp.returncode}']
        lines = []
        current = None
        for raw in cp.stdout.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if not line.startswith(' ') and stripped not in {'Services in production'} and not stripped.startswith('This session'):
                current = stripped.replace(' (linked)', '')
            elif current and stripped.startswith('status:'):
                status = stripped.split(':', 1)[1].strip()
                if current.startswith('3군단') or current in {'Redis', 'Postgres'}:
                    lines.append(f'{current}: {status}')
        return lines[:10]
    except Exception as exc:
        return [f'Railway status 확인 실패: {exc!r}']


def write_heartbeat(checked_at: str, results: dict[str, dict[str, Any]], any_failed: bool) -> None:
    payload = {
        'service': os.environ.get('RAILWAY_SERVICE_NAME', '3군단장'),
        'group': 'Marsos',
        'role': '3-legion commander',
        'status': 'degraded' if any_failed else 'alive',
        'location': os.environ.get('RAILWAY_PROJECT_NAME', '3군단'),
        'heartbeat_at': checked_at,
        'last_heartbeat_at': checked_at,
        'heartbeat_file_path': str(HEARTBEAT_PATH),
        'watchdog': {
            'state_path': str(STATE_PATH),
            'mode': 'role-aware',
            'any_failed': any_failed,
            'service_count': len(results),
        },
        'services': {
            name: {
                'role': r.get('role'),
                'ok': r.get('ok'),
                'status': r.get('status'),
                'latency_ms': r.get('latency_ms'),
                'reason': r.get('reason'),
                'failure_count': r.get('failure_count', 0),
                'last_ok_at': r.get('last_ok_at'),
                'health': r.get('health') or {},
            }
            for name, r in results.items()
        },
        'next_action': '정상 시 5분 주기 silent watchdog 유지; 장애/복구 시 Telegram 보고',
    }
    atomic_write_json(HEARTBEAT_PATH, payload)


def main() -> None:
    with state_lock():
        previous, state_error = load_state()
        raw_prev_results = previous.get('results') if isinstance(previous, dict) else None
        if isinstance(raw_prev_results, dict):
            prev_results = raw_prev_results
        else:
            prev_results = {}
            if previous and state_error is None:
                state_error = f'state results is {type(raw_prev_results).__name__}, expected object'
        prev_any_failed = bool(previous.get('any_failed'))

        results = {name: check_url(name, svc) for name, svc in SERVICES.items()}
        checked_at = now_iso()

        changed = []
        reminder_failures = []
        for name, r in results.items():
            old = prev_results.get(name) or {}
            old_ok = bool(old.get('ok')) if old else None
            new_ok = bool(r.get('ok'))
            old_failure_count = safe_int(old.get('failure_count'))

            if new_ok:
                r['failure_count'] = 0
                r['first_failed_at'] = None
                r['last_ok_at'] = checked_at
            else:
                r['failure_count'] = old_failure_count + 1
                r['first_failed_at'] = old.get('first_failed_at') or checked_at
                r['last_ok_at'] = old.get('last_ok_at')
                if r['failure_count'] > 1 and r['failure_count'] % STEADY_FAILURE_REMINDER_EVERY == 0:
                    reminder_failures.append(name)

            if old_ok is None or old_ok != new_ok or old.get('reason') != r.get('reason'):
                changed.append(name)

        failed = [r for r in results.values() if not r['ok']]
        any_failed = bool(failed)
        first_run = not previous
        first_or_changed_failure = any_failed and bool(first_run or changed)
        steady_failure_reminder = bool(reminder_failures)
        should_report = bool(state_error) or first_run or first_or_changed_failure or steady_failure_reminder or (prev_any_failed and not any_failed)

        state = {
            'checked_at': checked_at,
            'any_failed': any_failed,
            'results': results,
        }
        save_state(state)
        write_heartbeat(checked_at, results, any_failed)

    if not should_report:
        return

    icon = '🟢' if not any_failed else '🔴'
    title = '3군단 자동 감시 보고'
    if state_error:
        reason = '상태 파일 오류 감지'
    elif first_run:
        reason = '초기 기준선 생성'
    elif any_failed and steady_failure_reminder:
        reason = '장애 지속 알림'
    elif any_failed:
        reason = '장애/비정상 감지'
    elif prev_any_failed and not any_failed:
        reason = '복구 확인'
    else:
        reason = '상태 변경 감지'

    out = [f'{icon} {title}', f'- 시각(UTC): {state["checked_at"]}', f'- 사유: {reason}', '']
    if state_error:
        out.append(f'- 상태 파일 경고: {state_error}')
        out.append('- 조치: 현재 health 결과로 상태 파일을 atomic replace 방식으로 재생성함')
        out.append('')
    out.append('서비스 role-aware health:')
    for name, r in results.items():
        s_icon = '✅' if r['ok'] else '❌'
        detail = f'HTTP {r["status"]}' if r['status'] is not None else r['error']
        latency = f', {r["latency_ms"]}ms' if r['latency_ms'] is not None else ''
        role = r.get('role') or 'unknown'
        extra = f' ({r.get("reason")})'
        if not r['ok']:
            extra = f' ({r.get("reason") or "unknown"}, 연속 {r.get("failure_count", 1)}회)'
        out.append(f'- {s_icon} {name} [{role}]: {detail}{latency}{extra}')
        if not r['ok']:
            out.append(f'  - URL: {r["url"]}')
            if r.get('first_failed_at'):
                out.append(f'  - 최초 실패(UTC): {r["first_failed_at"]}')
            if r['body']:
                out.append(f'  - body(redacted): {r["body"]}')

    out.append('')
    out.append(f'heartbeat: {HEARTBEAT_PATH}')

    if any_failed:
        out.append('')
        out.append('Railway 상태 요약:')
        for line in railway_status_summary():
            out.append(f'- {line}')
        out.append('')
        out.append('권장 조치: Railway logs 확인 후 필요 시 해당 서비스 redeploy')

    print('\n'.join(out))


if __name__ == '__main__':
    main()
