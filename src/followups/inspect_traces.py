"""Langfuse 트레이스 디버깅 도구: 세션별 턴 트리, 지연, 토큰, 병렬 여부, 오류를 출력한다.

uv run python -m followups.inspect_traces            # 최근 세션들
uv run python -m followups.inspect_traces hoodie-3398a0
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(override=True)
BASE = os.environ["LANGFUSE_BASE_URL"]
AUTH = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()


def api(path: str, **params) -> dict:
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{BASE}/api/public/{path}?{q}", headers={"Authorization": f"Basic {AUTH}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def observations(trace_id: str) -> list[dict]:
    out, page = [], 1
    while True:
        r = api("observations", traceId=trace_id, page=page, limit=100)
        out += r["data"]
        if page >= r["meta"]["totalPages"]:
            return out
        page += 1


def show_trace(t: dict) -> dict:
    obs = observations(t["id"])
    by_parent: dict[str | None, list[dict]] = {}
    for o in obs:
        by_parent.setdefault(o.get("parentObservationId"), []).append(o)
    t0 = min(ts(o["startTime"]) for o in obs)
    gens = [o for o in obs if o["type"] == "GENERATION"]
    errors = [o for o in obs if o.get("level") == "ERROR"]
    tokens = sum((o.get("usageDetails") or {}).get("total", 0) or 0 for o in gens)
    dur = max(ts(o["endTime"]) for o in obs if o.get("endTime")) - t0
    print(f"\n▶ {t['name']}  {dur.total_seconds():.2f}s  gen={len(gens)} tokens={tokens} errors={len(errors)}  {BASE}/project/{t['projectId']}/traces/{t['id']}")

    keep = {"GENERATION", "TOOL", "AGENT"}
    interesting_spans = {"simple_worker", "confirm_gate", "supervisor_agent", "intent_extractor", "aggregator", "followup_generator", "planner", "apply_chip", "stack_merger"}

    def walk(pid, depth):
        for o in sorted(by_parent.get(pid, []), key=lambda o: o["startTime"]):
            name = o.get("name") or ""
            show = o["type"] in keep or name in interesting_spans or o.get("level") == "ERROR"
            if show:
                s = (ts(o["startTime"]) - t0).total_seconds()
                e = (ts(o["endTime"]) - t0).total_seconds() if o.get("endTime") else float("nan")
                bar = f"{s:6.2f}→{e:6.2f}s"
                extra = ""
                if o["type"] == "GENERATION":
                    u = o.get("usageDetails") or {}
                    extra = f" [{o.get('model')}] in={u.get('input', 0)} out={u.get('output', 0)}"
                if o.get("level") == "ERROR":
                    extra += f"  ❌ {(o.get('statusMessage') or '')[:140]}"
                print(f"   {'  ' * depth}{bar}  {o['type'][:4]} {name}{extra}")
            walk(o["id"], depth + (1 if show else 0))

    walk(None, 0)
    # 병렬 확인: 같은 턴에서 worker 스팬이 시간상 겹치는지
    workers = [o for o in obs if o.get("name") in ("simple_worker", "supervisor_agent", "confirm_gate") and o.get("endTime")]
    overlaps = [(a["name"], b["name"]) for i, a in enumerate(workers) for b in workers[i + 1:]
                if ts(a["startTime"]) < ts(b["endTime"]) and ts(b["startTime"]) < ts(a["endTime"])]
    if len(workers) > 1:
        print(f"   ↳ worker 스팬 {len(workers)}개, 시간 겹침 {len(overlaps)}쌍 {'(병렬 실행 확인)' if overlaps else '(순차)'}")
    return {"trace": t["name"], "seconds": dur.total_seconds(), "generations": len(gens), "tokens": tokens, "errors": len(errors)}


def main(args: list[str]):
    traces = api("traces", limit=50)["data"]
    sessions = args or sorted({t["sessionId"] for t in traces if t.get("sessionId")})
    for sid in sessions:
        ts_ = sorted(api("traces", sessionId=sid, limit=50)["data"], key=lambda t: t["timestamp"])
        print(f"\n════ session {sid} ({len(ts_)} traces)")
        for t in ts_:
            show_trace(t)


if __name__ == "__main__":
    main(sys.argv[1:])
