"""시나리오 러너: 멀티턴 대화를 재생하고, Langfuse에 턴 단위 트레이스를 남기고, 최종 상태를 검증한다.

uv run python -m followups.run              # 전체
uv run python -m followups.run hoodie       # 하나만
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.messages import HumanMessage
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.types import Command

from followups import db
from followups import llm as _llm  # noqa: F401  (.env 로드)
from followups.graph import build_graph
from followups.tools import open_toolbox

RUNS = Path(__file__).resolve().parents[2] / "runs"

# 단계: ("say", 텍스트) | ("chip", 칩 문구) | ("db", {파일: {키: 값}})
SCENARIOS: dict[str, dict] = {
    "hoodie": {
        "desc": "복합 의도(결제+배송지 문의) → 슬롯 채우기 → 의존성 해제 → HITL 결제",
        "steps": [
            ("say", "후드티 블랙 L 재고 있어?"),
            ("say", "아까 물어본 후드티 결제하고, 아참 배송지 변경도 지금 돼?"),
            ("chip", "이번 주문 배송지 변경"),
            ("say", "서울시 마포구 월드컵북로 400, 5층으로 해줘"),
            ("chip", "결제"),
        ],
        "expect": lambda log: _expect_paid_to("마포구"),
    },
    "parallel": {
        "desc": "서로 독립적인 두 의도가 같은 superstep에서 병렬 실행",
        "steps": [("say", "패딩 블랙 L 재고 있어? 그리고 기본 배송지를 부산시 해운대구 센텀중앙로 99로 바꿔줘")],
        "expect": lambda log: _expect_default_address("해운대구"),
    },
    "declined": {
        "desc": "결제 실패 → E 복구 칩 → 재시도 (같은 idempotency_key)",
        "db": {"payment": {"card_declined": True}},
        "steps": [
            ("say", "오버핏 후드티 그레이 L 바로 결제해줘"),
            ("chip", "결제"),
            ("db", {"payment": {"card_declined": False}}),
            ("chip", "다시 결제 시도"),
            ("chip", "결제"),
        ],
        "expect": lambda log: _expect_paid_to("강남구"),
    },
    "refund": {
        "desc": "판단형 의도 → supervisor + subagents (주문 → 약관 → 금액)",
        "steps": [("say", "지난달 산 후드티 사이즈가 안 맞는데, 교환이 나아 환불이 나아? 쿠폰 쓴 건 어떻게 돼?")],
        "expect": lambda log: _expect_refund_answer(log),
    },
}


def _expect_paid_to(fragment: str):
    with db.store("payment") as d:
        paid = [o for o in d["orders"].values() if o["status"] == "paid"]
    ok = len(paid) == 1 and fragment in paid[0]["address"]
    return ok, f"paid orders={len(paid)} address={paid[0]['address'] if paid else None}"


def _expect_refund_answer(log: dict):
    """주문 조회 → 약관 → 금액 계산을 모두 거쳐야 나오는 사실이 답변에 있어야 한다."""
    ans = log["turns"][-1]["answer"]
    need = {"환불액(카드 29,000 또는 합계 31,000)": "29,000" in ans or "31,000" in ans, "교환비 3,000원": "3,000" in ans, "쿠폰 언급": "쿠폰" in ans}
    return all(need.values()), f"checks={need}"


def _expect_default_address(fragment: str):
    with db.store("member") as d:
        addr = d["users"]["u1"]["default_address"]
    return fragment in addr, f"default_address={addr}"


def _patch_db(patch: dict):
    for name, kv in patch.items():
        with db.store(name, write=True) as d:
            d.update(kv)


async def run_scenario(name: str, spec: dict, app, lf) -> dict:
    db.reset(spec.get("db"))
    session_id = f"{name}-{uuid.uuid4().hex[:6]}"
    cfg_base = {"configurable": {"thread_id": session_id}}
    chips: list[dict] = []
    pending_interrupt = None
    log = {"scenario": name, "session_id": session_id, "turns": []}
    print(f"\n━━ {name}: {spec['desc']}  (session {session_id})")

    for i, (kind, value) in enumerate(spec["steps"], 1):
        if kind == "db":
            _patch_db(value)
            print(f"  [db] {value}")
            continue
        if kind == "say":
            inp = {"messages": [HumanMessage(content=value)], "user_id": "u1", "payload": None}
            shown = value
        else:
            chip = next((c for c in chips if c["label"] == value), None)
            if not chip:
                raise RuntimeError(f"칩 '{value}' 없음. 현재 칩: {[c['label'] for c in chips]}")
            p = chip["payload"]
            shown = f"[칩] {value}"
            if p["op"] == "confirm":
                # resume만 보내면 이전 턴 발화가 '이번 입력'으로 남아 답변이 엉뚱해진다 → 메시지도 함께 갱신
                inp = Command(resume=p, update={"messages": [HumanMessage(content=shown)]})
            elif p["op"] == "say":
                inp = {"messages": [HumanMessage(content=p["text"])], "payload": None}
            else:
                inp = {"messages": [HumanMessage(content=shown)], "payload": p}

        handler = CallbackHandler()
        t0 = time.perf_counter()
        with lf.start_as_current_observation(name=f"{name}/step-{i}", as_type="span", input=shown) as root, \
                propagate_attributes(session_id=session_id, user_id="u1", tags=[name], trace_name=f"{name}/step-{i}"):
            out = await app.ainvoke(inp, config={**cfg_base, "callbacks": [handler]})
            trace_id = lf.get_current_trace_id()
            intr = out.get("__interrupt__")
            if intr:
                pending_interrupt = intr[0].value
                bot, chips = pending_interrupt["message"], pending_interrupt["chips"]
            else:
                pending_interrupt = None
                bot, chips = out.get("answer", ""), out.get("followups", [])
            root.update(output={"answer": bot, "chips": [c["label"] for c in chips]})
        dt = time.perf_counter() - t0
        print(f"  👤 {shown}\n  🤖 {bot}\n     칩: {[c['label'] for c in chips]}   ({dt:.1f}s)")
        tasks = out.get("task_stack", [])
        log["turns"].append({"step": i, "input": shown, "answer": bot, "chips": chips, "seconds": round(dt, 2),
                             "trace_id": trace_id, "interrupt": bool(pending_interrupt),
                             "tasks": [{k: t.get(k) for k in ("id", "intent", "mode", "status", "phase", "missing", "depends_on", "slots", "error")} for t in tasks]})

    ok, detail = spec["expect"](log)
    log["expect"] = {"ok": ok, "detail": detail}
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} {detail}")
    return log


async def main(names: list[str]):
    lf = get_client()
    assert lf.auth_check(), "Langfuse 인증 실패 (.env 확인)"
    RUNS.mkdir(exist_ok=True)
    results = []
    async with AsyncExitStack() as stack:
        box = await open_toolbox(stack)
        app = build_graph(box)
        for n in names or list(SCENARIOS):
            try:
                results.append(await run_scenario(n, SCENARIOS[n], app, lf))
            except Exception as e:  # 시나리오 하나가 죽어도 나머지는 돌린다
                import traceback
                traceback.print_exc()
                results.append({"scenario": n, "error": repr(e)})
    lf.flush()
    out = RUNS / f"run-{time.strftime('%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n결과 저장: {out}")
    passed = sum(1 for r in results if r.get("expect", {}).get("ok"))
    print(f"합계: {passed}/{len(results)} PASS")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
