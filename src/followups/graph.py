"""메인 그래프: Custom workflow(바깥 틀) + Router(의도 추출 -> Send 병렬) + HITL + Follow-ups."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

import asyncio
import json
import os

from followups import attachments, guards, memory, sql_agent
from followups.llm import IS_OPENROUTER, MODEL_FAST, llm
from followups.state import (
    CATALOG, INTENT_SLOT_EXAMPLES, INTENT_SLOTS, TERMINAL, IntentResult, drop_ungrounded, Slots, State, Task, slot_model, apply_payload, attach_guidelines, compute_missing, expire_stale_consents,
    link_dependencies,
    merge_slots, merge_tasks, new_task, plan, refine_task, render_guidelines, rule_followups, select_followups,
)
from followups.supervisor import build_refund_fastpath, build_supervisor
from followups.tools import Toolbox

MAX_LOOPS = 6
# 2단계 의도 추출: 작은/국산 모델(Solar 등)은 큰 슬롯 스키마를 잘 못 채운다 → 의도 분리 후 의도별 스키마로 병렬 채움
TWO_STAGE_EXTRACT = os.getenv("TWO_STAGE_EXTRACT", "0" if IS_OPENROUTER else "1") == "1"
SLOT_FILL_METHOD = os.getenv("SLOT_FILL_METHOD", "json_schema")
PAYLOAD_OPS = {"fill_slots", "skip_task", "retry", "refine", "remember"}
# route -> worker 노드 이름. 새 처리 경로는 CATALOG의 route와 여기, 그리고 build_graph의 add_node만 늘리면 된다
ROUTE_NODES = {
    "simple": "simple_worker",
    "supervisor": "supervisor_agent",
    "sql": "sql_agent",
    "attachment": "attachment_worker",
    "report": "report_writer",
    "memory": "memory_writer",
}
WORKER_NODES = ["confirm_gate", *ROUTE_NODES.values()]
REFINABLE = {"data_query"}  # 끝난 작업을 조건만 바꿔 다시 돌릴 수 있는 의도


def _compact(obj: Any, limit: int = 600) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _open_tasks(tasks: list[Task]) -> list[Task]:
    return [t for t in tasks if t["status"] not in TERMINAL]


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    # 멀티모달 메시지: [{"type": "text", "text": ...}, {"type": "image_url", ...}] → 텍스트 부분만
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()


def _last_user_text(state: State) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return _text_of(m.content)
    return ""


def _text_only(messages: list) -> list:
    """의도 추출기에는 이미지를 다시 보내지 않는다 (첨부는 attachment_reader가 사실로 바꿔 둠)."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage) and not isinstance(m.content, str):
            out.append(HumanMessage(content=_text_of(m.content) or "[첨부]", id=m.id))
        else:
            out.append(m)
    return out


EXTRACTOR_HINTS = "\n".join(h for h in (sql_agent.EXTRACTOR_HINT, attachments.EXTRACTOR_HINT,
                                         memory.EXTRACTOR_HINT, guards.EXTRACTOR_HINT) if h)


class NextSteps(BaseModel):
    labels: list[str] = Field(description="사용자가 이어서 누를 만한 다음 행동 칩 문구 (최대 2개, 각 15자 이내)")


def build_graph(box: Toolbox, checkpointer=None):
    extractor = llm(MODEL_FAST).with_structured_output(IntentResult, method="function_calling")
    extractor_json = llm(MODEL_FAST).with_structured_output(IntentResult, method="json_schema")
    writer = llm(MODEL_FAST)
    next_steps = llm(MODEL_FAST).with_structured_output(NextSteps, method="function_calling")
    supervisor = build_supervisor(box)
    refund_fastpath = build_refund_fastpath(box)

    # ---------- ① 턴 시작 / 입력 가드 / 진입 분기 ----------
    def begin_turn(state: State) -> dict:
        turn = state.get("turn", 0) + 1
        return {"turn": turn, "loops": 0, "followups": [], "answer": "", "guard": None,
                "profile": memory.load_profile(state.get("user_id")),
                "task_stack": expire_stale_consents(state.get("task_stack", []), turn)}

    def route_entry(state: State) -> str:
        if (state.get("guard") or {}).get("blocked"):
            return "policy_reply"
        p = state.get("payload")
        if p and p.get("op") in PAYLOAD_OPS:
            return "apply_chip"
        return "attachment_reader" if attachments.has_attachment(state) else "intent_extractor"

    def apply_chip(state: State) -> dict:
        changed = apply_payload(state["task_stack"], state["payload"], state["turn"])
        merged = merge_tasks(state["task_stack"], changed)
        return {"task_stack": changed + link_dependencies(merged), "payload": None}

    async def intent_extractor(state: State, config: RunnableConfig) -> dict:
        open_tasks = [
            {k: t.get(k) for k in ("id", "intent", "mode", "status", "slots", "missing")}
            for t in _open_tasks(state.get("task_stack", []))
        ]
        sys = SystemMessage(content=(
            "너는 커머스 챗봇의 의도 추출기다. 마지막 사용자 발화를 의도 단위로 모두 나눈다.\n"
            "- 한 문장에 여러 요청이 있으면 각각 별도 의도로.\n"
            "- '가능해?', '돼?'처럼 가능 여부만 묻는 것은 mode=inquiry. 명확히 해 달라는 요청만 action.\n"
            "- '아까 그거'처럼 지칭하면 최근 엔티티의 id를 slots에 채운다.\n"
            "- 열린 작업의 빈 슬롯을 채우는 발화면 existing_task_id에 그 id를 넣고 같은 intent로.\n"
            "- 사용자가 말하지 않은 슬롯 값은 추측하지 않는다 (특히 scope).\n"
            f"{EXTRACTOR_HINTS}\n"
            f"최근 엔티티: {_compact(state.get('entity_memory', {}))}\n"
            f"첨부 사실: {_compact({k: v.get('facts') for k, v in (state.get('attachments') or {}).items()}, 1500)}\n"
            f"열린 작업: {_compact(open_tasks, 1500)}"
        ))
        recent = _text_only(state["messages"][-6:])
        res: IntentResult | None = None
        # 작은 모델은 가끔 도구 호출을 건너뛰어 None을 준다 → json_schema로 한 번 더, 그래도 없으면 'other'
        for ex_model, name in ((extractor, "intent_extractor.llm"), (extractor_json, "intent_extractor.retry")):
            try:
                res = await ex_model.ainvoke([sys, *recent], config={**config, "run_name": name})
            except Exception:
                res = None
            if res is not None and res.intents:
                break
        if res is None or not res.intents:
            from followups.state import ExtractedIntent

            res = IntentResult(intents=[ExtractedIntent(intent="other", mode="inquiry", span=_last_user_text(state))])
        intents = res.intents
        if TWO_STAGE_EXTRACT and intents:
            intents = await asyncio.gather(*(_fill_slots(i, sys, recent, config) for i in intents))
        # 근거 없는 값(사용자가 고르지 않은 DB 출처, 말하지 않은 배송지 범위 등)은 규칙으로 지운다
        text = _last_user_text(state)
        intents = [i.model_copy(update={"slots": Slots(**drop_ungrounded(i.slots.model_dump(), text))}) for i in intents]
        return {"extracted": [i.model_dump() for i in intents]}

    async def _fill_slots(ex, sys, recent, config):
        names = INTENT_SLOTS.get(ex.intent, [])
        if not names:
            return ex
        schema = slot_model(ex.intent)
        examples = "\n".join(f"- '{u}' → {json.dumps(v, ensure_ascii=False)}" for u, v in INTENT_SLOT_EXAMPLES.get(ex.intent, []))
        ask = SystemMessage(content=(
            f"{sys.content}\n\n"
            f"지금은 슬롯 채우기 단계다. 의도 '{ex.intent}'(발화 조각: '{ex.span}')에 대해 "
            f"아래 필드만 채운다: {', '.join(names)}.\n"
            "- 발화·최근 엔티티·첨부 사실·열린 작업에 근거가 있는 값만. 근거 없으면 비운다 (기본값·추측 금지).\n"
            "- 검색어(product_query)는 상품 이름만. 발화 전체를 넣지 않는다. 옵션은 '색상/사이즈' 형식.\n"
            f"예시:\n{examples}"
        ))
        got = None
        # 작은 스키마는 json_schema가 더 잘 채운다 (Solar 실측: function_calling은 option·address를 비움)
        for method in (SLOT_FILL_METHOD, "function_calling"):
            try:
                got = await llm(MODEL_FAST).with_structured_output(schema, method=method).ainvoke(
                    [ask, *recent], config={**config, "run_name": f"slot_filler.{ex.intent}"})
                break
            except Exception:
                continue
        if got is None:
            return ex  # 슬롯 채우기 실패 시 1단계 결과로 (빈칸은 칩으로 묻는다)
        filled = {k: v for k, v in got.model_dump().items()
                  if v not in (None, "", []) and not (isinstance(v, str) and v[:1] in "<(")}  # 예시 자리표시자 복사 방지
        base = {k: v for k, v in ex.slots.model_dump().items() if v not in (None, "", [])}
        return ex.model_copy(update={"slots": Slots(**{**base, **filled})})

    # ---------- ② 스택 병합 ----------
    def _refine_target(ex, tasks: list[Task]):
        """'구별로도 나눠줘'처럼 필수 슬롯 없이 조건만 바꾸는 후속 발화 → 가장 최근에 끝난 같은 의도 작업.

        모델이 existing_task_id를 못 채워도 구조(필수 슬롯이 비었고, 같은 의도의 끝난 작업이 있음)로 판단한다.
        """
        done = [t for t in tasks if t["intent"] == ex.intent and t["status"] == "done"]
        if not done:
            return None
        last = max(done, key=lambda t: t.get("turn", 0))
        required = CATALOG[ex.intent]["required"]
        given = {k: v for k, v in ex.slots.model_dump().items() if v not in (None, "", [], "auto")}
        # 필수 슬롯을 새로 말했으면(다른 지표) 새 조회. 모델이 앞 작업 값을 그대로 옮겨 적은 것은 새 값이 아니다
        if any(r in given and given[r] != last["slots"].get(r) for r in required):
            return None
        changes = {k for k, v in given.items() if k not in required and v != last["slots"].get(k)}
        return last if changes else None

    def stack_merger(state: State) -> dict:
        from followups.state import ExtractedIntent

        tasks = list(state.get("task_stack", []))
        turn = state["turn"]
        changed: list[Task] = []
        for raw in state.get("extracted", []):
            ex = ExtractedIntent.model_validate(raw)
            by_id = {t["id"]: t for t in merge_tasks(tasks, changed)}
            target = by_id.get(ex.existing_task_id or "")
            if not target:  # 같은 의도의 열린 작업이 있으면 이어 붙임 (조회 의도 제외)
                target = next((t for t in _open_tasks(list(by_id.values()))
                               if t["intent"] == ex.intent and ex.intent not in ("product_inquiry", "other")), None)
            if not target and ex.intent in REFINABLE:
                target = _refine_target(ex, list(by_id.values()))
            if target and target["intent"] == ex.intent and target["status"] == "done" and (
                    ex.existing_task_id or ex.intent in REFINABLE):
                # 끝난 작업을 조건만 바꿔 다시 ('구별로도 나눠줘'): 그 결과를 쓴 보고서도 다시 돈다
                changed += refine_task(merge_tasks(tasks, changed), target, ex.slots.model_dump(), turn)
                continue
            if target and target["intent"] == ex.intent:
                slots = merge_slots(target["slots"], ex.slots.model_dump())
                mode = "action" if (ex.mode == "action" or target["mode"] == "action") else "inquiry"
                t = {**target, "slots": slots, "mode": mode, "turn": turn}
                t["missing"] = compute_missing(t)
                if t["status"] == "blocked" and not t["missing"] and t["intent"] != "order_product":
                    t["status"] = "pending"
            else:
                t = new_task(f"T{len(by_id) + 1}", ex, turn, state.get("entity_memory", {}))
                if t["missing"]:
                    t["status"] = "blocked"
                # 정책을 넘는 선호(예: 개인정보 표시)는 동의 칩을 띄우기 전에 거른다
                if t["intent"] == "set_preference" and memory.policy_violation(t["slots"].get("pref_key"), t["slots"].get("pref_value")):
                    t = {**t, "status": "cancelled", "error": "policy_override", "missing": []}
            changed.append(t)
        merged = merge_tasks(tasks, changed)
        return {"task_stack": changed + link_dependencies(merged), "extracted": []}

    # ---------- ③ planner + 팬아웃 ----------
    def planner(state: State) -> dict:
        ready, changed = plan(state["task_stack"])
        # 실행할 작업에만 필요한 가이드라인 id를 붙인다 (멀티가이드라인)
        guided = attach_guidelines(ready, state.get("profile"))
        return {"task_stack": changed + guided, "loops": state.get("loops", 0) + 1, "ready": [t["id"] for t in ready]}

    def fan_out(state: State):
        if state["loops"] > MAX_LOOPS or not state.get("ready"):
            return ["aggregator", "followup_generator"]  # 서로 독립: 병렬로 돌려 턴 지연을 줄인다
        by_id = {t["id"]: t for t in state["task_stack"]}
        profile = state.get("profile") or {}
        sends = []
        for tid in state["ready"]:
            t = by_id[tid]
            inputs = {d: {k: by_id[d].get(k) for k in ("intent", "status", "slots", "result", "sql")}
                      for d in t.get("depends_on", []) if d in by_id}
            arg = {"task": t, "user_id": state["user_id"], "turn": state["turn"], "utterance": _last_user_text(state),
                   "inputs": inputs, "attachments": state.get("attachments") or {}, "profile": profile,
                   "guidelines_text": render_guidelines(t.get("guidelines", []), profile)}
            if t["intent"] == "order_product" and t.get("phase") == "checkout":
                sends.append(Send("confirm_gate", arg))
            else:
                sends.append(Send(ROUTE_NODES.get(t["route"], "simple_worker"), arg))
        return sends

    # ---------- ④ workers ----------
    async def simple_worker(arg: dict, config: RunnableConfig) -> dict:
        t: Task = {**arg["task"], "turn": arg["turn"]}
        uid, s = arg["user_id"], t["slots"]
        mem: dict = {}
        intent = t["intent"]

        if intent == "product_inquiry":
            q = s.get("product_query") or s.get("product_id") or arg["utterance"]
            r = await box.call("search_product", {"query": q}, config)
            hits = r.get("results", [])
            t.update(status="done", result={"products": hits, "asked_option": s.get("option")})
            if hits:
                p = hits[0]
                opt = s.get("option")
                mem["last_product"] = {"product_id": p["product_id"], "name": p["name"],
                                       "option": opt if opt and p["stock"].get(opt, 0) > 0 else None}

        elif intent == "order_product":  # prepare 단계: 임시 주문만 (돈 안 나감)
            if not s.get("product_id"):  # 검색어 → 상품 id 확정
                hits = (await box.call("search_product", {"query": s["product_query"]}, config)).get("results", [])
                if len(hits) != 1:
                    t.update(status="blocked", missing=["product_id"], error="product_not_found" if not hits else "ambiguous_product",
                             result={"candidates": [{"product_id": h["product_id"], "name": h["name"]} for h in hits]})
                    return {"task_stack": [t], "entity_memory": mem}
                s = {**s, "product_id": hits[0]["product_id"]}
                t["slots"] = s
                mem["last_product"] = {"product_id": hits[0]["product_id"], "name": hits[0]["name"], "option": s.get("option")}
            r = await box.call("create_draft_order", {"user_id": uid, "product_id": s["product_id"], "option": s["option"]}, config)
            if "error" in r:
                slots = {k: v for k, v in s.items() if k != "option"}
                t.update(status="blocked", slots=slots, missing=["option"], result=r, error=r["error"])
            else:
                t.update(phase="checkout", status="blocked", missing=[], result=r, error=None)

        elif intent == "change_shipping_address":
            if t["mode"] == "inquiry":
                r = await box.call("check_address_change", {"user_id": uid}, config)
                if r.get("possible"):
                    t.update(mode="action", converted=True, result=r)
                    t["missing"] = compute_missing(t)
                    t["status"] = "blocked" if t["missing"] else "pending"
                else:
                    t.update(status="done", result=r)
            else:
                tool = "set_checkout_address" if s["scope"] == "this_order" else "update_default_address"
                r = await box.call(tool, {"user_id": uid, "address": s["address"]}, config)
                t.update(status="done" if r.get("ok") else "failed", result=r)

        else:  # other: 도구 없이 짧게 답
            msg = await writer.ainvoke(
                [SystemMessage(content="커머스 챗봇. 한두 문장으로 답한다. 할 수 없는 일은 할 수 없다고 말한다."),
                 HumanMessage(content=arg["utterance"])], config={**config, "run_name": "other.llm"})
            t.update(status="done", result={"answer": msg.content})

        return {"task_stack": [t], "entity_memory": mem}

    async def confirm_gate(arg: dict, config: RunnableConfig) -> dict:
        """B 실행 확인: 되돌릴 수 없는 결제는 interrupt로 사용자 확인을 받은 뒤 실행."""
        t: Task = {**arg["task"], "turn": arg["turn"]}
        draft = t["result"] if t.get("result", {}).get("draft_id") else t.get("draft") or {}
        member = await box.call("get_member", {"user_id": arg["user_id"]}, config)
        address = member.get("checkout_address") or member.get("default_address")
        decision = interrupt({
            "type": "confirm",
            "task_id": t["id"],
            "message": f"{draft.get('product')} {draft.get('option')}, {draft.get('price', 0):,}원을 '{address}'(으)로 배송받도록 결제할까요?",
            "chips": [
                {"label": "결제", "type": "confirm", "payload": {"op": "confirm", "task_id": t["id"], "confirm": True}},
                {"label": "취소", "type": "confirm", "payload": {"op": "confirm", "task_id": t["id"], "confirm": False}},
            ],
        })
        if not decision.get("confirm"):
            t.update(status="cancelled", result={**draft, "cancelled_by_user": True})
            return {"task_stack": [t]}
        thread = config["configurable"]["thread_id"]
        r = await box.call("checkout", {"draft_id": draft["draft_id"], "idempotency_key": f"{thread}:{t['id']}"}, config)
        if "error" in r:
            t.update(status="failed", error=r["error"], result=draft, draft=draft)
        else:
            t.update(status="done", result={**r, "product": draft.get("product"), "option": draft.get("option")}, error=None, draft=draft)
        return {"task_stack": [t]}

    async def supervisor_agent(arg: dict, config: RunnableConfig) -> dict:
        t: Task = {**arg["task"], "turn": arg["turn"]}
        q = t["slots"].get("question") or arg["utterance"]
        if t["intent"] == "exchange_or_refund":
            fast = await refund_fastpath(arg["user_id"], q, arg["utterance"], t["slots"].get("product_query"), config)
            if fast:
                t.update(status="done", result={**fast.model_dump(), "path": "fastpath"})
                return {"task_stack": [t]}
        out = await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": f"user_id={arg['user_id']}\n질문: {q}\n사용자 원문: {arg['utterance']}"}]},
            config={**config, "run_name": "supervisor"},
        )
        res = out.get("structured_response")
        result = res.model_dump() if res else {"summary": out["messages"][-1].content}
        t.update(status="done", result=result)
        return {"task_stack": [t]}

    # ---------- ⑤ 합성 / ⑥ 후속 질문 ----------
    async def aggregator(state: State, config: RunnableConfig) -> dict:
        turn = state["turn"]
        touched = [t for t in state["task_stack"] if t.get("turn") == turn]
        # 보고서 작업이 이번 턴에 끝났으면 그 본문이 곧 답이다 (앞 작업 결과를 이미 종합함)
        report = next((t for t in touched if t["route"] == "report" and t["status"] == "done"), None)
        if report and (report.get("result") or {}).get("summary"):
            summary = report["result"]["summary"]
            return {"answer": summary, "messages": [AIMessage(content=summary)]}
        # supervisor 결과는 이미 사용자용 요약이므로 다시 요약하지 않는다 (LLM 호출 1회, 약 4초 절약)
        if len(touched) == 1 and touched[0]["route"] == "supervisor" and touched[0]["status"] == "done":
            summary = (touched[0]["result"] or {}).get("summary", "")
            return {"answer": summary, "messages": [AIMessage(content=summary)]}
        brief = [
            {"id": t["id"], "intent": t["intent"], "mode": t["mode"], "status": t["status"],
             "missing": t.get("missing"), "error": t.get("error"), "result": t.get("result")}
            for t in touched
        ]
        msg = await writer.ainvoke([
            SystemMessage(content=(
                "너는 커머스 챗봇의 답변 작성기다. 아래 작업 결과만 근거로 한국어 2~3문장으로 답한다.\n"
                "- status가 done이 아닌 일은 했다고 말하지 않는다. blocked는 무엇이 더 필요한지만 짧게.\n"
                "- 질문으로 끝내지 않는다 (선택지는 버튼으로 따로 보여 준다). 이모지 금지.\n"
                f"작업 결과: {_compact(brief, 3000)}"
            )),
            HumanMessage(content=_last_user_text(state)),
        ], config={**config, "run_name": "aggregator.llm"})
        return {"answer": msg.content, "messages": [AIMessage(content=msg.content)]}

    async def followup_generator(state: State, config: RunnableConfig) -> dict:
        tasks = state["task_stack"]
        rules = rule_followups(_open_tasks(tasks))  # failed도 열린 작업에 포함됨 (중복 칩 버그 수정)
        llm_next = []
        turn_done = [t for t in tasks if t.get("turn") == state["turn"] and t["status"] == "done"]
        if not rules and turn_done:
            sup = [a for t in turn_done if t["route"] == "supervisor" for a in (t["result"] or {}).get("proposed_actions", [])]
            if sup:
                labels = [l if len(l) <= 20 else l[:19] + "…" for l in sup[:2]]
            else:
                ns: NextSteps = await next_steps.ainvoke([
                    SystemMessage(content="방금 끝난 작업을 보고, 사용자가 이어서 할 만한 다음 행동 칩을 최대 2개 제안한다. 이미 한 일은 제외."),
                    HumanMessage(content=_compact([{"intent": t["intent"], "result": t["result"]} for t in turn_done], 1500)),
                ], config={**config, "run_name": "followups.next.llm"})
                labels = ns.labels
            llm_next = [{"label": l, "type": "next", "payload": {"op": "say", "text": l}} for l in labels]
        return {"followups": select_followups(rules, llm_next)}

    g = StateGraph(State)
    g.add_node("begin_turn", begin_turn)
    g.add_node("input_guard", guards.build_input_guard())
    g.add_node("policy_reply", guards.build_policy_reply())
    g.add_node("apply_chip", apply_chip)
    g.add_node("attachment_reader", attachments.build_attachment_reader())
    g.add_node("intent_extractor", intent_extractor)
    g.add_node("stack_merger", stack_merger)
    g.add_node("planner", planner)
    g.add_node("simple_worker", simple_worker)
    g.add_node("confirm_gate", confirm_gate)
    g.add_node("supervisor_agent", supervisor_agent)
    g.add_node("sql_agent", sql_agent.build_sql_worker(box))
    g.add_node("attachment_worker", attachments.build_attachment_worker(box))
    g.add_node("report_writer", attachments.build_report_writer(box))
    g.add_node("memory_writer", memory.build_memory_writer())
    g.add_node("aggregator", aggregator)
    g.add_node("output_guard", guards.build_output_guard())
    g.add_node("followup_generator", followup_generator)

    g.add_edge(START, "begin_turn")
    g.add_edge("begin_turn", "input_guard")
    g.add_conditional_edges("input_guard", route_entry, ["policy_reply", "apply_chip", "attachment_reader", "intent_extractor"])
    g.add_edge("policy_reply", END)
    g.add_edge("attachment_reader", "intent_extractor")
    g.add_edge("intent_extractor", "stack_merger")
    g.add_edge("stack_merger", "planner")
    g.add_edge("apply_chip", "planner")
    g.add_conditional_edges("planner", fan_out, [*WORKER_NODES, "aggregator", "followup_generator"])
    for w in WORKER_NODES:
        g.add_edge(w, "planner")  # 작업이 끝나면 다시 planner: 풀린 의존성을 같은 턴에 이어서 처리
    g.add_edge("aggregator", "output_guard")
    g.add_edge("output_guard", END)
    g.add_edge("followup_generator", END)
    return g.compile(checkpointer=checkpointer or make_checkpointer())


def make_checkpointer():
    """단기 메모리(스레드 상태) 저장소. 기본은 프로세스 메모리.

    서비스에서는 Postgres를 쓴다 (langgraph-checkpoint-postgres):
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()
            app = build_graph(box, checkpointer=saver)
    interrupt로 멈춘 턴을 재개하려면 상태가 남아 있어야 하므로, 보존 기간은 화면 세션 만료(예: 30분)와 따로 정한다.
    """
    return InMemorySaver()
