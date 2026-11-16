"""상태 스키마, 의도 카탈로그, 규칙(planner / follow-up). LLM 없이 테스트 가능한 순수 로직만 둔다."""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

IntentName = Literal[
    "product_inquiry", "order_product", "change_shipping_address",
    "exchange_or_refund", "delivery_incident", "other",
]
Status = Literal["pending", "blocked", "awaiting_confirm", "running", "done", "failed", "cancelled"]

# 의도 카탈로그: action 모드일 때 필수 슬롯, 위험도, 처리 경로
CATALOG: dict[str, dict[str, Any]] = {
    "product_inquiry": {"required": [], "risk": "low", "route": "simple"},
    "order_product": {"required": ["product_id", "option"], "risk": "high", "route": "simple"},
    "change_shipping_address": {"required": ["scope", "address"], "risk": "low", "route": "simple"},
    "exchange_or_refund": {"required": [], "risk": "low", "route": "supervisor"},
    "delivery_incident": {"required": [], "risk": "low", "route": "supervisor"},
    "other": {"required": [], "risk": "low", "route": "simple"},
}
SLOT_LABEL = {"product_id": "상품", "option": "옵션", "scope": "변경 범위", "address": "새 주소"}
TERMINAL = {"done", "cancelled"}


# ---------- LLM 구조화 출력 ----------
class Slots(BaseModel):
    product_id: str | None = Field(None, description="상품 id. 최근 엔티티에 있으면 그 id를 쓴다")
    product_query: str | None = Field(None, description="상품을 찾기 위한 검색어 (id를 모를 때)")
    option: str | None = Field(None, description="'색상/사이즈' 형식. 예: 블랙/L")
    scope: Literal["this_order", "default"] | None = Field(None, description="배송지 변경 범위. 사용자가 명확히 말했을 때만")
    address: str | None = None
    order_id: str | None = None
    question: str | None = Field(None, description="교환/환불/배송사고 등 판단형 질문 원문")


class ExtractedIntent(BaseModel):
    intent: IntentName
    mode: Literal["action", "inquiry"] = Field(description="실행 요청이면 action, 가능 여부·정보만 묻는 질문이면 inquiry. 애매하면 inquiry")
    slots: Slots = Field(default_factory=Slots)
    existing_task_id: str | None = Field(None, description="열린 작업을 이어가는 발화면 그 작업 id")
    span: str = Field(description="이 의도에 해당하는 발화 조각")


class IntentResult(BaseModel):
    intents: list[ExtractedIntent]


class SubagentResult(BaseModel):
    summary: str = Field(description="사용자에게 전할 결론 2~3문장")
    data: dict = Field(default_factory=dict)
    needs_input: list[str] = Field(default_factory=list, description="결론을 내려면 사용자에게 더 물어야 할 것")
    proposed_actions: list[str] = Field(default_factory=list, description="사용자가 이어서 누를 버튼 문구. 최대 2개, 각 12자 이내 (예: '교환 신청하기')")


# ---------- 그래프 상태 ----------
class Task(TypedDict, total=False):
    id: str
    intent: str
    mode: str
    slots: dict
    missing: list[str]
    depends_on: list[str]
    risk: str
    route: str
    status: Status
    phase: str           # order_product: prepare -> checkout
    result: dict | None
    error: str | None
    converted: bool      # inquiry -> action 전환됨
    turn: int            # 마지막으로 바뀐 턴
    draft: dict          # order_product: 임시 주문 (재시도용)


class FollowUp(TypedDict):
    label: str
    type: Literal["recover", "confirm", "slot", "convert", "next"]
    payload: dict


def merge_tasks(left: list[Task] | None, right: list[Task] | None) -> list[Task]:
    """병렬 worker가 동시에 갱신해도 id 기준으로 합친다 (뒤에 온 필드가 이김)."""
    by_id: dict[str, Task] = {t["id"]: dict(t) for t in (left or [])}
    for t in right or []:
        by_id[t["id"]] = {**by_id.get(t["id"], {}), **t}
    return list(by_id.values())


def merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_id: str
    turn: int
    loops: int
    payload: dict | None           # 칩 클릭 시 구조화 명령
    task_stack: Annotated[list[Task], merge_tasks]
    entity_memory: Annotated[dict, merge_dict]
    followups: list[FollowUp]
    answer: str
    extracted: list[dict]          # intent_extractor -> stack_merger
    ready: list[str]               # planner -> fan_out


# ---------- 규칙 ----------
def merge_slots(old: dict, new: dict) -> dict:
    """같은 작업에 슬롯을 합친다. question은 덮어쓰지 않고 이어 붙인다 (한 문장에서 나뉜 하위 질문 보존)."""
    out = {**old}
    for k, v in new.items():
        if not v:
            continue
        if k == "question" and out.get(k) and v not in out[k]:
            out[k] = f"{out[k]} / {v}"
        else:
            out[k] = v
    return out


def compute_missing(task: Task) -> list[str]:
    if task["mode"] != "action":
        return []
    slots = task["slots"]
    missing = [s for s in CATALOG[task["intent"]]["required"] if not slots.get(s)]
    if "product_id" in missing and slots.get("product_query"):
        missing.remove("product_id")  # worker가 검색어로 상품을 찾는다
    return missing


def new_task(tid: str, ex: ExtractedIntent, turn: int, memory: dict) -> Task:
    slots = {k: v for k, v in ex.slots.model_dump().items() if v}
    # 지시어 해석 보강: "아까 그 후드티" 인데 LLM이 id를 못 채웠으면 최근 상품으로
    last = memory.get("last_product") or {}
    if ex.intent == "order_product" and not slots.get("product_id"):
        q = (slots.get("product_query") or "").replace(" ", "")
        # 검색어가 없거나 최근 상품을 가리킬 때만 메모리로 채운다 (다른 상품을 말했는데 최근 상품을 결제하면 사고)
        if last.get("product_id") and (not q or q in last.get("name", "").replace(" ", "")):
            slots["product_id"] = last["product_id"]
        if last.get("option") and slots.get("product_id") == last.get("product_id"):
            slots.setdefault("option", last.get("option"))
        slots = {k: v for k, v in slots.items() if v}
    mode = "action" if ex.intent == "order_product" else ex.mode
    t: Task = {
        "id": tid, "intent": ex.intent, "mode": mode, "slots": slots, "depends_on": [],
        "risk": CATALOG[ex.intent]["risk"], "route": CATALOG[ex.intent]["route"],
        "status": "pending", "result": None, "error": None, "converted": False, "turn": turn,
    }
    if ex.intent == "order_product":
        t["phase"] = "prepare"
    t["missing"] = compute_missing(t)
    return t


def link_dependencies(tasks: list[Task]) -> list[Task]:
    """주소 변경이 열려 있으면 결제는 그 뒤로 (주소 확정 -> 결제)."""
    open_addr = [t for t in tasks if t["intent"] == "change_shipping_address" and t["status"] not in TERMINAL | {"failed"}]
    changed = []
    for t in tasks:
        if t["intent"] == "order_product" and t["status"] not in TERMINAL:
            deps = sorted(set(t.get("depends_on", [])) | {a["id"] for a in open_addr})
            if deps != t.get("depends_on", []):
                changed.append({**t, "depends_on": deps})
    return changed


def deps_done(task: Task, tasks: list[Task]) -> bool:
    by_id = {t["id"]: t for t in tasks}
    return all(by_id.get(d, {}).get("status") in TERMINAL for d in task.get("depends_on", []))


def plan(tasks: list[Task]) -> tuple[list[Task], list[Task]]:
    """(지금 실행할 작업, 상태가 바뀐 작업)을 돌려준다.

    - pending: 바로 실행 (order_product의 prepare 단계는 의존성과 무관하게 실행: 돈이 안 나감)
    - blocked: 빈 슬롯이 채워졌고 의존성이 풀렸으면 pending으로
    - awaiting_confirm: 의존성이 풀리면 confirm 노드로 (interrupt)
    """
    ready, changed = [], []
    for t in tasks:
        st = t["status"]
        if st == "blocked" and not t.get("missing"):
            if t["intent"] == "order_product" and t.get("phase") == "checkout":
                if deps_done(t, tasks):
                    t = {**t, "status": "awaiting_confirm"}
                    changed.append(t)
            else:
                t = {**t, "status": "pending"}
                changed.append(t)
            st = t["status"]
        if st == "pending":
            if t["intent"] == "order_product" and t.get("phase") == "checkout" and not deps_done(t, tasks):
                continue
            ready.append(t)
        elif st == "awaiting_confirm" and deps_done(t, tasks):
            ready.append(t)
    return ready, changed


PRIORITY = {"recover": 0, "confirm": 1, "slot": 2, "convert": 3, "next": 4}


def rule_followups(tasks: list[Task]) -> list[FollowUp]:
    """반드시 떠야 하는 후속 질문(E 복구, A 슬롯, C 전환)은 규칙으로 만든다. B 확인은 interrupt가 맡는다."""
    out: list[FollowUp] = []
    for t in tasks:
        tid = t["id"]
        if t["status"] == "failed":
            if t["intent"] == "order_product":
                out += [
                    {"label": "다시 결제 시도", "type": "recover", "payload": {"op": "retry", "task_id": tid}},
                    {"label": "주문 취소", "type": "recover", "payload": {"op": "skip_task", "task_id": tid}},
                ]
            else:
                out.append({"label": "다시 시도", "type": "recover", "payload": {"op": "retry", "task_id": tid}})
        elif t["status"] == "blocked" and t.get("missing"):
            kind = "convert" if t.get("converted") else "slot"
            first = t["missing"][0]
            if t["intent"] == "change_shipping_address" and first == "scope":
                out += [
                    {"label": "이번 주문 배송지 변경", "type": kind, "payload": {"op": "fill_slots", "task_id": tid, "slots": {"scope": "this_order"}}},
                    {"label": "기본 배송지 변경", "type": kind, "payload": {"op": "fill_slots", "task_id": tid, "slots": {"scope": "default"}}},
                    {"label": "변경 안 함", "type": kind, "payload": {"op": "skip_task", "task_id": tid}},
                ]
            elif first == "product_id" and (t.get("result") or {}).get("candidates"):
                out += [
                    {"label": c["name"], "type": "slot", "payload": {"op": "fill_slots", "task_id": tid, "slots": {"product_id": c["product_id"]}}}
                    for c in t["result"]["candidates"][:3]
                ]
            elif first == "option" and (t.get("result") or {}).get("available"):
                out += [
                    {"label": opt, "type": "slot", "payload": {"op": "fill_slots", "task_id": tid, "slots": {"option": opt}}}
                    for opt in t["result"]["available"][:3]
                ]
            else:
                out.append({"label": f"{SLOT_LABEL.get(first, first)} 입력하기", "type": kind,
                            "payload": {"op": "ask_text", "task_id": tid, "slot": first}})
    out.sort(key=lambda f: PRIORITY[f["type"]])
    return out


def select_followups(rule: list[FollowUp], llm_next: list[FollowUp], limit: int = 3) -> list[FollowUp]:
    """우선순위 E > B > A > C > D. 막힌 일이 있으면 D(next)는 숨긴다."""
    blocking = [f for f in rule if f["type"] in ("recover", "confirm", "slot", "convert")]
    if blocking:
        # 같은 작업의 선택지는 묶어서 보여 준다 (첫 작업 우선)
        first_task = blocking[0]["payload"].get("task_id")
        same = [f for f in blocking if f["payload"].get("task_id") == first_task]
        return same[:limit]
    return llm_next[:limit]


def apply_payload(tasks: list[Task], payload: dict, turn: int) -> list[Task]:
    """칩 클릭(구조화 명령)을 작업 상태에 반영한다. 의도 추출을 건너뛴다."""
    by_id = {t["id"]: t for t in tasks}
    t = by_id.get(payload.get("task_id", ""))
    if not t:
        return []
    op = payload["op"]
    if op == "fill_slots":
        slots = {**t["slots"], **payload["slots"]}
        nt = {**t, "slots": slots, "mode": "action", "turn": turn}
        nt["missing"] = compute_missing(nt)
        if t["status"] == "blocked" and not nt["missing"]:
            nt["status"] = "pending"
        return [nt]
    if op == "skip_task":
        return [{**t, "status": "cancelled", "turn": turn}]
    if op == "retry":
        return [{**t, "status": "pending", "error": None, "turn": turn}]
    return []
