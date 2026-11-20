"""상태 스키마, 의도 카탈로그, 규칙(planner / follow-up). LLM 없이 테스트 가능한 순수 로직만 둔다."""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

IntentName = Literal[
    "product_inquiry", "order_product", "change_shipping_address",
    "exchange_or_refund", "delivery_incident",
    "data_query", "attachment_analysis", "report_compose", "set_preference",
    "other",
]
Status = Literal["pending", "blocked", "awaiting_confirm", "running", "done", "failed", "cancelled"]

# 의도 카탈로그
#   required: action 모드일 때 필수 슬롯
#   risk:     high면 실행 전 사용자 확인
#   route:    처리 노드 (graph.ROUTE_NODES)
#   produces / consumes: 산출물 종류. consumes가 겹치는 열린 생산 작업이 있으면 그 뒤로 미룬다 (link_dependencies)
#   force_action: 문의(inquiry)로 추출돼도 action으로 다룬다
CATALOG: dict[str, dict[str, Any]] = {
    "product_inquiry": {"required": [], "risk": "low", "route": "simple"},
    "order_product": {"required": ["product_id", "option"], "risk": "high", "route": "simple",
                      "consumes": ["shipping_address"], "force_action": True},
    "change_shipping_address": {"required": ["scope", "address"], "risk": "low", "route": "simple",
                                "produces": ["shipping_address"]},
    "exchange_or_refund": {"required": [], "risk": "low", "route": "supervisor"},
    "delivery_incident": {"required": [], "risk": "low", "route": "supervisor"},
    # 멀티스키마 Text-to-SQL: 무엇(metric)을 모르면 칩으로 묻는다. 어느 DB(source)는 worker가 스키마 인덱스로 고르고,
    # 비슷한 후보가 여럿이면 그때 선택지 칩으로 묻는다 (required에 두면 worker가 불리지 않아 후보를 못 만든다)
    "data_query": {"required": ["metric"], "risk": "low", "route": "sql",
                   "produces": ["dataset"], "force_action": True},
    # 첨부(이미지) 해석: 입구에서 attachment_reader가 사실을 뽑고, 이 작업은 그 사실을 정리한다
    "attachment_analysis": {"required": ["attachment_id"], "risk": "low", "route": "attachment",
                            "produces": ["attachment_facts"], "force_action": True},
    # 앞 작업 결과를 가이드라인(톤·양식)에 맞춰 합성
    "report_compose": {"required": [], "risk": "low", "route": "report",
                       "consumes": ["dataset", "attachment_facts"], "force_action": True},
    # 장기 메모리 기록: 사용자 동의(consent) 칩을 거쳐야만 쓴다
    "set_preference": {"required": ["pref_key", "pref_value", "consent"], "risk": "low", "route": "memory",
                       "force_action": True},
    "other": {"required": [], "risk": "low", "route": "simple"},
}
# 의도 프로필: 추출기가 고를 수 있는 의도 집합. 도메인이 다른 의도가 섞이면 작은 모델이 헷갈린다.
# 도메인 플러그인(followups.domains.register)이 자기 프로필과 의도를 여기에 더한다
INTENT_PROFILES: dict[str, list[str]] = {
    "all": list(CATALOG),
    "commerce": ["product_inquiry", "order_product", "change_shipping_address", "exchange_or_refund", "delivery_incident",
                 "data_query", "attachment_analysis", "report_compose", "set_preference", "other"],
}
SLOT_LABEL = {
    "product_id": "상품", "option": "옵션", "scope": "변경 범위", "address": "새 주소",
    "metric": "지표", "source": "데이터 출처", "period": "기간", "region": "지역", "group_by": "구분 기준",
    "attachment_id": "첨부", "tone": "문체", "pref_key": "설정 항목", "pref_value": "설정 값", "consent": "적용 여부", "y_unit": "단위",
}
LIST_SLOTS = {"group_by"}  # 병합 시 합집합
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
    # data_query (Text-to-SQL)
    metric: str | None = Field(None, description="조회할 지표. 예: 점포 수, 매출액")
    source: str | None = Field(None, description="데이터 출처(DB) id. 사용자가 명확히 고른 경우만")
    period: str | None = Field(None, description="기간. 'YYYY-MM~YYYY-MM' 형식")
    region: str | None = Field(None, description="지역. 예: 마포구")
    group_by: list[str] | None = Field(None, description="구분 기준. 예: ['업종'], ['구']")
    # attachment_analysis / report_compose
    attachment_id: str | None = Field(None, description="첨부 id. '아까 보낸 차트'면 최근 첨부 id")
    tone: str | None = Field(None, description="보고서 문체/양식. 예: 3분기 보고서")
    # set_preference
    pref_key: str | None = Field(None, description="영구 선호 항목. 예: decimal_places, table_format")
    pref_value: str | None = Field(None, description="선호 값. 예: 2")
    y_unit: str | None = Field(None, description="첨부 차트의 값 단위를 사용자가 알려 준 경우. 예: 백만원")


# 의도별로 채울 수 있는 슬롯 (2단계 추출: 의도 분리 → 의도별 작은 스키마로 슬롯 채우기)
INTENT_SLOTS: dict[str, list[str]] = {
    "product_inquiry": ["product_query", "product_id", "option"],
    "order_product": ["product_id", "product_query", "option"],
    "change_shipping_address": ["scope", "address"],
    "exchange_or_refund": ["question", "product_query", "order_id"],
    "delivery_incident": ["question", "order_id"],
    "data_query": ["metric", "region", "period", "group_by", "source"],
    "attachment_analysis": ["attachment_id", "y_unit"],
    "report_compose": ["tone", "attachment_id"],
    "set_preference": ["pref_key", "pref_value"],
    "other": [],
}


# 슬롯 채우기 예시 (작은 모델용 few-shot). 발화 조각 → 채울 값
INTENT_SLOT_EXAMPLES: dict[str, list[tuple[str, dict]]] = {
    "product_inquiry": [("패딩 블랙 L 재고 있어?", {"product_query": "패딩", "option": "블랙/L"})],
    # '아까 그 후드티'처럼 지칭만 하면 비워 둔다: new_task가 최근 엔티티로 채운다 (규칙)
    "order_product": [("오버핏 후드티 그레이 L 바로 결제해줘", {"product_query": "오버핏 후드티", "option": "그레이/L"}),
                      ("아까 물어본 후드티 결제해줘", {"product_query": "후드티"})],
    "change_shipping_address": [("기본 배송지를 부산시 해운대구 센텀중앙로 99로 바꿔줘",
                                 {"scope": "default", "address": "부산시 해운대구 센텀중앙로 99"}),
                                ("서울시 마포구 월드컵북로 400, 5층으로 해줘", {"address": "서울시 마포구 월드컵북로 400, 5층"})],
    "exchange_or_refund": [("지난달 산 후드티 교환이 나아 환불이 나아?", {"question": "교환이 나은지 환불이 나은지", "product_query": "후드티"})],
    "data_query": [("마포구 업종별 점포 수 뽑아줘", {"metric": "점포 수", "region": "마포구", "group_by": ["업종"]}),
                   ("구별로도 나눠줘", {"group_by": ["구"]})],
    "report_compose": [("3분기 보고서 톤으로 정리해줘", {"tone": "3분기 보고서"})],
    "set_preference": [("앞으로 숫자는 반올림하지 말고 소수점 둘째 자리까지 보여줘", {"pref_key": "decimal_places", "pref_value": "2"})],
}

# 발화에 근거 단어가 없으면 버리는 슬롯 (작은 모델이 지어내기 쉬운 값). 칩·참조로 채워지는 경우는 해당 없음
SLOT_EVIDENCE: dict[str, tuple[str, ...]] = {
    "source": ("상권", "사업자", "sangkwon", "bizreg"),
    "scope": ("이번", "기본", "default", "this_order"),
    "period": ("월", "분기", "년", "개월", "기간", "~", "-"),
}


def drop_ungrounded(slots: dict, text: str) -> dict:
    """근거 단어가 발화에 없는 슬롯 값을 지운다 ('auto'는 '모름'이라는 뜻이라 그대로 둔다)."""
    out = dict(slots)
    for k, words in SLOT_EVIDENCE.items():
        v = out.get(k)
        if v and v != "auto" and not any(w in text for w in words):
            out.pop(k)
    return out


def slot_model(intent: str) -> type[BaseModel]:
    """그 의도의 슬롯만 가진 구조화 출력 스키마. 작은 모델이 큰 스키마에서 값을 빠뜨리거나 지어내는 것을 줄인다."""
    from pydantic import create_model

    fields = {k: (Slots.model_fields[k].annotation, Slots.model_fields[k]) for k in INTENT_SLOTS.get(intent, [])}
    return create_model(f"Slots_{intent}", **fields)


class ExtractedIntent(BaseModel):
    intent: str = Field(description="의도 이름")
    mode: Literal["action", "inquiry"] = Field(description="실행 요청이면 action, 가능 여부·정보만 묻는 질문이면 inquiry. 애매하면 inquiry")
    slots: Slots = Field(default_factory=Slots)
    existing_task_id: str | None = Field(None, description="열린 작업을 이어가는 발화면 그 작업 id")
    span: str = Field(description="이 의도에 해당하는 발화 조각")


class IntentResult(BaseModel):
    intents: list[ExtractedIntent]


def intent_result_model(allowed: list[str]) -> type[IntentResult]:
    """프로필의 의도만 고를 수 있는 추출 스키마 (결과는 ExtractedIntent의 하위 클래스라 그대로 쓴다)."""
    from pydantic import create_model

    lit = Literal[tuple(allowed)]  # type: ignore[valid-type]
    item = create_model("ExtractedIntentP", __base__=ExtractedIntent, intent=(lit, ...))
    return create_model("IntentResultP", __base__=IntentResult, intents=(list[item], ...))


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
    guidelines: list[str]  # 이 작업에 붙은 가이드라인 id (planner가 부착)
    sql: dict            # data_query: {"tables", "query", "rows"}
    result_ref: str      # 대용량 결과 참조 키


class FollowUp(TypedDict):
    label: str
    type: Literal["recover", "policy", "confirm", "slot", "convert", "next"]
    payload: dict


def merge_tasks(left: list[Task] | None, right: list[Task] | None) -> list[Task]:
    """병렬 worker가 동시에 갱신해도 id 기준으로 합친다 (뒤에 온 필드가 이김)."""
    by_id: dict[str, Task] = {t["id"]: dict(t) for t in (left or [])}
    for t in right or []:
        by_id[t["id"]] = {**by_id.get(t["id"], {}), **t}
    return list(by_id.values())


def merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


def add_list(left: list | None, right: list | None) -> list:
    return [*(left or []), *(right or [])]


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
    attachments: Annotated[dict, merge_dict]  # attachment_id -> {kind, facts, object_key, turn}
    profile: dict                  # 확정된 장기 메모리 {key: value} (begin_turn에서 로드, 읽기 전용)
    guard: dict | None             # input_guard 판정 {blocked, reason, alternatives}
    policy_flags: Annotated[list[dict], add_list]  # 가드 판정 기록 (감사용, 누적)


# ---------- 규칙 ----------
CONSENT_TTL_TURNS = 2


def expire_stale_consents(tasks: list[Task], turn: int) -> list[Task]:
    """답하지 않은 메모리 동의 칩이 다음 단계 칩을 계속 가리지 않도록, N턴이 지나면 자동 취소한다."""
    return [
        {**t, "status": "cancelled", "error": "consent_expired"}
        for t in tasks
        if t["intent"] == "set_preference" and t["status"] == "blocked" and "consent" in (t.get("missing") or [])
        and turn - t.get("turn", turn) >= CONSENT_TTL_TURNS
    ]


def merge_slots(old: dict, new: dict) -> dict:
    """같은 작업에 슬롯을 합친다. question은 덮어쓰지 않고 이어 붙인다 (한 문장에서 나뉜 하위 질문 보존)."""
    out = {**old}
    for k, v in new.items():
        if not v and v is not False:
            continue
        if k in LIST_SLOTS:
            prev = out.get(k) or []
            out[k] = prev + [x for x in v if x not in prev]
        elif k == "question" and out.get(k) and v not in out[k]:
            out[k] = f"{out[k]} / {v}"
        else:
            out[k] = v
    return out


def compute_missing(task: Task) -> list[str]:
    if task["mode"] != "action":
        return []
    slots = task["slots"]
    missing = [s for s in CATALOG[task["intent"]]["required"] if slots.get(s) in (None, "", [])]
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
    if ex.intent == "attachment_analysis" and not slots.get("attachment_id"):
        att = (memory.get("last_attachment") or {}).get("attachment_id")
        if att:
            slots["attachment_id"] = att
    mode = "action" if CATALOG[ex.intent].get("force_action") else ex.mode
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
    """카탈로그의 produces/consumes로 의존성을 건다.

    소비 작업(예: 결제·보고서)은 같은 산출물을 만드는 열린 작업(예: 주소 변경·데이터 조회) 뒤로 미룬다.
    """
    producers: dict[str, list[str]] = {}
    latest_done: dict[str, Task] = {}
    for t in tasks:
        for kind in CATALOG.get(t["intent"], {}).get("produces", []):
            if t["status"] == "done":
                if kind not in latest_done or t.get("turn", 0) >= latest_done[kind].get("turn", 0):
                    latest_done[kind] = t
            elif t["status"] not in TERMINAL | {"failed"}:
                producers.setdefault(kind, []).append(t["id"])
    changed = []
    for t in tasks:
        consumes = CATALOG.get(t["intent"], {}).get("consumes", [])
        if not consumes or t["status"] in TERMINAL:
            continue
        need = {pid for kind in consumes for pid in producers.get(kind, []) if pid != t["id"]}
        # 열린 생산 작업이 없으면 가장 최근에 끝난 결과를 입력으로 쓴다 ('아까 조회한 걸 보고서로')
        if not t.get("depends_on"):
            need |= {latest_done[k]["id"] for k in consumes if k in latest_done and not producers.get(k)}
        deps = sorted(set(t.get("depends_on", [])) | need)
        if deps != t.get("depends_on", []):
            changed.append({**t, "depends_on": deps})
    return changed


def deps_done(task: Task, tasks: list[Task]) -> bool:
    by_id = {t["id"]: t for t in tasks}
    return all(by_id.get(d, {}).get("status") in TERMINAL for d in task.get("depends_on", []))


def plan(tasks: list[Task]) -> tuple[list[Task], list[Task]]:
    """(지금 실행할 작업, 상태가 바뀐 작업)을 돌려준다.

    - pending: 의존성이 풀렸으면 실행. 단 prepare 단계(order_product 임시 주문)는 의존성과 무관하게 실행: 돈이 안 나감
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
            # 실행 직전 재확인: 필수 슬롯이 비었는데 pending이면(병합·모드 전환 경로의 빈틈) 실행하지 않고 칩으로 묻는다
            need = compute_missing(t)
            if need:
                t = {**t, "status": "blocked", "missing": need}
                changed.append(t)
                continue
            if t.get("phase") != "prepare" and not deps_done(t, tasks):
                continue
            ready.append(t)
        elif st == "awaiting_confirm" and deps_done(t, tasks):
            ready.append(t)
    return ready, changed


PRIORITY = {"recover": 0, "policy": 1, "confirm": 2, "slot": 3, "convert": 4, "next": 5}


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
            elif first == "consent":
                out += [
                    {"label": "앞으로도 적용", "type": "confirm", "payload": {"op": "remember", "task_id": tid, "confirm": True}},
                    {"label": "이번만", "type": "confirm", "payload": {"op": "remember", "task_id": tid, "confirm": False}},
                ]
            elif ((t.get("result") or {}).get("choices") or {}).get(first):
                # worker가 고를 수 있는 값을 알려 준 경우 (예: 스키마 후보 DB): {"choices": {slot: [{"label", "value"}]}}
                out += [
                    {"label": c["label"], "type": kind, "payload": {"op": "fill_slots", "task_id": tid, "slots": {first: c["value"]}}}
                    for c in t["result"]["choices"][first][:3]
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
    blocking = [f for f in rule if f["type"] in ("recover", "policy", "confirm", "slot", "convert")]
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
    if op == "refine":
        return refine_task(tasks, t, payload.get("slots", {}), turn)
    if op == "remember":
        if not payload.get("confirm"):
            return [{**t, "status": "cancelled", "turn": turn}]
        nt = {**t, "slots": {**t["slots"], "consent": True}, "turn": turn}
        nt["missing"] = compute_missing(nt)
        if not nt["missing"]:
            nt["status"] = "pending"
        return [nt]
    return []


def refine_task(tasks: list[Task], t: Task, slots: dict, turn: int) -> list[Task]:
    """끝난 작업의 슬롯 일부만 바꿔 다시 실행한다 ('구별로도 나눠줘').

    그 산출물을 쓴 소비 작업(보고서 등)도 다시 돌도록 pending으로 되돌리고 의존성을 건다.
    """
    nt = {**t, "slots": merge_slots(t["slots"], slots), "mode": "action", "turn": turn, "error": None}
    nt["missing"] = compute_missing(nt)
    nt["status"] = "blocked" if nt["missing"] else "pending"
    out = [nt]
    produced = set(CATALOG.get(t["intent"], {}).get("produces", []))
    if produced:
        for d in tasks:
            if d["id"] == t["id"] or d["status"] != "done":
                continue
            if produced & set(CATALOG.get(d["intent"], {}).get("consumes", [])):
                deps = sorted(set(d.get("depends_on", [])) | {t["id"]})
                out.append({**d, "status": "pending", "depends_on": deps, "turn": turn})
    return out


# ---------- 가이드라인 ----------
# scope: "*" 항상 | "route:<route>" | "intent:<intent>" | 사용자 선호는 프로필에서 user.<key>로 생성
# priority: 작을수록 우선. 사용자 선호는 형식만 바꿀 수 있고 기관 정책을 넘지 못한다 (항상 맨 뒤).
GUIDELINES: dict[str, dict[str, Any]] = {
    "org.security": {"scope": "*", "priority": 0,
                     "text": "개인 식별 정보(주민등록번호·전화번호·계좌 등)는 조회·출력하지 않는다. 근거 없는 수치는 쓰지 않는다."},
    "sql.readonly": {"scope": "route:sql", "priority": 1,
                     "text": "단일 SELECT 문만 생성한다. 허용된 테이블만 쓰고 결과 행 수를 제한한다."},
    "report.formal": {"scope": "intent:report_compose", "priority": 2,
                      "text": "공문서 보고서 문체(개조식, '~함/~임')로 쓴다. 핵심 수치를 먼저, 해석은 뒤에."},
    "attachment.units": {"scope": "route:attachment", "priority": 2,
                         "text": "차트는 축 단위와 기간을 먼저 확인하고, 읽을 수 없는 값은 추정하지 않는다."},
}
USER_PREF_TEXT = {
    "decimal_places": "숫자는 반올림하지 않고 소수점 {v}자리까지 표기한다.",
    "table_format": "표는 {v} 형식으로 보여 준다.",
    "currency": "금액은 {v} 단위로 표기한다.",
}


def select_guidelines(task: Task, profile: dict | None = None) -> list[str]:
    """작업에 필요한 가이드라인 id만 고른다 (우선순위 순, 사용자 선호는 맨 뒤)."""
    route, intent = task.get("route"), task.get("intent")
    ids = [gid for gid, g in GUIDELINES.items()
           if g["scope"] == "*" or g["scope"] == f"route:{route}" or g["scope"] == f"intent:{intent}"]
    ids.sort(key=lambda gid: GUIDELINES[gid]["priority"])
    ids += [f"user.{k}" for k in sorted(profile or {})]
    return ids


def render_guidelines(ids: list[str], profile: dict | None = None) -> str:
    """가이드라인 id를 프롬프트 조각으로."""
    lines = []
    for gid in ids:
        if gid.startswith("user."):
            k = gid[5:]
            v = (profile or {}).get(k)
            if v is None:
                continue
            tpl = USER_PREF_TEXT.get(k, k + ": {v}")
            lines.append(f"- [사용자 선호] {tpl.format(v=v)}")
        elif gid in GUIDELINES:
            lines.append(f"- [{gid}] {GUIDELINES[gid]['text']}")
    return "\n".join(lines)


def attach_guidelines(tasks: list[Task], profile: dict | None = None) -> list[Task]:
    """실행 대상 작업에 가이드라인 id를 붙인다. 바뀐 작업만 돌려준다."""
    out = []
    for t in tasks:
        ids = select_guidelines(t, profile)
        if ids != t.get("guidelines"):
            out.append({**t, "guidelines": ids})
    return out
