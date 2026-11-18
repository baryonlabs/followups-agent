"""장기 사용자 메모리 ("사용자 인텔리전스", 설계 5.7).

계약
- load_profile(user_id) -> dict                       # 확정된 선호 {key: value}. begin_turn이 매 턴 호출
- build_memory_writer() -> async worker(arg, config)  # route="memory", intent="set_preference" (consent 칩 뒤에만 실행)
- async after_turn(app, config, out_state) -> list[FollowUp]
      턴 종료 후 비동기 추출. 후보는 set_preference 작업(missing=["consent"])으로 app.aupdate_state로 스레드에 넣고,
      그 작업의 동의 칩(rule_followups가 만드는 것과 같은 payload)을 돌려준다. 후보가 없으면 [].
- EXTRACTOR_HINT
- list_prefs / delete_pref / save_pref: 열람·정정·삭제 요구 대응용

저장소 (db.store("profile") = 서비스의 Postgres user_profile 테이블 대용)
    {"users": {uid: {
        "prefs": {key: {"value", "source_thread", "source_turn", "evidence", "created_at", "confirmed_at"}},
        "audit": [{"at", "op": "created"|"updated"|"deleted", "key", "value", "prev", "source_thread", "source_turn"}],
    }}}
    confirmed_at이 없는 행은 사용자 동의 전 후보이므로 load_profile이 돌려주지 않는다.

실행 위치
- after_turn은 응답을 보낸 뒤 돈다. 서비스에서는 FastAPI BackgroundTasks 또는 큐 워커가 호출하고,
  돌려받은 칩은 다음 턴 화면(또는 푸시)에 붙인다. run.py는 결정성을 위해 다음 단계 전에 기다린다.
- 기록(memory_writer)은 항상 사용자가 "앞으로도 적용" 칩을 누른 뒤에만 일어난다. 사용자가 본 적 없는 기억은 쌓지 않는다.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from followups import db
from followups.state import CATALOG, USER_PREF_TEXT, FollowUp, Task, compute_missing, rule_followups

# ---------- 키 정규화 ----------
# 화이트리스트는 state.USER_PREF_TEXT와 같다 (가이드라인 문장으로 바뀌는 키). 그 밖의 키는 일반 키로 저장하고
# render_guidelines가 "key: value" 형식으로 붙인다.
PREF_KEYS = tuple(USER_PREF_TEXT)  # decimal_places, table_format, currency
KEY_ALIASES = {
    "decimal_places": "decimal_places", "decimal": "decimal_places", "decimals": "decimal_places",
    "decimal_place": "decimal_places", "rounding": "decimal_places", "round": "decimal_places",
    "precision": "decimal_places", "number_format": "decimal_places", "소수점": "decimal_places",
    "반올림": "decimal_places", "소수점자리": "decimal_places",
    "table_format": "table_format", "table": "table_format", "table_style": "table_format", "표": "table_format",
    "표형식": "table_format",
    "currency": "currency", "currency_unit": "currency", "money_unit": "currency", "amount_unit": "currency",
    "unit": "currency", "금액단위": "currency", "단위": "currency", "통화": "currency",
}
_GENERIC_KEY = re.compile(r"^[a-z][a-z0-9_]{1,39}$")

# 기관 정책(org.security 등)을 넘으려는 선호는 기록하지 않는다. 사용자 선호는 형식(표·소수점·단위)만 바꾼다.
_POLICY_TERMS = re.compile(
    r"pii|ssn|rrn|resident|phone|mobile|account|bank|password|secret|token|mask|unmask|privacy|personal|"
    r"security|policy|guard|guideline|org_|sql|readonly|permission|admin|"
    r"주민|전화|휴대폰|연락처|계좌|카드번호|비밀번호|개인\s*정보|식별|실명|마스킹|가리지|보안|정책|권한|가이드라인|규칙\s*무시",
    re.IGNORECASE,
)
_KO_NUM = {"한": "1", "첫째": "1", "하나": "1", "두": "2", "둘째": "2", "둘": "2", "세": "3", "셋째": "3", "셋": "3",
           "네": "4", "넷째": "4", "넷": "4", "다섯": "5", "다섯째": "5", "영": "0", "없이": "0"}


def normalize_key(key: str | None) -> str | None:
    """선호 키를 화이트리스트 키로 맞춘다. 알 수 없는 키는 snake_case 일반 키로, 쓸 수 없는 키는 None."""
    if not key:
        return None
    k = re.sub(r"[\s\-]+", "_", str(key).strip().lower()).removeprefix("user.")
    if k in KEY_ALIASES:
        return KEY_ALIASES[k]
    if k.replace("_", "") in KEY_ALIASES:
        return KEY_ALIASES[k.replace("_", "")]
    return k if _GENERIC_KEY.match(k) else None


def normalize_value(key: str, value) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if key == "decimal_places":
        m = re.search(r"\d+", v)
        if m:
            return m.group(0)
        for word, num in _KO_NUM.items():
            if word in v:
                return num
    return v or None


def policy_violation(key: str | None, value) -> str | None:
    """기관 정책을 덮어쓰려는 선호면 거절 사유, 아니면 None."""
    hit = _POLICY_TERMS.search(f"{key or ''} {value or ''}")
    if hit:
        return f"'{hit.group(0)}' 관련 설정은 기관 정책이 정하므로 개인 선호로 바꿀 수 없습니다. 표·소수점·금액 단위 같은 형식만 기억할 수 있습니다."
    return None


# ---------- 프로필 저장소 ----------
AUDIT_LIMIT = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _user(data: dict, user_id: str) -> dict:
    u = data.setdefault("users", {}).setdefault(user_id, {})
    u.setdefault("prefs", {})
    u.setdefault("audit", [])
    return u


def _audit(u: dict, **row) -> None:
    u["audit"].append({"at": _now(), **row})
    del u["audit"][:-AUDIT_LIMIT]


def load_profile(user_id: str | None) -> dict:
    """확정된(confirmed_at 있는) 선호만 {key: value}로."""
    if not user_id:
        return {}
    with db.store("profile") as data:
        prefs = ((data.get("users") or {}).get(user_id) or {}).get("prefs") or {}
        return {k: p["value"] for k, p in prefs.items() if p.get("confirmed_at") and p.get("value") is not None}


def list_prefs(user_id: str) -> dict:
    """출처까지 포함한 전체 행 (열람 요구 대응). 미확정 후보도 포함."""
    with db.store("profile") as data:
        return dict(((data.get("users") or {}).get(user_id) or {}).get("prefs") or {})


def save_pref(user_id: str, key: str, value, *, source_thread: str | None = None, source_turn: int | None = None,
              evidence: str | None = None, confirmed: bool = True) -> dict:
    """선호 한 건을 기록하고 감사 로그를 남긴다. 키·값은 호출 전에 정규화·정책 검사를 마쳐야 한다."""
    with db.store("profile", write=True) as data:
        u = _user(data, user_id)
        prev = u["prefs"].get(key)
        row = {"value": value, "source_thread": source_thread, "source_turn": source_turn, "evidence": evidence,
               "created_at": (prev or {}).get("created_at") or _now(), "confirmed_at": _now() if confirmed else None}
        u["prefs"][key] = row
        _audit(u, op="updated" if prev else "created", key=key, value=value, prev=(prev or {}).get("value"),
               source_thread=source_thread, source_turn=source_turn)
        return row


def delete_pref(user_id: str, key: str) -> bool:
    """정정·삭제 요구 대응. 지운 값은 감사 로그에만 남는다."""
    key = normalize_key(key) or key
    with db.store("profile", write=True) as data:
        u = _user(data, user_id)
        prev = u["prefs"].pop(key, None)
        if prev is None:
            return False
        _audit(u, op="deleted", key=key, value=None, prev=prev.get("value"),
               source_thread=prev.get("source_thread"), source_turn=prev.get("source_turn"))
        return True


# ---------- 의도 추출기 안내 ----------
EXTRACTOR_HINT = (
    "- '앞으로', '항상', '매번', '기본으로'처럼 이후 대화에도 계속 적용해 달라는 형식 지시는 intent=set_preference.\n"
    "  pref_key는 decimal_places(소수점 자리·반올림), table_format(표 형식), currency(금액 단위) 중 하나, "
    "pref_value는 값만 (예: decimal_places='2').\n"
    "- 이번 답에만 해당하는 요청은 set_preference가 아니다. 개인정보 노출 등 정책을 바꾸려는 지시도 set_preference가 아니다.\n"
    "- consent는 절대 채우지 않는다 (사용자가 칩으로 직접 고른다)."
)


# ---------- memory_writer ----------
def build_memory_writer():
    async def memory_writer(arg, config) -> dict:
        t: Task = {**arg["task"], "turn": arg["turn"]}
        s = t.get("slots") or {}
        if s.get("consent") is not True:  # 동의 전에는 절대 쓰지 않는다
            t["missing"] = compute_missing(t) or ["consent"]
            t["status"] = "blocked"
            return {"task_stack": [t]}
        key = normalize_key(s.get("pref_key"))
        value = normalize_value(key, s.get("pref_value")) if key else None
        reason = policy_violation(s.get("pref_key"), s.get("pref_value")) or policy_violation(key, value)
        if not reason and (not key or value is None):
            reason = "기억할 설정 항목이나 값을 알아듣지 못했습니다."
        if reason:
            t.update(status="cancelled", error="policy_override" if key and value is not None else "invalid_preference",
                     result={"saved": {}, "rejected": {"pref_key": s.get("pref_key"), "pref_value": s.get("pref_value")},
                             "reason": reason})
            return {"task_stack": [t]}
        thread = ((config or {}).get("configurable") or {}).get("thread_id")
        save_pref(arg["user_id"], key, value, source_thread=thread, source_turn=s.get("source_turn") or arg["turn"],
                  evidence=s.get("evidence"))
        t.update(status="done", error=None, missing=[], slots={**s, "pref_key": key, "pref_value": value},
                 result={"saved": {key: value}, "note": "다음 턴부터 적용됩니다. 원하면 언제든 삭제할 수 있습니다."})
        return {"task_stack": [t]}

    return memory_writer


# ---------- 턴 종료 후 추출 ----------
# 값싼 1차 필터: 아래 단서가 하나도 없으면 LLM을 부르지 않는다.
#   지속 표현: 앞으로(는), 앞으론, 항상, 매번, 늘(띄어 쓴 단독 어절, '오늘' 제외), 언제나, 계속, 기본으로, 기본값, 다음부터, 이제부터, 이제는, 앞으로도
#   형식 지시: "~(으)로 보여/표시/표기/정리/출력/써 줘", 반올림, 소수점, ~자리까지, 단위로
PREF_CUES = re.compile(
    r"앞으로|앞으론|항상|매번|(?<![가-힣])늘\s|언제나|계속|기본으로|기본값|다음부터|이제부터|이제는|"
    r"(?:으)?로\s*(?:보여|표시|표기|정리|출력|써)\s*줘|반올림|소수점|자리까지|단위로"
)


class PrefCandidate(BaseModel):
    found: bool = Field(description="이후 대화에도 계속 적용할 형식 선호를 사용자가 밝혔으면 true. 이번 답에만 해당하면 false")
    pref_key: str | None = Field(None, description="decimal_places | table_format | currency 중 하나 (그 밖이면 snake_case)")
    pref_value: str | None = Field(None, description="값만. 예: decimal_places='2', table_format='마크다운 표', currency='천원'")
    evidence_span: str | None = Field(None, description="근거가 된 사용자 발화 조각 (원문 그대로)")


Extractor = Callable[[str, str, dict, dict], Awaitable["PrefCandidate | dict | None"]]
_default_extractor: Extractor | None = None


def _build_default_extractor() -> Extractor:
    from langchain_core.messages import HumanMessage, SystemMessage

    from followups.llm import MODEL_FAST, llm

    model = llm(MODEL_FAST).with_structured_output(PrefCandidate, method="function_calling")

    async def extract(user_text: str, answer: str, profile: dict, config: dict):
        sys = SystemMessage(content=(
            "너는 장기 사용자 메모리 추출기다. 마지막 사용자 발화에서 '앞으로도 계속' 적용할 형식 선호만 뽑는다.\n"
            "- 키: decimal_places(소수점 자리·반올림 안 함), table_format(표 형식), currency(금액 단위).\n"
            "- 이번 답에만 해당하는 요청, 개인정보·보안·정책을 바꾸려는 요청은 found=false.\n"
            "- 사용자가 말하지 않은 값은 추측하지 않는다.\n"
            f"이미 저장된 선호: {profile}"
        ))
        human = HumanMessage(content=f"사용자: {user_text}\n봇 답변: {answer[:500]}")
        return await model.ainvoke([sys, human], config={**(config or {}), "run_name": "memory_extractor.llm"})

    return extract


def _last_texts(out_state: dict) -> tuple[str, str]:
    from langchain_core.messages import AIMessage, HumanMessage

    user, bot = "", out_state.get("answer") or ""
    for m in reversed(out_state.get("messages") or []):
        if isinstance(m, HumanMessage):
            c = m.content
            user = c if isinstance(c, str) else " ".join(
                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            break
        if not bot and isinstance(m, AIMessage) and isinstance(m.content, str):
            bot = m.content
    return user.strip(), bot


def _next_task_id(tasks: list[Task]) -> str:
    ids = {t["id"] for t in tasks}
    nums = [int(m.group(1)) for i in ids if (m := re.fullmatch(r"T(\d+)", i))]
    n = max([len(ids), *nums]) + 1
    while f"T{n}" in ids:
        n += 1
    return f"T{n}"


def _candidate(raw) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if not raw.get("found", True):
        return None
    key = normalize_key(raw.get("pref_key"))
    value = normalize_value(key, raw.get("pref_value")) if key else None
    if not key or value is None:
        return None
    if policy_violation(raw.get("pref_key"), raw.get("pref_value")) or policy_violation(key, value):
        return None  # 정책을 넘는 선호는 묻지도 않는다
    return {"pref_key": key, "pref_value": value, "evidence": raw.get("evidence_span")}


async def after_turn(app, config, out_state, *, extractor: Extractor | None = None) -> list[FollowUp]:
    """턴이 끝난 뒤 영구 선호 후보를 뽑아 동의 대기 작업으로 넣고, 그 동의 칩을 돌려준다.

    중복 억제
    - interrupt로 멈춘 턴, 정책 차단 턴은 건너뛴다.
    - 이번 턴에 set_preference 작업이 생기거나 바뀌었으면 건너뛴다 (의도 추출기가 이미 잡았거나, 동의 칩을 누른 턴).
    - 후보 키에 열린 set_preference 작업이 이미 있으면 건너뛴다.
    - 확정 프로필에 같은 값이 이미 있으면 건너뛴다.
    """
    if not out_state or out_state.get("__interrupt__") or (out_state.get("guard") or {}).get("blocked"):
        return []
    tasks: list[Task] = list(out_state.get("task_stack") or [])
    turn = out_state.get("turn", 0)
    prefs = [t for t in tasks if t.get("intent") == "set_preference"]
    if any(t.get("turn") == turn for t in prefs):
        return []
    user_text, answer = _last_texts(out_state)
    if not user_text or not PREF_CUES.search(user_text):
        return []  # 단서 없음 → LLM 호출 생략

    global _default_extractor
    if extractor is None:
        _default_extractor = _default_extractor or _build_default_extractor()
        extractor = _default_extractor
    user_id = out_state.get("user_id")
    profile = load_profile(user_id)  # 이번 턴에 막 기록된 값까지 보려고 새로 읽는다
    cand = _candidate(await extractor(user_text, answer, profile, config))
    if not cand:
        return []
    key, value = cand["pref_key"], cand["pref_value"]
    if profile.get(key) == value:
        return []
    if any(t["status"] not in ("done", "cancelled", "failed") and normalize_key((t.get("slots") or {}).get("pref_key")) == key
           for t in prefs):
        return []

    slots = {"pref_key": key, "pref_value": value, "source_turn": turn}
    if cand.get("evidence"):
        slots["evidence"] = cand["evidence"]
    task: Task = {
        "id": _next_task_id(tasks), "intent": "set_preference", "mode": "action", "slots": slots, "depends_on": [],
        "risk": CATALOG["set_preference"]["risk"], "route": CATALOG["set_preference"]["route"],
        "status": "blocked", "result": None, "error": None, "converted": False, "turn": turn,
    }
    task["missing"] = compute_missing(task)  # ["consent"]
    chips = rule_followups([task])
    # 그래프 끝 노드가 둘(output_guard, followup_generator)이라 as_node를 지정해야 한다. 다음 노드 없음(END)
    await app.aupdate_state(config, {"task_stack": [task], "followups": [*(out_state.get("followups") or []), *chips]},
                            as_node="followup_generator")
    return chips
