"""멀티스키마 Text-to-SQL worker (route="sql", intent="data_query").

계약
- build_sql_worker(box) -> async worker(arg: WorkerArg, config) -> {"task_stack": [task], "entity_memory"?: {...}}
- EXTRACTOR_HINT: 의도 추출기 시스템 프롬프트에 덧붙일 data_query 안내

파이프라인 (설계 5.4: 단계마다 슬롯을 채우고, 막히면 칩으로 묻는다)
  ① 스키마 선택   source가 비었으면 스키마 인덱스로 고름. 비슷하면 blocked missing=[source] + 출처 선택지
  ② 조건 확정     metric/period/region/group_by: 슬롯 → 첨부 사실 → 프로필 → 기본값(기간: 데이터 전체 6개월)
  ③ SQL 생성     선택된 출처의 DDL + 가이드라인만 넣어 구조화 출력 (generate_sql, 테스트에서 교체 가능)
  ④ 검증         schema_index.validate_sql. PII → failed "pii_column" (실행 안 함). 형식 오류는 1회 재생성
  ⑤ 비용 확인     추정 행 수 초과 → blocked missing=[cost_ok] + "실행 / 조건 줄이기" 선택지
  ⑥ 실행         datahub MCP 서버(read-only) → result/sql 기록. 실패 → failed (규칙이 "다시 시도" 칩)
  refine          source가 이미 있으므로 ①을 건너뛰고 ②부터 다시 (같은 출처 보장)

비용 확인 슬롯(cost_ok) 설계
- cost_ok는 CATALOG required가 아니다. worker가 직접 missing=["cost_ok"]로 막고 result.choices.cost_ok를 준다.
- 칩(fill_slots) → compute_missing이 required만 다시 계산해 []가 되므로 blocked → pending → worker 재실행.
  이때 slots.cost_ok로 사용자의 답을 읽는다: "run"이면 비용 확인을 건너뛰고 실행, "reduce"면 조건(기간→지역)을 칩으로 묻는다.
- cost_ok는 실행(또는 reduce 처리) 때 슬롯에서 지운다 → 다음 refine에서 비용을 다시 확인한다 (확인이 다른 SQL로 새지 않음).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from followups.schema_index import COST_LIMIT_ROWS, MAX_LIMIT, over_budget, route_source, validate_sql
from followups.sql_data import GUS, MONTHS, SOURCES, ddl

EXTRACTOR_HINT = (
    "- 통계·데이터 조회(점포 수, 매출, 개업·폐업 수 등을 'DB에서 뽑아줘', '보여줘')는 intent=data_query.\n"
    "  metric=지표 이름('점포 수', '매출액', '폐업 수'), region=지역('마포구'), period='YYYY-MM~YYYY-MM'(말한 경우만),\n"
    "  group_by=구분 기준 목록('업종별'→['업종'], '구별'→['구'], '월별'→['월']).\n"
    "  source는 사용자가 DB를 직접 고른 경우만 'sangkwon'(상권 DB) 또는 'bizreg'(사업자 DB), 말하지 않았으면 'auto'로 둔다 (시스템이 고르거나 묻는다).\n"
    "- 앞서 조회한 데이터를 다시 나누거나 조건만 바꾸는 후속 발화('구별로도 나눠줘', '5월만 다시')는 새 조회가 아니다:\n"
    "  최근 엔티티 last_dataset.task_id(또는 열린 data_query 작업 id)를 existing_task_id로, 바뀐 슬롯만 채운다 (예: group_by=['구'])."
)

MAX_ROWS_IN_STATE = 50      # 상태에 남기는 행 수 (나머지는 row_count로만)
AUTO = {"", "auto", "unknown", "모름"}
COST_CHOICES = [{"label": "실행", "value": "run"}, {"label": "조건 줄이기", "value": "reduce"}]
PERIOD_CHOICES = [
    {"label": "최근 3개월", "value": f"{MONTHS[-3]}~{MONTHS[-1]}"},
    {"label": "최근 1개월", "value": f"{MONTHS[-1]}~{MONTHS[-1]}"},
]
DEFAULT_PERIOD = f"{MONTHS[0]}~{MONTHS[-1]}"  # 기간을 말하지 않으면 데이터가 있는 최근 6개월
GROUP_COLUMNS = {"업종": "industry", "구": "gu", "자치구": "gu", "지역": "gu", "월": "ym", "월별": "ym", "분기": "quarter",
                 "연월": "ym"}


# ---------- ③ SQL 생성 (교체 가능) ----------
class SQLDraft(BaseModel):
    query: str = Field(description="SQLite SELECT 문 하나. 세미콜론·주석 없이")
    explanation: str = Field("", description="무엇을 집계하는지 한 줄")


SqlGenerator = Callable[[dict, Any], Awaitable["SQLDraft | str"]]
_sql_llm = None


async def llm_generate_sql(req: dict, config=None) -> SQLDraft:
    """선택된 출처의 DDL만 넣고 LLM으로 SQL을 만든다 (코드 특화 모델: MODEL_SQL)."""
    global _sql_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    from followups.llm import MODEL_MAIN, llm

    if _sql_llm is None:
        _sql_llm = llm(os.getenv("MODEL_SQL", MODEL_MAIN)).with_structured_output(SQLDraft, method="function_calling")
    sys = SystemMessage(content=(
        "너는 SQLite SQL 작성기다. 아래 DDL의 테이블·컬럼만 써서 조건에 맞는 SELECT 문 하나를 만든다.\n"
        "- 집계는 GROUP BY로, 결과 컬럼에는 알아보기 쉬운 별칭을 붙인다. SELECT * 금지, [조회 금지] 컬럼 금지.\n"
        "- 기간 조건은 ym('YYYY-MM') 또는 quarter('YYYY-Qn') 컬럼으로 건다. LIMIT은 200 이하.\n"
        f"지켜야 할 지침:\n{req.get('guidelines') or '-'}\n"
        f"DDL ({req['source']}):\n{req['ddl']}\n"
        f"조건: {json.dumps(req['conditions'], ensure_ascii=False)}\n"
        + (f"직전 SQL이 거부됨: {req['feedback']}. 고쳐서 다시 작성.\n" if req.get("feedback") else "")
    ))
    cfg = {**(config or {}), "run_name": "sql_agent.generate.llm"}
    return await _sql_llm.ainvoke([sys, HumanMessage(content=req.get("utterance") or req["conditions"].get("metric", ""))],
                                  config=cfg)


generate_sql: SqlGenerator = llm_generate_sql  # 테스트에서 monkeypatch


# ---------- ② 조건 ----------
def _norm_region(r: str | None) -> str | None:
    if not r:
        return None
    r = str(r).strip()
    for g in GUS:
        if r == g or r == g[:-1] or g in r:
            return g
    return r


def _fact_values(facts: Any) -> dict:
    """첨부 사실에서 region/period를 찾는다. dict, [{"key","value"}], [dict] 형식을 모두 받는다."""
    keys = {"region": ("region", "지역", "gu"), "period": ("period", "기간")}
    out: dict = {}
    items: list[dict] = []
    if isinstance(facts, dict):
        items = [facts]
    elif isinstance(facts, list):
        items = [f for f in facts if isinstance(f, dict)]
    for it in items:
        if "key" in it and "value" in it:
            it = {it["key"]: it["value"]}
        for slot, names in keys.items():
            for n in names:
                if it.get(n) and slot not in out:
                    out[slot] = it[n]
    return out


def resolve_conditions(slots: dict, attachments: dict | None, profile: dict | None) -> tuple[dict, dict]:
    """(조건, 출처표시). 슬롯 → 첨부 사실(최근 것 우선) → 프로필 → 기본값 순."""
    cond = {"metric": slots.get("metric"), "period": slots.get("period"), "region": slots.get("region"),
            "group_by": list(slots.get("group_by") or [])}
    origin = {k: "slot" for k, v in cond.items() if v}
    atts = sorted((attachments or {}).values(), key=lambda a: -(a.get("turn") or 0) if isinstance(a, dict) else 0)
    for a in atts:
        if not isinstance(a, dict):
            continue
        for k, v in _fact_values(a.get("facts")).items():
            if not cond.get(k):
                cond[k], origin[k] = v, "attachment"
    prof = profile or {}
    for k, pk in (("region", "region"), ("region", "default_region"), ("period", "default_period")):
        if not cond.get(k) and prof.get(pk):
            cond[k], origin[k] = prof[pk], "profile"
    if not cond.get("period"):
        cond["period"], origin["period"] = DEFAULT_PERIOD, "default"
    cond["region"] = _norm_region(cond.get("region"))
    cond["group_columns"] = [GROUP_COLUMNS.get(g, g) for g in cond["group_by"]]
    return cond, origin


def period_quarters(period: str | None) -> list[str]:
    """'2026-01~2026-06' → ['2026-Q1', '2026-Q2'] (분기 DB용)."""
    m = re.findall(r"(\d{4})-(\d{2})", period or "")
    if not m:
        return []
    (y1, m1), (y2, m2) = m[0], m[-1]
    out, y, q = [], int(y1), (int(m1) - 1) // 3 + 1
    end = (int(y2), (int(m2) - 1) // 3 + 1)
    while (y, q) <= end and len(out) < 20:
        out.append(f"{y}-Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def _route_query(slots: dict, utterance: str) -> str:
    """스키마 검색어: 지표 + 구분 기준. 발화 전체는 넣지 않는다 ('3분기 보고서' 같은 다른 요청이 출처를 흔든다)."""
    parts = [slots.get("metric") or "", *(slots.get("group_by") or [])]
    q = " ".join(p for p in parts if p).strip()
    return q or utterance or ""


def _match_source(v: Any) -> str | None:
    """source 슬롯 값 → 출처 id. 'auto'/빈 값은 None, 이름('상권 DB')으로 말한 경우도 보정."""
    if not v or str(v).strip().lower() in AUTO:
        return None
    v = str(v).strip()
    if v in SOURCES:
        return v
    return next((s for s, m in SOURCES.items() if v in m["label"] or m["label"].split("(")[0] in v), None)


# ---------- worker ----------
def _blocked(t: dict, slot: str, choices: list[dict], **extra) -> dict:
    return {**t, "status": "blocked", "missing": [slot], "error": None,
            "result": {"choices": {slot: choices}, **extra}}


def build_sql_worker(box, generate: SqlGenerator | None = None):
    async def sql_worker(arg, config=None) -> dict:
        gen = generate or generate_sql
        t = {**arg["task"], "turn": arg["turn"]}
        slots = dict(t.get("slots") or {})

        # ① 스키마 선택 (refine이면 source가 이미 있으므로 건너뜀)
        source = _match_source(slots.get("source"))
        if not source:
            r = route_source(_route_query(slots, arg.get("utterance", "")))
            if r["source"] is None:
                slots.pop("source", None)
                choices = [{"label": SOURCES[s]["label"], "value": s} for s in r["candidates"]]
                return {"task_stack": [_blocked({**t, "slots": slots}, "source", choices,
                                                scores=r["scores"], tables=r["tables"][:4])]}
            source = r["source"]
        slots["source"] = source
        if not slots.get("metric"):
            return {"task_stack": [{**t, "slots": slots, "status": "blocked", "missing": ["metric"], "result": None}]}

        # ② 조건 확정
        cond, origin = resolve_conditions(slots, arg.get("attachments"), arg.get("profile"))
        if slots.get("cost_ok") == "reduce":
            slots.pop("cost_ok")
            if origin.get("period") in ("default", None):
                return {"task_stack": [_blocked({**t, "slots": slots}, "period", PERIOD_CHOICES)]}
            if not cond.get("region"):
                return {"task_stack": [_blocked({**t, "slots": slots}, "region",
                                                [{"label": g, "value": g} for g in GUS])]}
            # 이미 기간·지역이 좁혀져 있으면 그대로 실행 (더 줄일 조건이 없음)
            slots["cost_ok"] = "run"
        if SOURCES[source]["grain"] == "분기":
            cond["quarters"] = period_quarters(cond["period"])

        # ③ SQL 생성 → ④ 검증 (형식 오류는 한 번 다시 생성, PII는 재시도 없이 차단)
        req = {"source": source, "ddl": ddl(source), "guidelines": arg.get("guidelines_text", ""),
               "conditions": cond, "utterance": arg.get("utterance", ""), "feedback": None}
        v = None
        for _ in range(2):
            try:
                draft = await gen(req, config)
            except Exception as e:  # LLM 오류 → 실패 (다시 시도 칩)
                return {"task_stack": [{**t, "slots": slots, "status": "failed", "missing": [],
                                        "error": "sql_generation_error", "result": {"detail": str(e)[:200]}}]}
            query = draft.query if isinstance(draft, SQLDraft) else str(draft)
            v = validate_sql(query, source)
            if v.ok or v.error == "pii_column":
                break
            req = {**req, "feedback": f"{v.error}: {v.detail}"}
        if not v.ok:
            # 거부된 SQL 원문은 남기지 않는다 (PII 컬럼을 읽는 SQL이 상태·트레이스에 남지 않게)
            return {"task_stack": [{**t, "slots": slots, "status": "failed", "missing": [], "error": v.error,
                                    "result": {"detail": v.detail, "source": source},
                                    "sql": {"tables": v.tables, "query": None, "rejected": v.error}}]}

        # ⑤ 비용 확인
        if over_budget(v) and slots.get("cost_ok") != "run":
            msg = f"약 {v.est_rows:,}행을 읽습니다 (기준 {COST_LIMIT_ROWS:,}행). 실행할까요?"
            blocked = _blocked({**t, "slots": slots}, "cost_ok", COST_CHOICES, est_rows=v.est_rows, message=msg,
                               source=source)
            return {"task_stack": [{**blocked, "sql": {"tables": v.tables, "query": v.query}}]}

        # ⑥ 실행 (MCP datahub 서버)
        try:
            res = await box.call("run_query", {"source": source, "sql": v.query, "max_rows": MAX_LIMIT}, config)
        except Exception as e:
            res = {"error": "execution_error", "detail": str(e)[:200]}
        slots.pop("cost_ok", None)  # 확인은 이번 실행에만 쓴다
        sql = {"tables": v.tables, "query": v.query}
        if res.get("error"):
            return {"task_stack": [{**t, "slots": slots, "status": "failed", "missing": [], "error": res["error"],
                                    "result": {"detail": res.get("detail", ""), "source": source}, "sql": sql}]}
        rows = res.get("rows") or []
        result = {
            "columns": res.get("columns", []), "rows": rows[:MAX_ROWS_IN_STATE], "row_count": len(rows),
            "truncated": bool(res.get("truncated")) or len(rows) > MAX_ROWS_IN_STATE,
            "source": source, "source_label": SOURCES[source]["label"],
            "conditions": {k: cond.get(k) for k in ("metric", "period", "region", "group_by")},
            "assumed": {k: o for k, o in origin.items() if o != "slot"}, "est_rows": v.est_rows,
        }
        done = {**t, "slots": slots, "status": "done", "missing": [], "error": None, "result": result, "sql": sql}
        memo = {"last_dataset": {"task_id": t["id"], "source": source, "metric": cond.get("metric"),
                                 "group_by": cond.get("group_by"), "columns": result["columns"]}}
        return {"task_stack": [done], "entity_memory": memo}

    return sql_worker
