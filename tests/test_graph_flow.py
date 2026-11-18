"""통합: 실제 그래프 + 실제 MCP 서버(datahub 등)로 6장 시나리오(첨부 → 출처 칩 → 보고서 → 구별 refine)를 돌린다.

LLM만 가짜로 바꾼다 (의도 추출·답변·SQL 생성·VLM·보고서). 상태 전이·의존성·칩·가드·SQL 검증/실행은 전부 진짜다.
"""
from __future__ import annotations

import base64
from contextlib import AsyncExitStack, asynccontextmanager

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from followups import attachments, db, graph, sql_agent
from followups.state import ExtractedIntent, IntentResult, Slots

CHART = db.ROOT / "data" / "attachments" / "mapo_sales_2026h1.png"
FACTS = {
    "kind": "chart", "title": "마포구 월별 매출 (2026년 1~6월, 단위: 백만원)", "region": "마포구",
    "period": "2026-01~2026-06", "y_unit": "백만원",
    "series": [{"label": "매출", "values": [{"x": f"2026-0{i + 1}", "y": v} for i, v in
                                            enumerate([1250, 1310, 1290, 870, 1180, 1240])]}],
    "anomalies": [{"x": "2026-04", "note": "전월 대비 급감"}],
}


class FakeModel:
    """graph.build_graph가 만드는 LLM 자리. 의도 추출은 SCRIPT에서 차례로 꺼낸다."""

    script: list[IntentResult] = []

    def __init__(self, schema=None):
        self.schema = schema

    def with_structured_output(self, schema, **kw):
        return FakeModel(schema)

    async def ainvoke(self, messages, config=None):
        if self.schema is IntentResult:
            return FakeModel.script.pop(0)
        if self.schema is not None:
            return self.schema(labels=[])
        return AIMessage(content="요약 답변")


def _ex(intent, **slots):
    existing = slots.pop("existing_task_id", None)
    return ExtractedIntent(intent=intent, mode="action", slots=Slots(**slots), span="", existing_task_id=existing)


async def fake_sql(req, config=None):
    cols = req["conditions"]["group_columns"] or ["industry"]
    sel = ", ".join(cols)
    if req["source"] == "sangkwon":
        q = (f"SELECT {sel}, SUM(store_count) AS 점포수 FROM stores_monthly WHERE gu = '마포구' "
             f"AND ym BETWEEN '2026-01' AND '2026-06' GROUP BY {sel}")
    else:
        q = f"SELECT {sel}, SUM(active_count) AS 점포수 FROM active_businesses WHERE gu = '마포구' GROUP BY {sel}"
    return sql_agent.SQLDraft(query=q, explanation="")


REPORTS: list[list] = []


async def fake_report(messages, config):
    REPORTS.append(messages)
    return f"□ 마포구 4월 매출 870백만원으로 급감함 (보고서 {len(REPORTS)}판)"


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLLOWUPS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(attachments, "OBJECT_DIR", tmp_path / "uploads")
    monkeypatch.setattr(graph, "llm", lambda *a, **k: FakeModel())
    monkeypatch.setattr(graph, "build_supervisor", lambda box: None)
    monkeypatch.setattr(graph, "build_refund_fastpath", lambda box: None)
    monkeypatch.setattr(sql_agent, "generate_sql", fake_sql)
    monkeypatch.setattr(attachments, "write_report", fake_report)

    async def fake_read(url, text, config):
        return dict(FACTS)

    monkeypatch.setattr(attachments, "read_image_facts", fake_read)
    db.reset()
    REPORTS.clear()


@asynccontextmanager
async def make_app():
    """MCP 세션은 같은 task 안에서 열고 닫아야 한다 (anyio cancel scope) → fixture가 아니라 테스트 본문에서."""
    from followups.tools import open_toolbox

    async with AsyncExitStack() as stack:
        box = await open_toolbox(stack)
        yield graph.build_graph(box)


def _by_intent(state, intent):
    return [t for t in state["task_stack"] if t["intent"] == intent]


async def test_chart_db_report_refine_flow(fakes):
    async with make_app() as app:
        await _flow(app)


async def _flow(app):
    url = "data:image/png;base64," + base64.b64encode(CHART.read_bytes()).decode()
    att_id = attachments.attachment_id_of(url)
    cfg = {"configurable": {"thread_id": "flow"}}

    # 턴 1: 차트 + 복합 요청 → 첨부 해석 done, 데이터 조회는 출처 칩으로 막힘, 보고서는 대기
    FakeModel.script = [IntentResult(intents=[
        _ex("attachment_analysis", attachment_id=att_id),
        _ex("data_query", metric="점포 수", group_by=["업종"], region="마포구", source="auto"),
        _ex("report_compose", tone="3분기 보고서", attachment_id=att_id),
    ])]
    msg = HumanMessage(content=[{"type": "text", "text": "4월에 왜 떨어졌는지, 업종별 점포 수 뽑아서 보고서로"},
                                {"type": "image_url", "image_url": {"url": url}}])
    out = await app.ainvoke({"messages": [msg], "user_id": "u1", "payload": None}, cfg)
    assert "base64" not in str(out["attachments"])  # 원본은 상태에 없다
    assert _by_intent(out, "attachment_analysis")[0]["status"] == "done"
    q = _by_intent(out, "data_query")[0]
    rep = _by_intent(out, "report_compose")[0]
    assert q["status"] == "blocked" and q["missing"] == ["source"]
    assert rep["status"] == "pending" and q["id"] in rep["depends_on"]
    labels = [c["label"] for c in out["followups"]]
    assert "상권 DB(월별)" in labels
    assert "sql.readonly" in q["guidelines"] and "report.formal" not in q["guidelines"]  # 작업별 가이드라인

    # 턴 2: 출처 칩 → 의도 추출 없이 SQL 실행 → 의존성 풀린 보고서가 같은 턴에 작성
    chip = next(c for c in out["followups"] if c["label"] == "상권 DB(월별)")
    out = await app.ainvoke({"messages": [HumanMessage(content="[칩] 상권 DB(월별)")], "payload": chip["payload"]}, cfg)
    q = _by_intent(out, "data_query")[0]
    rep = _by_intent(out, "report_compose")[0]
    assert q["status"] == "done" and q["slots"]["source"] == "sangkwon" and q["result"]["row_count"] > 0
    assert rep["status"] == "done" and out["answer"] == rep["result"]["summary"]
    prompt = str(REPORTS[-1])
    assert "report.formal" in prompt and "sql.readonly" not in prompt  # 보고서에는 보고서 지침만
    assert FakeModel.script == []  # 칩 턴은 LLM 의도 추출을 건너뛴다

    # 턴 3: '구별로도 나눠줘' → 끝난 조회를 refine (출처 재선택 없음) → 보고서도 다시
    FakeModel.script = [IntentResult(intents=[_ex("data_query", group_by=["구"], existing_task_id=q["id"])])]
    out = await app.ainvoke({"messages": [HumanMessage(content="구별로도 나눠줘")], "payload": None}, cfg)
    q2 = _by_intent(out, "data_query")[0]
    rep2 = _by_intent(out, "report_compose")[0]
    assert len(_by_intent(out, "data_query")) == 1  # 새 작업이 아니라 같은 작업
    assert q2["slots"]["group_by"] == ["업종", "구"] and q2["slots"]["source"] == "sangkwon"
    assert "gu" in q2["sql"]["query"] and q2["status"] == "done"
    assert rep2["status"] == "done" and len(REPORTS) == 2


async def test_policy_block_never_calls_llm(fakes):
    FakeModel.script = []  # 추출기가 불리면 IndexError
    cfg = {"configurable": {"thread_id": "pii"}}
    async with make_app() as app:
        out = await app.ainvoke({"messages": [HumanMessage(content="대표자 주민번호랑 같이 명단 뽑아줘")],
                                 "user_id": "u1", "payload": None}, cfg)
    assert out["guard"]["blocked"] and not out.get("task_stack")
    assert [f["type"] for f in out["followups"]] == ["policy", "policy"]
