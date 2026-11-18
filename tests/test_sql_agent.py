"""2단계: 멀티스키마 Text-to-SQL (스키마 라우팅 · 검증기 · 비용 확인 · worker 종단). LLM 없이 돈다."""
import sqlite3
import sys

import pytest

from followups import schema_index, sql_agent, sql_data
from followups.mcp_servers import datahub
from followups.schema_index import estimate_rows, over_budget, route_source, validate_sql
from followups.sql_agent import SQLDraft, build_sql_worker, period_quarters, resolve_conditions
from followups.state import ExtractedIntent, Slots, apply_payload, new_task, rule_followups


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLLOWUPS_DATA_DIR", str(tmp_path))
    return tmp_path


class FakeBox:
    """MCP 대신 datahub.execute를 직접 부른다 (같은 함수가 MCP 도구 본체)."""

    def __init__(self, fail: Exception | None = None):
        self.calls, self.fail = [], fail

    async def call(self, name, args, config=None):
        self.calls.append((name, args))
        if self.fail:
            raise self.fail
        assert name == "run_query"
        return datahub.execute(**args)


def fake_gen(*queries):
    """호출 순서대로 SQL을 돌려주는 가짜 생성기. 받은 요청은 .reqs에 남는다."""
    qs = list(queries)

    async def gen(req, config=None):
        gen.reqs.append(req)
        return SQLDraft(query=qs.pop(0) if len(qs) > 1 else qs[0])

    gen.reqs = []
    return gen


def _task(**slots):
    ex = ExtractedIntent(intent="data_query", mode="action", slots=Slots(**slots), span="")
    t = new_task("T2", ex, 1, {})
    return {**t, "status": "pending" if not t["missing"] else "blocked"}


def _arg(task, turn=1, utterance="", attachments=None, profile=None):
    return {"task": task, "user_id": "u1", "turn": turn, "utterance": utterance, "inputs": {},
            "attachments": attachments or {}, "profile": profile or {}, "guidelines_text": "- [sql.readonly] 단일 SELECT"}


MAPO_BY_MONTH = ("SELECT ym, SUM(sales_mkrw) AS sales FROM sales_monthly "
                 "WHERE gu = '마포구' AND ym BETWEEN '2026-01' AND '2026-06' GROUP BY ym ORDER BY ym")
STORES_BY_IND = ("SELECT industry AS 업종, SUM(store_count) AS 점포수 FROM stores_monthly "
                 "WHERE ym BETWEEN '2026-01' AND '2026-06' GROUP BY industry ORDER BY industry")
STORES_BY_IND_GU = ("SELECT gu AS 구, industry AS 업종, SUM(store_count) AS 점포수 FROM stores_monthly "
                    "WHERE ym BETWEEN '2026-01' AND '2026-06' GROUP BY gu, industry ORDER BY gu, industry")


# ---------- 목 DB ----------
def test_seed_mapo_sales_matches_chart(tmp_data):
    res = datahub.execute("sangkwon", MAPO_BY_MONTH)
    assert res["columns"] == ["ym", "sales"]
    totals = {ym: round(v) for ym, v in res["rows"]}
    assert totals == sql_data.MAPO_SALES_TOTAL
    assert totals["2026-04"] < min(v for k, v in totals.items() if k != "2026-04")
    assert (tmp_data / "sangkwon.sqlite").exists()


def test_build_is_deterministic_and_rebuilds_when_missing(tmp_data):
    p = sql_data.build("bizreg")
    first = p.read_bytes()
    assert sql_data.build("bizreg") == p and p.read_bytes() == first  # 같은 버전이면 그대로
    p.unlink()
    sql_data.build("bizreg")
    con = sqlite3.connect(p)
    n = con.execute("SELECT COUNT(*) FROM business_owners").fetchone()[0]
    con.close()
    assert n == sql_data.table_row_counts("bizreg")["business_owners"]


def test_readonly_connection_rejects_writes(tmp_data):
    p = sql_data.build("sangkwon")
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM sales_monthly")
    con.close()
    assert datahub.execute("sangkwon", "DELETE FROM sales_monthly")["error"] == "not_select"


def test_statement_timeout(tmp_data):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    assert datahub.execute("sangkwon", endless, timeout_ms=50)["error"] == "timeout"


def test_ddl_has_only_selected_source_and_marks_pii():
    d = sql_data.ddl("bizreg")
    assert "business_owners" in d and "stores_monthly" not in d
    assert "owner_rrn TEXT,  -- 대표자 주민등록번호 (개인정보) [조회 금지]" in d


# ---------- ① 스키마 라우팅 ----------
def test_store_count_is_ambiguous_between_sources():
    r = route_source("점포 수 업종")
    assert r["source"] is None and set(r["candidates"]) == {"sangkwon", "bizreg"}


@pytest.mark.parametrize("q,src", [("매출액", "sangkwon"), ("마포구 월별 매출", "sangkwon"),
                                   ("신규 개업 폐업 수", "bizreg"), ("분기별 점포 수", "bizreg"),
                                   ("월별 점포 수", "sangkwon")])
def test_clear_source_wins(q, src):
    assert route_source(q)["source"] == src


def test_no_match_offers_all_sources():
    r = route_source("유동인구")
    assert r["source"] is None and r["candidates"] == list(sql_data.SOURCES)


# ---------- ④ 검증기 ----------
def test_valid_select_gets_limit():
    v = validate_sql("select gu, sum(sales_mkrw) from sales_monthly group by gu", "sangkwon")
    assert v.ok and v.query.endswith("LIMIT 200") and v.tables == ["sales_monthly"]


@pytest.mark.parametrize("lim,expect", [(10, "LIMIT 10"), (5000, "LIMIT 200")])
def test_limit_kept_or_capped(lim, expect):
    v = validate_sql(f"SELECT gu FROM stores_monthly LIMIT {lim}", "sangkwon")
    assert v.ok and v.query.endswith(expect)


@pytest.mark.parametrize("sql,err", [
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("DROP TABLE stores_monthly", "not_select"),
    ("UPDATE stores_monthly SET store_count = 0", "not_select"),
    ("INSERT INTO stores_monthly VALUES ('x','y','z',1)", "not_select"),
    ("SELECT gu FROM business_registrations", "table_not_allowed"),  # 다른 출처의 테이블
    ("SELECT name FROM sqlite_master", "table_not_allowed"),
    ("SELECT * FROM stores_monthly", "select_star"),
    ("SELEC gu FRM", "parse_error"),
    ("", "parse_error"),
])
def test_rejects(sql, err):
    v = validate_sql(sql, "sangkwon")
    assert not v.ok and v.error == err, (v.error, v.detail)


@pytest.mark.parametrize("sql", [
    "SELECT owner_name, owner_rrn FROM business_owners",
    "SELECT gu, COUNT(*) FROM business_owners WHERE phone LIKE '010%' GROUP BY gu",
    "SELECT b.phone FROM business_owners b",
    "SELECT * FROM business_owners",
    "SELECT o.* FROM business_owners o",
    "WITH x AS (SELECT owner_rrn FROM business_owners) SELECT COUNT(*) FROM x",
])
def test_pii_blocked(sql):
    v = validate_sql(sql, "bizreg")
    assert not v.ok and v.error == "pii_column"


def test_aggregate_over_pii_table_allowed():
    v = validate_sql("SELECT gu, COUNT(*) AS 대표자수 FROM business_owners GROUP BY gu", "bizreg")
    assert v.ok


def test_cte_names_are_not_tables():
    v = validate_sql("WITH m AS (SELECT gu, store_count FROM stores_monthly) SELECT gu, SUM(store_count) FROM m GROUP BY gu",
                     "sangkwon")
    assert v.ok and v.tables == ["stores_monthly"]


# ---------- ⑤ 비용 ----------
def test_cost_estimate_uses_filters():
    full = validate_sql("SELECT industry, SUM(store_count) FROM stores_monthly GROUP BY industry", "sangkwon")
    narrow = validate_sql(STORES_BY_IND, "sangkwon")
    by_gu = validate_sql(MAPO_BY_MONTH, "sangkwon")
    assert full.est_rows == 1_800_000 and over_budget(full)
    assert narrow.est_rows == 450_000 and not over_budget(narrow)
    assert by_gu.est_rows < narrow.est_rows
    assert estimate_rows(narrow.tree, "sangkwon") == narrow.est_rows


# ---------- ② 조건 ----------
def test_conditions_from_attachment_then_profile_then_default():
    att = {"A1": {"kind": "chart", "turn": 1, "facts": {"region": "마포구", "period": "2026-01~2026-06", "unit": "백만원"}}}
    cond, origin = resolve_conditions({"metric": "점포 수", "group_by": ["업종"]}, att, {})
    assert cond["region"] == "마포구" and cond["period"] == "2026-01~2026-06" and cond["group_columns"] == ["industry"]
    assert origin == {"metric": "slot", "group_by": "slot", "region": "attachment", "period": "attachment"}

    cond, origin = resolve_conditions({"metric": "매출", "region": "강남"}, {}, {"default_period": "2026-05~2026-06"})
    assert cond["region"] == "강남구" and cond["period"] == "2026-05~2026-06" and origin["period"] == "profile"

    cond, origin = resolve_conditions({"metric": "매출"}, {"A1": {"facts": [{"key": "지역", "value": "종로구"}]}}, None)
    assert cond["region"] == "종로구" and origin["period"] == "default" and cond["period"] == sql_agent.DEFAULT_PERIOD


def test_period_quarters():
    assert period_quarters("2026-01~2026-06") == ["2026-Q1", "2026-Q2"]
    assert period_quarters("2025-11~2026-02") == ["2025-Q4", "2026-Q1"]
    assert period_quarters(None) == []


# ---------- worker 종단 ----------
async def test_ambiguous_source_chip_then_result():
    box = FakeBox()
    gen = fake_gen(STORES_BY_IND)
    worker = build_sql_worker(box, generate=gen)
    t = _task(metric="점포 수", group_by=["업종"], source="auto")
    assert t["status"] == "pending"  # 'auto'면 worker까지 와서 스키마 선택

    out = (await worker(_arg(t), None))["task_stack"][0]
    assert out["status"] == "blocked" and out["missing"] == ["source"] and "source" not in out["slots"]
    assert gen.reqs == [] and box.calls == []
    chips = rule_followups([out])
    assert [c["label"] for c in chips] == ["상권 DB(월별)", "사업자 DB(분기별)"]

    after = apply_payload([out], chips[0]["payload"], 2)[0]
    assert after["status"] == "pending" and after["slots"]["source"] == "sangkwon"
    res = await worker(_arg(after, turn=2), None)
    done = res["task_stack"][0]
    assert done["status"] == "done" and done["error"] is None
    assert done["result"]["columns"] == ["업종", "점포수"] and done["result"]["row_count"] == 3
    assert done["result"]["source"] == "sangkwon" and done["sql"]["tables"] == ["stores_monthly"]
    assert done["sql"]["query"].endswith("LIMIT 200")
    assert res["entity_memory"]["last_dataset"]["task_id"] == "T2"
    # ③ 생성기에는 고른 출처의 DDL과 가이드라인만 간다
    req = gen.reqs[0]
    assert "stores_monthly" in req["ddl"] and "business_owners" not in req["ddl"]
    assert "sql.readonly" in req["guidelines"] and req["conditions"]["period"] == sql_agent.DEFAULT_PERIOD


async def test_clear_source_filled_without_chip():
    worker = build_sql_worker(FakeBox(), generate=fake_gen(MAPO_BY_MONTH))
    t = _task(metric="매출액", region="마포구", source="auto")
    done = (await worker(_arg(t), None))["task_stack"][0]
    assert done["status"] == "done" and done["slots"]["source"] == "sangkwon"
    assert done["result"]["assumed"]["period"] == "default"


async def test_source_by_label_is_normalised():
    worker = build_sql_worker(FakeBox(), generate=fake_gen(STORES_BY_IND))
    done = (await worker(_arg(_task(metric="점포 수", source="상권 DB")), None))["task_stack"][0]
    assert done["status"] == "done" and done["slots"]["source"] == "sangkwon"


async def test_refine_skips_schema_selection_and_regenerates(monkeypatch):
    box = FakeBox()
    gen = fake_gen(STORES_BY_IND, STORES_BY_IND_GU)
    worker = build_sql_worker(box, generate=gen)
    t = _task(metric="점포 수", group_by=["업종"], source="sangkwon")
    done = (await worker(_arg(t), None))["task_stack"][0]
    assert done["status"] == "done"

    def no_routing(*a, **k):
        raise AssertionError("refine에서 스키마 선택을 다시 하면 안 된다")

    monkeypatch.setattr(sql_agent, "route_source", no_routing)
    refined = apply_payload([done], {"op": "refine", "task_id": "T2", "slots": {"group_by": ["구"]}}, 3)[0]
    assert refined["status"] == "pending" and refined["slots"]["group_by"] == ["업종", "구"]
    again = (await worker(_arg(refined, turn=3, utterance="구별로도 나눠줘"), None))["task_stack"][0]
    assert again["status"] == "done" and again["turn"] == 3
    assert again["result"]["columns"] == ["구", "업종", "점포수"] and again["result"]["row_count"] == 9
    assert again["slots"]["source"] == "sangkwon" and len(gen.reqs) == 2
    assert gen.reqs[1]["conditions"]["group_columns"] == ["industry", "gu"]


async def test_pii_query_fails_without_execution():
    box = FakeBox()
    worker = build_sql_worker(box, generate=fake_gen("SELECT owner_name, owner_rrn FROM business_owners"))
    t = _task(metric="대표자 명단", source="bizreg")
    out = (await worker(_arg(t, utterance="대표자 주민번호랑 같이 명단 뽑아줘"), None))["task_stack"][0]
    assert out["status"] == "failed" and out["error"] == "pii_column"
    assert box.calls == []  # 실행하지 않음
    assert out["sql"]["query"] is None and "SELECT" not in str(out).upper()  # 거부된 SQL 원문을 남기지 않음
    assert [c["label"] for c in rule_followups([out])] == ["다시 시도"]


async def test_format_error_regenerates_once_with_feedback():
    gen = fake_gen("SELECT gu FROM business_registrations", STORES_BY_IND)
    worker = build_sql_worker(FakeBox(), generate=gen)
    done = (await worker(_arg(_task(metric="점포 수", source="sangkwon")), None))["task_stack"][0]
    assert done["status"] == "done" and len(gen.reqs) == 2
    assert gen.reqs[1]["feedback"].startswith("table_not_allowed")


async def test_cost_confirm_run():
    box = FakeBox()
    worker = build_sql_worker(box, generate=fake_gen("SELECT industry, SUM(sales_mkrw) AS s FROM sales_monthly GROUP BY industry"))
    t = _task(metric="매출액", group_by=["업종"], source="sangkwon")
    out = (await worker(_arg(t), None))["task_stack"][0]
    assert out["status"] == "blocked" and out["missing"] == ["cost_ok"] and box.calls == []
    assert "1,800,000행" in out["result"]["message"]
    chips = rule_followups([out])
    assert [c["label"] for c in chips] == ["실행", "조건 줄이기"]

    # fill_slots: cost_ok는 required가 아니므로 missing이 비고 pending으로 → worker 재실행
    after = apply_payload([out], chips[0]["payload"], 2)[0]
    assert after["status"] == "pending" and after["missing"] == [] and after["slots"]["cost_ok"] == "run"
    done = (await worker(_arg(after, turn=2), None))["task_stack"][0]
    assert done["status"] == "done" and "cost_ok" not in done["slots"] and len(box.calls) == 1


async def test_cost_confirm_reduce_asks_period():
    gen = fake_gen("SELECT industry, SUM(sales_mkrw) AS s FROM sales_monthly GROUP BY industry",
                   "SELECT industry, SUM(sales_mkrw) AS s FROM sales_monthly WHERE ym BETWEEN '2026-04' AND '2026-06' GROUP BY industry")
    worker = build_sql_worker(FakeBox(), generate=gen)
    out = (await worker(_arg(_task(metric="매출액", source="sangkwon")), None))["task_stack"][0]
    reduce = next(c for c in rule_followups([out]) if c["label"] == "조건 줄이기")
    t2 = apply_payload([out], reduce["payload"], 2)[0]
    asked = (await worker(_arg(t2, turn=2), None))["task_stack"][0]
    assert asked["status"] == "blocked" and asked["missing"] == ["period"] and "cost_ok" not in asked["slots"]
    chips = rule_followups([asked])
    assert chips[0]["label"] == "최근 3개월"
    t3 = apply_payload([asked], chips[0]["payload"], 3)[0]
    assert t3["status"] == "pending"
    done = (await worker(_arg(t3, turn=3), None))["task_stack"][0]
    assert done["status"] == "done" and done["result"]["conditions"]["period"] == "2026-04~2026-06"


async def test_execution_failure_marks_failed():
    worker = build_sql_worker(FakeBox(fail=RuntimeError("서버 끊김")), generate=fake_gen(STORES_BY_IND))
    out = (await worker(_arg(_task(metric="점포 수", source="sangkwon")), None))["task_stack"][0]
    assert out["status"] == "failed" and out["error"] == "execution_error"
    assert rule_followups([out])[0]["payload"]["op"] == "retry"


async def test_generation_error_marks_failed():
    async def boom(req, config=None):
        raise RuntimeError("401")

    worker = build_sql_worker(FakeBox(), generate=boom)
    out = (await worker(_arg(_task(metric="점포 수", source="sangkwon")), None))["task_stack"][0]
    assert out["status"] == "failed" and out["error"] == "sql_generation_error"


async def test_module_generator_is_monkeypatchable(monkeypatch):
    monkeypatch.setattr(sql_agent, "generate_sql", fake_gen(STORES_BY_IND))
    worker = build_sql_worker(FakeBox())
    out = (await worker(_arg(_task(metric="점포 수", source="sangkwon")), None))["task_stack"][0]
    assert out["status"] == "done"


async def test_real_mcp_session(tmp_data):
    """datahub를 실제 stdio MCP 서버로 띄워 도구를 부른다 (Toolbox 경로)."""
    import os

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from followups.tools import ROOT, Toolbox

    cfg = {"datahub": {"command": sys.executable, "args": ["-m", "followups.mcp_servers.datahub"], "transport": "stdio",
                       "cwd": str(ROOT), "env": {**os.environ, "FASTMCP_LOG_LEVEL": "WARNING"}}}
    client = MultiServerMCPClient(cfg)
    async with client.session("datahub") as session:
        box = Toolbox(by_server={"datahub": await load_mcp_tools(session)})
        worker = build_sql_worker(box, generate=fake_gen(MAPO_BY_MONTH))
        done = (await worker(_arg(_task(metric="매출액", region="마포구", source="sangkwon")), None))["task_stack"][0]
        pii = await box.call("run_query", {"source": "bizreg", "sql": "SELECT owner_rrn FROM business_owners"})
    assert done["status"] == "done" and done["result"]["row_count"] == 6
    assert pii["error"] == "pii_column"


def test_extractor_hint_mentions_refine():
    assert "data_query" in sql_agent.EXTRACTOR_HINT and "existing_task_id" in sql_agent.EXTRACTOR_HINT
    assert schema_index.MAX_LIMIT == 200
