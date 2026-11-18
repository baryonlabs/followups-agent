"""4단계 멀티모달: 첨부 판독·해석·보고서 합성 (가짜 VLM/작성기로 LLM 없이)."""
import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from followups import attachments as A
from followups.state import ExtractedIntent, Slots, apply_payload, new_task, render_guidelines, rule_followups, select_guidelines

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-chart-bytes"
URI = "data:image/png;base64," + base64.b64encode(PNG).decode()
FACTS = {
    "kind": "chart", "title": "마포구 월별 매출", "region": "마포구", "period": "2026-01~2026-06", "y_unit": "백만원",
    "series": [{"label": "매출", "values": [{"x": f"2026-0{i + 1}", "y": v}
                                           for i, v in enumerate([1250, 1310, 1290, 870, 1180, 1240])]}],
    "anomalies": [{"x": "2026-04", "note": "전월 대비 급감"}], "unreadable": [],
}


def _img_msg(text="이 차트 봐줘", uri=URI):
    return HumanMessage(content=[{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": uri}}])


def _ex(intent, **slots):
    return ExtractedIntent(intent=intent, mode="action", slots=Slots(**slots), span="")


@pytest.fixture(autouse=True)
def _objects(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "OBJECT_DIR", tmp_path / "uploads")
    return tmp_path


# ---------- has_attachment ----------
def test_has_attachment_on_str_and_list_content():
    assert not A.has_attachment({"messages": [HumanMessage(content="그냥 텍스트")]})
    assert not A.has_attachment({"messages": [HumanMessage(content=[{"type": "text", "text": "글만"}])]})
    assert A.has_attachment({"messages": [_img_msg()]})
    # 이미지가 이전 메시지에만 있고 이번 턴은 텍스트면 읽지 않는다
    assert not A.has_attachment({"messages": [_img_msg(), AIMessage(content="네"), HumanMessage(content="다음")]})


def test_has_attachment_skips_already_read_image():
    att_id = A.attachment_id_of(URI)
    assert not A.has_attachment({"messages": [_img_msg()], "attachments": {att_id: {"facts": FACTS}}})
    other = "data:image/png;base64," + base64.b64encode(b"other").decode()
    assert A.has_attachment({"messages": [_img_msg(uri=other)], "attachments": {att_id: {"facts": FACTS}}})


# ---------- reader ----------
async def test_reader_stores_facts_and_last_attachment_without_base64(monkeypatch, _objects):
    calls = []

    async def fake_vlm(url, text, config):
        calls.append((url[:22], text))
        return {**FACTS, "text": None}

    monkeypatch.setattr(A, "read_image_facts", fake_vlm)
    out = await A.build_attachment_reader()({"messages": [_img_msg("4월에 왜 떨어졌어?")], "turn": 1}, {})
    att_id = A.attachment_id_of(URI)
    rec = out["attachments"][att_id]
    assert calls == [("data:image/png;base64,", "4월에 왜 떨어졌어?")]
    assert rec["kind"] == "chart" and rec["turn"] == 1 and rec["facts"]["region"] == "마포구"
    assert "text" not in rec["facts"]  # 빈 값은 버림
    assert out["entity_memory"]["last_attachment"] == {"attachment_id": att_id, "kind": "chart", "title": "마포구 월별 매출"}
    # 원본은 상태에 없고 오브젝트 스토리지(대역: 로컬 파일)에만
    assert URI.split(",")[1] not in json.dumps(out, ensure_ascii=False)
    stored = list((_objects / "uploads").iterdir())
    assert len(stored) == 1 and stored[0].read_bytes() == PNG and rec["object_key"].endswith(stored[0].name)


async def test_reader_injected_and_vlm_failure_recorded():
    async def boom(url, text, config):
        raise RuntimeError("401")

    out = await A.build_attachment_reader(read=boom)({"messages": [_img_msg()], "turn": 2}, {})
    rec = next(iter(out["attachments"].values()))
    assert rec["facts"]["error"].startswith("RuntimeError") and rec["kind"] == "other"


async def test_reader_noop_when_nothing_new():
    assert await A.build_attachment_reader(read=None)({"messages": [HumanMessage(content="안녕")]}, {}) == {}


# ---------- attachment worker ----------
def _att_arg(facts, slots=None, turn=1):
    att_id = "att_x"
    t = new_task("T1", _ex("attachment_analysis", attachment_id=att_id), turn, {})
    t["slots"].update(slots or {})
    return {"task": t, "turn": turn, "attachments": {att_id: {"kind": facts["kind"], "facts": facts, "object_key": "k", "turn": 1}}}


async def test_worker_summarizes_april_drop():
    out = await A.build_attachment_worker(None)(_att_arg(FACTS), {})
    t = out["task_stack"][0]
    assert t["status"] == "done" and t["result"]["facts"] == FACTS
    s = t["result"]["summary"]
    assert "4월 870" in s and "1,290" in s and "420 감소" in s and "-32.6%" in s and "백만원" in s


def test_new_task_fills_attachment_from_last_attachment():
    t = new_task("T1", _ex("attachment_analysis"), 2, {"last_attachment": {"attachment_id": "att_9", "kind": "chart"}})
    assert t["slots"]["attachment_id"] == "att_9" and t["missing"] == []


async def test_worker_blocks_on_unreadable_unit_then_chip_fills_it():
    facts = {**FACTS, "y_unit": None, "unreadable": ["y_unit"]}
    arg = _att_arg(facts)
    t = (await A.build_attachment_worker(None)(arg, {}))["task_stack"][0]
    assert t["status"] == "blocked" and t["missing"] == ["y_unit"]
    chips = rule_followups([t])
    assert [c["label"] for c in chips] == ["단위: 백만원", "단위: 천원", "단위: 원"]
    # 칩 클릭 → compute_missing(카탈로그 필수만)이 비어 pending → worker 재실행
    t2 = apply_payload([t], chips[0]["payload"], 2)[0]
    assert t2["status"] == "pending" and t2["slots"]["y_unit"] == "백만원"
    out = await A.build_attachment_worker(None)({**arg, "task": t2, "turn": 2}, {})
    done = out["task_stack"][0]
    assert done["status"] == "done" and done["missing"] == [] and "단위: 백만원" in done["result"]["summary"]
    assert out["attachments"]["att_x"]["facts"]["y_unit"] == "백만원"  # 확정 단위를 상태에도 반영


async def test_worker_fails_on_missing_or_unreadable_attachment():
    arg = _att_arg(FACTS)
    arg["task"]["slots"]["attachment_id"] = "att_none"
    assert (await A.build_attachment_worker(None)(arg, {}))["task_stack"][0]["error"] == "attachment_not_found"
    bad = _att_arg({"kind": "other", "error": "x", "unreadable": ["전체"]})
    assert (await A.build_attachment_worker(None)(bad, {}))["task_stack"][0]["status"] == "failed"


# ---------- report writer ----------
DATASET = {"columns": ["month", "industry", "stores"], "rows": [["2026-03", "음식점", 412], ["2026-04", "음식점", 377]],
           "row_count": 2, "source": "sangkwon"}


def _report_arg(inputs, slots=None):
    t = new_task("T3", _ex("report_compose", tone="3분기 보고서", **(slots or {})), 2, {})
    gids = select_guidelines(t)
    return {"task": {**t, "guidelines": gids}, "turn": 2, "inputs": inputs, "attachments": {},
            "guidelines_text": render_guidelines(gids)}


def _inputs():
    return {
        "T1": {"intent": "attachment_analysis", "status": "done", "slots": {"attachment_id": "att_x"},
               "result": {"facts": FACTS, "summary": "..."}},
        "T2": {"intent": "data_query", "status": "done", "slots": {"metric": "점포 수", "source": "sangkwon", "group_by": ["업종"]},
               "result": DATASET},
    }


async def test_report_writer_prompt_has_only_inputs_and_guidelines():
    seen = {}

    async def fake_write(messages, config):
        seen["sys"], seen["human"] = messages[0].content, messages[1].content
        return "마포구 매출 동향 보고\n□ 현황\n○ 4월 매출 870백만원, 전월 1,290 대비 420 감소(-32.6%)\n○ 음식점 점포 412→377"

    out = await A.build_report_writer(None, write=fake_write)(_report_arg(_inputs()), {})
    t = out["task_stack"][0]
    assert t["status"] == "done" and t["result"]["sources"] == ["T1", "T2"]
    assert t["result"]["summary"].startswith("마포구 매출 동향 보고") and "unverified_numbers" not in t["result"]
    assert "[report.formal]" in seen["sys"] and "[org.security]" in seen["sys"] and "3분기 보고서" in seen["sys"]
    assert "[sql.readonly]" not in seen["sys"]  # 다른 작업용 가이드라인은 안 들어감
    assert "[자료]에 없는 수치" in seen["sys"]
    body = json.loads(seen["human"].split("[자료]\n", 1)[1].split("\n\n")[0])
    assert [m["type"] for m in body] == ["attachment", "dataset"]
    assert body[1]["rows"] == DATASET["rows"] and body[1]["conditions"]["group_by"] == ["업종"]
    assert {"from": "2026-03", "to": "2026-04", "diff": -420} .items() <= next(
        c for c in body[0]["changes"] if c["to"] == "2026-04").items()


async def test_report_writer_flags_numbers_not_in_inputs(monkeypatch):
    async def liar(messages, config):
        return "○ 4월 매출 870백만원\n○ 전년 동기 대비 15.3% 감소, 폐업 점포 58개"

    monkeypatch.setattr(A, "write_report", liar)
    t = (await A.build_report_writer(None)(_report_arg(_inputs()), {}))["task_stack"][0]
    assert t["result"]["unverified_numbers"] == ["15.3", "58"]
    assert "※ 검증 필요" in t["result"]["summary"]


async def test_report_writer_without_inputs_does_not_call_llm():
    async def must_not_call(messages, config):
        raise AssertionError("LLM 호출 금지")

    failed = {"T2": {"intent": "data_query", "status": "failed", "slots": {}, "result": None}}
    for inputs in ({}, failed):
        t = (await A.build_report_writer(None, write=must_not_call)(_report_arg(inputs), {}))["task_stack"][0]
        assert t["status"] == "done" and t["result"]["empty"] and "근거 자료가 없어" in t["result"]["summary"]
    assert "T2(data_query) failed" in t["result"]["summary"]


async def test_report_writer_uses_attachment_named_in_slots():
    seen = {}

    async def fake_write(messages, config):
        seen["human"] = messages[1].content
        return "○ 4월 870"

    arg = _report_arg({}, {"attachment_id": "att_x"})
    arg["attachments"] = {"att_x": {"kind": "chart", "facts": FACTS, "object_key": "k", "turn": 1}}
    t = (await A.build_report_writer(None, write=fake_write)(arg, {}))["task_stack"][0]
    assert t["result"]["sources"] == ["att_x"] and "마포구" in seen["human"]


def test_extractor_hint_mentions_intents():
    for w in ("attachment_analysis", "report_compose", "last_attachment", "tone"):
        assert w in A.EXTRACTOR_HINT


def test_scenarios_module_imports():
    from followups.scenarios import multimodal

    assert set(multimodal.SCENARIOS) == {"mapo_report", "chart_only"}
    assert multimodal.SCENARIOS["mapo_report"]["steps"][1] == ("chip", "상권 DB(월별)")
