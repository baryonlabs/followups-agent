"""5단계: 장기 사용자 메모리 (프로필 저장소 · memory_writer · after_turn). LLM 호출 없음."""
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from followups import db, memory
from followups.state import apply_payload, render_guidelines, select_guidelines


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    db.reset()
    yield tmp_path


def _pref_task(tid="T1", consent=None, **slots):
    s = {"pref_key": "decimal_places", "pref_value": "2", **slots}
    if consent is not None:
        s["consent"] = consent
    return {"id": tid, "intent": "set_preference", "mode": "action", "slots": s, "depends_on": [], "risk": "low",
            "route": "memory", "status": "pending" if consent else "blocked",
            "missing": [] if consent else ["consent"], "result": None, "error": None, "turn": 1}


def _arg(task, turn=2):
    return {"task": task, "user_id": "u1", "turn": turn, "utterance": "[칩] 앞으로도 적용", "inputs": {},
            "attachments": {}, "profile": {}, "guidelines_text": ""}


CFG = {"configurable": {"thread_id": "th-1"}}


# ---------- 저장소 ----------
def test_profile_crud_and_provenance():
    assert memory.load_profile("u1") == {}
    memory.save_pref("u1", "decimal_places", "2", source_thread="th-1", source_turn=3, evidence="소수점 둘째 자리까지")
    rows = memory.list_prefs("u1")
    assert rows["decimal_places"]["source_thread"] == "th-1" and rows["decimal_places"]["source_turn"] == 3
    assert rows["decimal_places"]["evidence"] == "소수점 둘째 자리까지" and rows["decimal_places"]["confirmed_at"]
    assert memory.load_profile("u1") == {"decimal_places": "2"}

    memory.save_pref("u1", "decimal_places", "3", source_thread="th-2", source_turn=1)
    assert memory.load_profile("u1") == {"decimal_places": "3"}
    assert memory.delete_pref("u1", "소수점") is True  # 별칭으로도 삭제
    assert memory.delete_pref("u1", "decimal_places") is False
    assert memory.load_profile("u1") == {}
    with db.store("profile") as d:
        audit = d["users"]["u1"]["audit"]
    assert [a["op"] for a in audit] == ["created", "updated", "deleted"]
    assert audit[1]["prev"] == "2" and audit[2]["prev"] == "3"


def test_load_profile_only_confirmed():
    memory.save_pref("u1", "currency", "천원", confirmed=False)
    memory.save_pref("u1", "table_format", "마크다운", source_thread="t", source_turn=1)
    assert memory.load_profile("u1") == {"table_format": "마크다운"}
    assert memory.load_profile(None) == {} and memory.load_profile("nobody") == {}


def test_normalize_key_and_value():
    assert memory.normalize_key("Decimal Places") == "decimal_places"
    assert memory.normalize_key("rounding") == "decimal_places"
    assert memory.normalize_key("표") == "table_format"
    assert memory.normalize_key("chart_color") == "chart_color"  # 일반 키
    assert memory.normalize_key("한글키") is None
    assert memory.normalize_value("decimal_places", "소수점 둘째 자리") == "2"
    assert memory.normalize_value("decimal_places", "2자리") == "2"


# ---------- memory_writer ----------
async def test_writer_requires_consent():
    w = memory.build_memory_writer()
    out = await w(_arg(_pref_task()), CFG)
    t = out["task_stack"][0]
    assert t["status"] == "blocked" and t["missing"] == ["consent"]
    assert memory.load_profile("u1") == {}


async def test_writer_saves_with_provenance():
    w = memory.build_memory_writer()
    out = await w(_arg(_pref_task(consent=True, pref_key="decimal", source_turn=1, evidence="둘째 자리까지")), CFG)
    t = out["task_stack"][0]
    assert t["status"] == "done" and t["result"]["saved"] == {"decimal_places": "2"}
    row = memory.list_prefs("u1")["decimal_places"]
    assert row["source_thread"] == "th-1" and row["source_turn"] == 1 and row["evidence"] == "둘째 자리까지"
    assert memory.load_profile("u1") == {"decimal_places": "2"}


@pytest.mark.parametrize("key,value", [
    ("show_pii", "true"), ("주민번호_표시", "보임"), ("table_format", "전화번호까지 마스킹 없이"), ("security", "off"),
])
async def test_writer_rejects_policy_override(key, value):
    w = memory.build_memory_writer()
    out = await w(_arg(_pref_task(consent=True, pref_key=key, pref_value=value)), CFG)
    t = out["task_stack"][0]
    assert t["status"] == "cancelled" and t["result"]["saved"] == {} and "정책" in t["result"]["reason"]
    assert memory.list_prefs("u1") == {}


# ---------- after_turn (실제 그래프, 가짜 추출기) ----------
@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from followups.graph import build_graph
    from followups.tools import Toolbox

    return build_graph(Toolbox(by_server={"payment": [], "member": [], "policy": []}))


async def _seed(app, user_text, tasks=(), turn=1):
    cfg = {"configurable": {"thread_id": f"th-{uuid.uuid4().hex[:6]}"}}
    await app.aupdate_state(cfg, {
        "messages": [HumanMessage(content=user_text), AIMessage(content="표로 정리했습니다.")],
        "user_id": "u1", "turn": turn, "task_stack": list(tasks), "answer": "표로 정리했습니다.", "followups": [],
    }, as_node="followup_generator")
    return cfg, (await app.aget_state(cfg)).values


class FakeExtractor:
    def __init__(self, result):
        self.result, self.calls = result, 0

    async def __call__(self, user_text, answer, profile, config):
        self.calls += 1
        return self.result


CAND = {"found": True, "pref_key": "decimal_places", "pref_value": "2", "evidence_span": "소수점 둘째 자리까지"}
DONE = {"id": "T1", "intent": "data_query", "mode": "action", "slots": {}, "status": "done", "turn": 1, "route": "sql"}


async def test_after_turn_adds_task_and_chips(app):
    cfg, out = await _seed(app, "앞으로 숫자는 반올림하지 말고 소수점 둘째 자리까지 보여줘", [DONE])
    ex = FakeExtractor(CAND)
    chips = await memory.after_turn(app, cfg, out, extractor=ex)
    assert ex.calls == 1
    assert [c["label"] for c in chips] == ["앞으로도 적용", "이번만"]
    assert all(c["payload"] == {"op": "remember", "task_id": "T2", "confirm": c["label"] == "앞으로도 적용"} for c in chips)
    snap = await app.aget_state(cfg)
    t2 = next(t for t in snap.values["task_stack"] if t["id"] == "T2")
    assert t2["status"] == "blocked" and t2["missing"] == ["consent"] and t2["route"] == "memory"
    assert t2["slots"]["evidence"] == "소수점 둘째 자리까지" and t2["turn"] == 1
    assert snap.next == ()
    assert memory.load_profile("u1") == {}  # 동의 전에는 기록 없음

    # 칩 클릭 → remember → memory_writer → 프로필 기록 → 가이드라인 반영 (기관 정책 뒤)
    [nt] = apply_payload(snap.values["task_stack"], chips[0]["payload"], 2)
    assert nt["status"] == "pending" and nt["slots"]["consent"] is True
    res = await memory.build_memory_writer()(_arg(nt), cfg)
    assert res["task_stack"][0]["status"] == "done"
    profile = memory.load_profile("u1")
    assert memory.list_prefs("u1")["decimal_places"]["source_thread"] == cfg["configurable"]["thread_id"]
    text = render_guidelines(select_guidelines({"route": "sql", "intent": "data_query"}, profile), profile)
    assert "[사용자 선호] 숫자는 반올림하지 않고 소수점 2자리까지" in text
    assert text.index("org.security") < text.index("사용자 선호")


async def test_after_turn_skips_without_cue(app):
    cfg, out = await _seed(app, "마포구 업종별 점포 수 알려줘", [DONE])
    ex = FakeExtractor(CAND)
    assert await memory.after_turn(app, cfg, out, extractor=ex) == []
    assert ex.calls == 0


async def test_after_turn_skips_interrupt_and_policy_block(app):
    cfg, out = await _seed(app, "앞으로 소수점 둘째 자리까지", [DONE])
    ex = FakeExtractor(CAND)
    assert await memory.after_turn(app, cfg, {**out, "__interrupt__": [object()]}, extractor=ex) == []
    assert await memory.after_turn(app, cfg, {**out, "guard": {"blocked": True}}, extractor=ex) == []
    assert ex.calls == 0


async def test_after_turn_duplicate_suppression(app):
    ex = FakeExtractor(CAND)
    # 1) 의도 추출기가 이번 턴에 이미 set_preference 작업을 만들었음
    cfg, out = await _seed(app, "앞으로 소수점 둘째 자리까지", [{**_pref_task(), "turn": 1}])
    assert await memory.after_turn(app, cfg, out, extractor=ex) == []
    # 2) 같은 키의 열린 작업이 이전 턴부터 있음
    cfg, out = await _seed(app, "앞으로 소수점 둘째 자리까지", [{**_pref_task(), "turn": 1}], turn=2)
    assert await memory.after_turn(app, cfg, out, extractor=ex) == []
    # 3) 프로필에 같은 값이 이미 확정됨
    memory.save_pref("u1", "decimal_places", "2", source_thread="x", source_turn=1)
    cfg, out = await _seed(app, "앞으로 소수점 둘째 자리까지", [DONE])
    assert await memory.after_turn(app, cfg, out, extractor=ex) == []
    # 4) 추출기가 후보 없음 / 정책 위반 후보
    cfg, out = await _seed(app, "앞으로 그렇게 해줘", [DONE])
    assert await memory.after_turn(app, cfg, out, extractor=FakeExtractor({"found": False})) == []
    bad = FakeExtractor({"found": True, "pref_key": "show_pii", "pref_value": "true", "evidence_span": ""})
    assert await memory.after_turn(app, cfg, out, extractor=bad) == []
    snap = await app.aget_state(cfg)
    assert [t["id"] for t in snap.values["task_stack"]] == ["T1"]


async def test_after_turn_new_value_differs_from_profile(app):
    memory.save_pref("u1", "decimal_places", "1", source_thread="x", source_turn=1)
    cfg, out = await _seed(app, "앞으로 소수점 둘째 자리까지", [DONE])
    chips = await memory.after_turn(app, cfg, out, extractor=FakeExtractor(CAND))
    assert [c["label"] for c in chips] == ["앞으로도 적용", "이번만"]


def test_scenarios_import():
    from followups.scenarios.memory import SCENARIOS

    assert {"memory_remember", "memory_once"} <= set(SCENARIOS)
    assert all(callable(s["expect"]) and s["steps"] for s in SCENARIOS.values())
