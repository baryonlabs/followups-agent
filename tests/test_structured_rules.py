"""1단계: 구조적 멀티턴 규칙 일반화 (produces/consumes 의존성, refine, remember, 선택지 칩, 가이드라인)."""
from followups.state import (
    GUIDELINES, ExtractedIntent, Slots, apply_payload, attach_guidelines, link_dependencies, merge_slots,
    merge_tasks, new_task, plan, render_guidelines, rule_followups, select_followups, select_guidelines,
)


def _ex(intent, mode="action", **slots):
    return ExtractedIntent(intent=intent, mode=mode, slots=Slots(**slots), span="")


def _tasks_for_report():
    q = new_task("T2", _ex("data_query", mode="inquiry", metric="점포 수", group_by=["업종"]), 1, {})
    r = new_task("T3", _ex("report_compose", tone="3분기 보고서"), 1, {})
    return q, r


def test_new_intents_force_action_and_compute_missing():
    q, r = _tasks_for_report()
    assert q["mode"] == "action" and q["missing"] == [] and q["route"] == "sql"  # 출처는 worker가 고르거나 묻는다
    assert r["missing"] == [] and r["route"] == "report"


def test_report_depends_on_open_producers_and_waits():
    q, r = _tasks_for_report()
    q = {**q, "status": "blocked", "missing": ["source"]}  # worker가 출처 선택지를 내고 막힌 상태
    changed = link_dependencies([q, r])
    assert changed == [{**r, "depends_on": ["T2"]}]
    tasks = merge_tasks([q, r], changed)
    ready, _ = plan(tasks)
    assert ready == []  # 보고서는 데이터가 나올 때까지 대기

    done = merge_tasks(tasks, [{"id": "T2", "status": "done", "missing": []}])
    ready, _ = plan(done)
    assert [t["id"] for t in ready] == ["T3"]


def test_new_consumer_uses_latest_done_producer_without_blocking():
    q, r = _tasks_for_report()
    q = {**q, "status": "done"}
    changed = link_dependencies([q, r])
    assert changed == [{**r, "depends_on": ["T2"]}]  # 입력으로 연결 (아까 조회한 걸 보고서로)
    ready, _ = plan(merge_tasks([q, r], changed))
    assert [t["id"] for t in ready] == ["T3"]  # 이미 끝났으니 바로 실행


def test_choices_result_becomes_slot_chips_then_fill():
    q, _ = _tasks_for_report()
    q = {**q, "status": "blocked", "missing": ["source"], "result": {"choices": {"source": [
        {"label": "상권 DB(월별)", "value": "sangkwon"}, {"label": "사업자 DB(분기별)", "value": "bizreg"}]}}}
    chips = rule_followups([q])
    assert [c["label"] for c in chips] == ["상권 DB(월별)", "사업자 DB(분기별)"]
    after = apply_payload([q], chips[0]["payload"], 2)[0]
    assert after["slots"]["source"] == "sangkwon" and after["status"] == "pending"


def test_refine_reruns_producer_and_its_consumers():
    q, r = _tasks_for_report()
    q = {**q, "slots": {**q["slots"], "source": "sangkwon"}, "missing": [], "status": "done"}
    r = {**r, "status": "done", "depends_on": ["T2"]}
    changed = apply_payload([q, r], {"op": "refine", "task_id": "T2", "slots": {"group_by": ["구"]}}, 3)
    by = {t["id"]: t for t in changed}
    assert by["T2"]["slots"]["group_by"] == ["업종", "구"] and by["T2"]["status"] == "pending"
    assert by["T3"]["status"] == "pending" and by["T3"]["depends_on"] == ["T2"]
    ready, _ = plan(merge_tasks([q, r], changed))
    assert [t["id"] for t in ready] == ["T2"]  # 보고서는 재조회 뒤에


def test_list_slot_merge_is_union():
    assert merge_slots({"group_by": ["업종"]}, {"group_by": ["구", "업종"]})["group_by"] == ["업종", "구"]


def test_preference_needs_consent_chip():
    t = new_task("T5", _ex("set_preference", pref_key="decimal_places", pref_value="2"), 4, {})
    t = {**t, "status": "blocked"}
    assert t["missing"] == ["consent"]
    chips = rule_followups([t])
    assert [c["label"] for c in chips] == ["앞으로도 적용", "이번만"] and chips[0]["type"] == "confirm"
    yes = apply_payload([t], chips[0]["payload"], 5)[0]
    assert yes["status"] == "pending" and yes["slots"]["consent"] is True
    no = apply_payload([t], chips[1]["payload"], 5)[0]
    assert no["status"] == "cancelled"


def test_policy_chip_ranks_after_recover_before_slot():
    rule = [
        {"label": "a", "type": "slot", "payload": {"op": "fill_slots", "task_id": "T1"}},
        {"label": "b", "type": "policy", "payload": {"op": "say", "text": "x", "task_id": None}},
    ]
    from followups.state import PRIORITY
    rule.sort(key=lambda f: PRIORITY[f["type"]])
    assert select_followups(rule, [])[0]["type"] == "policy"


def test_guidelines_are_scoped_per_task_and_user_prefs_last():
    q, r = _tasks_for_report()
    prof = {"decimal_places": "2"}
    assert select_guidelines(q, prof) == ["org.security", "sql.readonly", "user.decimal_places"]
    assert select_guidelines(r, prof) == ["org.security", "report.formal", "user.decimal_places"]
    assert "sql.readonly" not in select_guidelines(r)  # 보고서에 SQL 지침이 섞이지 않는다
    text = render_guidelines(select_guidelines(q, prof), prof)
    assert text.index("org.security") < text.index("사용자 선호")
    assert "소수점 2자리" in text
    assert attach_guidelines([{**q, "guidelines": select_guidelines(q)}]) == []  # 안 바뀌면 갱신 없음
    assert all(g["priority"] >= 0 for g in GUIDELINES.values())


def test_attachment_analysis_uses_last_attachment():
    t = new_task("T1", _ex("attachment_analysis"), 1, {"last_attachment": {"attachment_id": "A1"}})
    assert t["slots"]["attachment_id"] == "A1" and t["missing"] == []
