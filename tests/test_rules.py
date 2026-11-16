"""LLM 없이 도는 규칙 테스트: reducer, 의존성, planner, follow-up 우선순위, 칩 payload."""
from followups.state import (
    ExtractedIntent, Slots, apply_payload, link_dependencies, merge_tasks, new_task, plan,
    rule_followups, select_followups,
)


def _ex(intent, mode="action", **slots):
    return ExtractedIntent(intent=intent, mode=mode, slots=Slots(**slots), span="")


MEM = {"last_product": {"product_id": "P-HOODIE-01", "name": "후드티", "option": "블랙/L"}}


def test_merge_tasks_parallel_updates_by_id():
    a = [{"id": "T1", "status": "pending", "slots": {}}, {"id": "T2", "status": "pending"}]
    out = merge_tasks(a, [{"id": "T2", "status": "done"}, {"id": "T3", "status": "pending"}])
    assert {t["id"]: t["status"] for t in out} == {"T1": "pending", "T2": "done", "T3": "pending"}


def test_order_fills_from_entity_memory():
    t = new_task("T1", _ex("order_product"), 1, MEM)
    assert t["slots"] == {"product_id": "P-HOODIE-01", "option": "블랙/L"}
    assert t["missing"] == [] and t["risk"] == "high" and t["phase"] == "prepare"


def test_address_inquiry_has_no_missing_and_order_depends_on_it():
    order = new_task("T1", _ex("order_product"), 1, MEM)
    addr = new_task("T2", _ex("change_shipping_address", mode="inquiry"), 1, MEM)
    assert addr["missing"] == []
    changed = link_dependencies([order, addr])
    assert changed[0]["depends_on"] == ["T2"]


def test_plan_prepare_runs_even_if_deps_open_but_checkout_waits():
    order = {**new_task("T1", _ex("order_product"), 1, MEM), "depends_on": ["T2"]}
    addr = {**new_task("T2", _ex("change_shipping_address", mode="inquiry"), 1, MEM)}
    ready, _ = plan([order, addr])
    assert {t["id"] for t in ready} == {"T1", "T2"}  # prepare + 문의는 병렬

    order_ck = {**order, "phase": "checkout", "status": "blocked", "missing": []}
    addr_blocked = {**addr, "status": "blocked", "missing": ["scope", "address"], "mode": "action"}
    ready, changed = plan([order_ck, addr_blocked])
    assert ready == [] and changed == []

    addr_done = {**addr_blocked, "status": "done", "missing": []}
    ready, changed = plan([order_ck, addr_done])
    assert [t["id"] for t in ready] == ["T1"] and ready[0]["status"] == "awaiting_confirm"


def test_followup_priority_and_hide_next_when_blocked():
    failed = {"id": "T1", "intent": "order_product", "status": "failed", "missing": []}
    blocked = {"id": "T2", "intent": "change_shipping_address", "status": "blocked", "missing": ["scope", "address"], "converted": True}
    rules = rule_followups([blocked, failed])
    assert rules[0]["type"] == "recover"
    nxt = [{"label": "배송 알림", "type": "next", "payload": {"op": "say", "text": "x"}}]
    sel = select_followups(rules, nxt)
    assert all(f["type"] != "next" for f in sel)
    assert {f["payload"]["task_id"] for f in sel} == {"T1"}  # 가장 급한 작업 하나의 선택지만


def test_scope_chip_then_address_text_flow():
    addr = {"id": "T2", "intent": "change_shipping_address", "mode": "action", "status": "blocked",
            "slots": {}, "missing": ["scope", "address"], "converted": True}
    chips = rule_followups([addr])
    assert [c["label"] for c in chips] == ["이번 주문 배송지 변경", "기본 배송지 변경", "변경 안 함"]
    after = apply_payload([addr], chips[0]["payload"], 2)[0]
    assert after["slots"]["scope"] == "this_order" and after["missing"] == ["address"] and after["status"] == "blocked"
    assert rule_followups([after])[0]["payload"]["op"] == "ask_text"


def test_skip_task_unblocks_order():
    order = {"id": "T1", "intent": "order_product", "phase": "checkout", "status": "blocked", "missing": [], "depends_on": ["T2"]}
    addr = {"id": "T2", "intent": "change_shipping_address", "status": "blocked", "missing": ["scope"], "slots": {}, "mode": "action"}
    changed = apply_payload([order, addr], {"op": "skip_task", "task_id": "T2"}, 2)
    ready, _ = plan(merge_tasks([order, addr], changed))
    assert [t["id"] for t in ready] == ["T1"]


def test_product_query_counts_as_product_and_does_not_fall_back_to_other_product():
    # 이전 대화에 후드티가 있어도, 사용자가 '패딩'을 말하면 후드티로 채우면 안 된다
    t = new_task("T1", _ex("order_product", product_query="경량 패딩", option="블랙/L"), 1, MEM)
    assert "product_id" not in t["slots"] and t["missing"] == []
    # 같은 상품을 검색어로 가리키면 메모리로 채운다
    t2 = new_task("T1", _ex("order_product", product_query="후드티"), 1, MEM)
    assert t2["slots"]["product_id"] == "P-HOODIE-01"


def test_ambiguous_product_offers_candidate_chips():
    t = {"id": "T1", "intent": "order_product", "status": "blocked", "missing": ["product_id"],
         "result": {"candidates": [{"product_id": "A", "name": "후드티 A"}, {"product_id": "B", "name": "후드티 B"}]}}
    assert [c["label"] for c in rule_followups([t])] == ["후드티 A", "후드티 B"]


def test_merge_slots_keeps_both_sub_questions():
    from followups.state import merge_slots
    out = merge_slots({"question": "교환이 나은지 환불이 나은지", "product_query": "후드티"},
                      {"question": "쿠폰 처리 방식", "product_query": "후드티", "option": None})
    assert out["question"] == "교환이 나은지 환불이 나은지 / 쿠폰 처리 방식"
    assert "option" not in out
