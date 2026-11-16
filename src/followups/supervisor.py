"""판단·조사형 의도 전용 supervisor. 하위 에이전트를 도구로 감싸고(Subagents 패턴), 최종 답만 메인으로 올린다."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

import asyncio

from followups.llm import MAIN_REASONING_EFFORT, MODEL_FAST, MODEL_MAIN, llm
from followups.state import SubagentResult
from followups.tools import Toolbox


def _as_tool(agent, name: str, description: str) -> StructuredTool:
    async def _run(request: str, config: RunnableConfig) -> str:
        # 컨텍스트 격리: 하위 에이전트는 요청 한 줄만 받고, 최종 답만 돌려준다
        out = await agent.ainvoke({"messages": [{"role": "user", "content": request}]}, config=config)
        content = out["messages"][-1].content
        return content if isinstance(content, str) else str(content)

    return StructuredTool.from_function(coroutine=_run, name=name, description=description)


def build_supervisor(box: Toolbox):
    order_agent = create_agent(
        llm(MODEL_FAST),
        tools=box.subset("payment", names=["list_orders"]) + box.subset("member", names=["get_member"]),
        system_prompt="주문 조회 담당. 도구로 주문과 회원 정보를 조회하고, 질문에 필요한 사실(주문 id, 옵션, 결제 구성, 배송완료 경과일, 회원 등급)만 짧게 답한다.",
        name="order_agent",
    )
    policy_agent = create_agent(
        llm(MODEL_FAST),
        tools=box.subset("policy"),
        system_prompt="약관 해석 담당. search_policy로 조항을 찾고, 결론과 근거 조항 id만 3문장 이내로 답한다. 조항에 없는 내용은 지어내지 않는다.",
        name="policy_agent",
    )
    pricing_agent = create_agent(
        llm(MODEL_FAST),
        tools=box.subset("payment", names=["quote_return"]),
        system_prompt="금액 계산 담당. quote_return으로 환불/교환 금액을 계산하고, 도구 결과의 숫자만 모드별 한 줄씩 답한다. 직접 계산하거나 설명을 덧붙이지 않는다.",
        name="pricing_agent",
    )
    tools = [
        _as_tool(order_agent, "ask_order", "회원의 주문 내역·결제 구성·배송 상태를 조회한다. 요청에 user_id를 포함할 것."),
        _as_tool(policy_agent, "ask_policy", "반품·교환·쿠폰·적립금·배송 약관을 해석한다. 결론과 근거 조항만 돌려준다."),
        _as_tool(pricing_agent, "ask_pricing", "주문 id로 환불(refund)과 교환(exchange) 시 금액을 계산한다."),
    ]
    return create_agent(
        llm(MODEL_MAIN, reasoning_effort=MAIN_REASONING_EFFORT),
        tools=tools,
        system_prompt=(
            "커머스 상담 supervisor. 하위 에이전트(ask_order, ask_policy, ask_pricing)를 필요한 만큼만 순서대로 호출한다. "
            "앞 결과를 보고 다음 호출을 정한다. 서로 독립인 호출은 한 번에 함께 부른다. "
            "사용자가 비교를 물으면 summary 첫 문장에 추천과 이유를 쓴다. 금액은 ask_pricing 결과 숫자만 쓴다. "
            "사용자에게 직접 질문하지 말고, 더 필요한 정보는 needs_input에, "
            "이어서 할 만한 실행은 proposed_actions에 담는다. 실행(교환 신청 등)은 하지 않는다."
        ),
        # Sonnet 5.5는 강제 tool_choice(any/tool)를 거부 → ToolStrategy 대신 제공자 네이티브 json_schema 사용
        response_format=ProviderStrategy(SubagentResult),
        name="supervisor",
    )


def build_refund_fastpath(box: Toolbox):
    """교환/환불 비교 전용 빠른 경로.

    일반 supervisor는 Sonnet이 '누구를 부를지' 판단하는 왕복을 2번 거친다 (약 12초).
    이 질문은 필요한 정보가 늘 같으므로(주문, 약관, 두 모드 금액) 처음부터 병렬로 모으고
    Sonnet은 마지막 종합 1번만 한다. 주문을 하나로 특정하지 못하면 None을 돌려 일반 경로로 넘긴다.
    """
    # 출력 축소: 추론 강도 low + 상한 400토큰 + '조항별 한 줄' 형식 (이전: 911토큰 중 추론 475, 5.2초)
    policy_agent = create_agent(
        llm(MODEL_FAST, max_tokens=400, reasoning_effort="low"),
        tools=box.subset("policy"),
        system_prompt=(
            "약관 해석 담당. search_policy를 한 번만 호출한다 (질문의 핵심 키워드를 한 쿼리에 모두 넣는다). "
            "답은 관련 조항마다 한 줄씩 '[조항id] 결론' 형식으로만 쓴다. 최대 4줄, 설명·인사·요약 문단 금지. "
            "조항에 없는 내용은 쓰지 않는다."
        ),
        name="policy_agent",
    )
    synth = llm(MODEL_MAIN, reasoning_effort=MAIN_REASONING_EFFORT).with_structured_output(SubagentResult, method="json_schema")

    async def resolve_order(user_id: str, product_query: str | None, config) -> dict | None:
        orders, products = await asyncio.gather(
            box.call("list_orders", {"user_id": user_id}, config),
            box.call("search_product", {"query": product_query or ""}, config),
        )
        ids = {p["product_id"] for p in products.get("results", [])}
        delivered = [o for o in orders.get("orders", []) if o["status"] == "delivered" and (not ids or o["product_id"] in ids)]
        return delivered[0] if len(delivered) == 1 else None

    async def pricing(order: dict, config) -> dict:
        refund, exchange = await asyncio.gather(
            box.call("quote_return", {"order_id": order["order_id"], "mode": "refund"}, config),
            box.call("quote_return", {"order_id": order["order_id"], "mode": "exchange"}, config),
        )
        return {"refund": refund, "exchange": exchange}

    async def run(user_id: str, question: str, utterance: str, product_query: str | None, config) -> SubagentResult | None:
        order = await resolve_order(user_id, product_query, config)
        if not order:
            return None
        policy_out, prices = await asyncio.gather(
            policy_agent.ainvoke({"messages": [{"role": "user", "content": f"반품·교환·쿠폰·적립금 관점에서: {question}"}]}, config=config),
            pricing(order, config),
        )
        policy = policy_out["messages"][-1].content
        return await synth.ainvoke([
            {"role": "system", "content": (
                "커머스 상담 종합 담당. 아래 사실만 근거로 답한다. 사용자가 비교를 물으면 summary 첫 문장에 추천과 이유를 쓴다. "
                "금액은 금액 계산 결과의 숫자만 쓰고 직접 계산하지 않는다. 경과일은 '배송완료 후 N일'로 표현한다. "
                "proposed_actions는 버튼 문구 최대 2개, 각 12자 이내."
            )},
            {"role": "user", "content": f"사용자 원문: {utterance}\n질문: {question}\n주문: {order}\n약관 해석: {policy}\n금액 계산: {prices}"},
        ], config={**config, "run_name": "refund_fastpath.synth"})

    return run
