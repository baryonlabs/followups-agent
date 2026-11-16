"""MCP Server C: 약관/정책 검색 (stdio). policy subagent 전용."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("policy", log_level="WARNING")

SECTIONS = {
    "R-1": "제12조(청약철회) 회원은 상품을 배송받은 날부터 30일 이내에 청약을 철회(반품)할 수 있다. 단순 변심의 경우 반품 배송비 3,000원은 회원이 부담한다.",
    "R-2": "제13조(교환) 사이즈·색상 교환은 배송완료 후 30일 이내 1회 가능하다. 교환 왕복배송비 6,000원은 회원 부담이나, GOLD 이상 등급은 50%를 회사가 지원한다.",
    "C-1": "제20조(쿠폰) 쿠폰을 사용한 주문을 반품하면 사용한 쿠폰은 복구되지 않고 소멸한다. 부분 반품 시에도 동일하다. 교환은 주문이 유지되므로 쿠폰 혜택이 유지된다.",
    "C-2": "제21조(적립금) 반품 시 사용한 적립금은 환불일 기준으로 재적립되며 유효기간은 기존 유효기간을 따른다.",
    "D-1": "제30조(배송) 결제 완료 후 출고 전까지 배송지를 변경할 수 있다. 출고 후에는 택배사를 통해 변경을 요청해야 한다.",
    "D-2": "제31조(배송사고) 배송완료 처리 후 수령하지 못한 경우 7일 이내 신고하면 회사가 조사 후 재발송 또는 환불한다.",
}
KEYWORDS = {
    "R-1": ["반품", "환불", "철회", "변심"],
    "R-2": ["교환", "사이즈", "색상"],
    "C-1": ["쿠폰", "할인"],
    "C-2": ["적립금", "포인트"],
    "D-1": ["배송지", "주소"],
    "D-2": ["배송사고", "미수령", "분실"],
}


@mcp.tool()
def search_policy(query: str, top_k: int = 6) -> dict:
    """약관에서 질문과 관련된 조항을 모두 찾는다 (키워드가 하나라도 맞으면 포함). 조항 id와 원문을 돌려준다."""
    scored = sorted(
        ((sum(k in query for k in kws), sid) for sid, kws in KEYWORDS.items()),
        reverse=True,
    )
    hits = [sid for score, sid in scored if score > 0][:top_k]
    return {"sections": [{"id": s, "text": SECTIONS[s]} for s in hits]}


if __name__ == "__main__":
    mcp.run()
