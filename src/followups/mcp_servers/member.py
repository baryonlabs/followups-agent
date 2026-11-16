"""MCP Server B: 회원/배송 DB (stdio)."""
from mcp.server.fastmcp import FastMCP

from followups.db import store

mcp = FastMCP("member", log_level="WARNING")


@mcp.tool()
def get_member(user_id: str) -> dict:
    """회원 정보와 기본 배송지, 이번 결제용 배송지를 조회한다."""
    with store("member") as d:
        u = d["users"].get(user_id)
        if not u:
            return {"error": f"회원 {user_id} 없음"}
        return {"user_id": user_id, **u}


@mcp.tool()
def check_address_change(user_id: str) -> dict:
    """지금 배송지를 바꿀 수 있는지 확인한다 (이번 주문 / 기본 배송지 둘 다)."""
    with store("member") as d:
        ok = d["address_change_open"]
        u = d["users"][user_id]
    return {
        "possible": ok,
        "scopes": ["this_order", "default"],
        "current_default": u["default_address"],
        "note": "결제 전이면 이번 주문 배송지를, 언제든 기본 배송지를 바꿀 수 있습니다." if ok else "출고가 시작되어 변경할 수 없습니다.",
    }


@mcp.tool()
def set_checkout_address(user_id: str, address: str) -> dict:
    """이번 결제(다음 checkout 한 건)에만 쓰는 배송지를 지정한다."""
    with store("member", write=True) as d:
        d["users"][user_id]["checkout_address"] = address
    return {"ok": True, "scope": "this_order", "address": address}


@mcp.tool()
def update_default_address(user_id: str, address: str) -> dict:
    """회원의 기본 배송지를 바꾼다."""
    with store("member", write=True) as d:
        d["users"][user_id]["default_address"] = address
    return {"ok": True, "scope": "default", "address": address}


if __name__ == "__main__":
    mcp.run()
