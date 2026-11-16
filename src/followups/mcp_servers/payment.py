"""MCP Server A: 상품/정산/결제 (stdio)."""
from mcp.server.fastmcp import FastMCP

from followups.db import store

mcp = FastMCP("payment", log_level="WARNING")


@mcp.tool()
def search_product(query: str) -> dict:
    """상품명 키워드로 상품을 찾고 옵션별 재고를 돌려준다."""
    q = query.replace(" ", "")
    with store("payment") as d:
        hits = [
            {"product_id": pid, **p}
            for pid, p in d["products"].items()
            if any(tok in p["name"].replace(" ", "") for tok in [q, q[:2]]) or pid == query
        ]
    return {"results": hits}


@mcp.tool()
def create_draft_order(user_id: str, product_id: str, option: str) -> dict:
    """결제 전 임시 주문(draft)을 만든다. 돈이 나가지 않는다."""
    with store("payment", write=True) as d:
        p = d["products"].get(product_id)
        if not p:
            return {"error": f"상품 {product_id} 없음"}
        if p["stock"].get(option, 0) <= 0:
            return {"error": f"{option} 품절", "available": [o for o, n in p["stock"].items() if n > 0]}
        draft_id = f"D-{product_id}-{option}".replace("/", "_")
        d["drafts"][draft_id] = {"user_id": user_id, "product_id": product_id, "option": option, "price": p["price"], "status": "draft"}
    return {"draft_id": draft_id, "product": p["name"], "option": option, "price": p["price"]}


@mcp.tool()
def checkout(draft_id: str, idempotency_key: str) -> dict:
    """임시 주문을 실제로 결제한다. 같은 idempotency_key로 다시 부르면 기존 결과를 돌려준다."""
    with store("member") as m:
        pass
    with store("payment", write=True) as d:
        for oid, o in d["orders"].items():
            if o.get("idempotency_key") == idempotency_key:
                return {"order_id": oid, "status": o["status"], "replayed": True, "address": o.get("address")}
        draft = d["drafts"].get(draft_id)
        if not draft:
            return {"error": f"임시 주문 {draft_id} 없음"}
        if d["card_declined"]:
            return {"error": "card_declined", "message": "카드 승인이 거절되었습니다."}
        user = m["users"][draft["user_id"]]
        address = user["checkout_address"] or user["default_address"]
        oid = f"O-{d['next_order_no']}"
        d["next_order_no"] += 1
        d["products"][draft["product_id"]]["stock"][draft["option"]] -= 1
        d["orders"][oid] = {**draft, "status": "paid", "idempotency_key": idempotency_key, "address": address, "eta": "내일 도착 예정"}
        draft["status"] = "paid"
    return {"order_id": oid, "status": "paid", "address": address, "price": draft["price"], "eta": "내일 도착 예정"}


@mcp.tool()
def list_orders(user_id: str) -> dict:
    """회원의 주문 목록을 돌려준다."""
    with store("payment") as d:
        return {"orders": [{"order_id": k, **v} for k, v in d["orders"].items() if v["user_id"] == user_id]}


@mcp.tool()
def quote_return(order_id: str, mode: str, new_option: str | None = None) -> dict:
    """반품(mode=refund) 또는 교환(mode=exchange) 시 고객이 돌려받거나 내는 금액을 계산한다."""
    with store("payment") as d:
        o = d["orders"].get(order_id)
        if not o:
            return {"error": f"주문 {order_id} 없음"}
        p = d["products"][o["product_id"]]
        if mode == "refund":
            # 쿠폰은 소멸, 적립금은 재적립, 카드 결제분 환불, 단순변심 반품배송비 3,000원 차감
            return {"mode": "refund", "customer_pays_now": 0, "card_refund": o["card"] - 3000, "points_restored": o["points"],
                    "coupon_lost": o["coupon"], "return_fee_deducted": 3000, "customer_keeps_item": False}
        if mode == "exchange":
            opt = new_option or o["option"].split("/")[0] + "/L"
            in_stock = p["stock"].get(opt, 0) > 0
            # 사이즈 교환: 왕복배송비 6,000원 중 GOLD 등급 50% 지원, 쿠폰 혜택 유지
            return {"mode": "exchange", "new_option": opt, "in_stock": in_stock, "customer_pays_now": 3000,
                    "card_refund": 0, "coupon_kept": o["coupon"], "customer_keeps_item": True}
        return {"error": "mode는 refund 또는 exchange"}


if __name__ == "__main__":
    mcp.run()
