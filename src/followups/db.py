"""목(mock) 백엔드 저장소. MCP 서버 프로세스끼리 공유하는 JSON 파일 + 파일 잠금."""
from __future__ import annotations

import copy
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FOLLOWUPS_DATA_DIR", ROOT / "data"))

SEED: dict[str, dict] = {
    "member": {
        "users": {
            "u1": {
                "name": "김민지",
                "grade": "GOLD",
                "default_address": "서울시 강남구 테헤란로 1, 1001호",
                "checkout_address": None,  # 다음 결제에만 쓰는 배송지 (이번 주문 배송지)
            }
        },
        "address_change_open": True,  # 출고 전이면 변경 가능
    },
    "payment": {
        "products": {
            "P-HOODIE-01": {
                "name": "오버핏 기모 후드티",
                "price": 39000,
                "stock": {"블랙/L": 3, "블랙/M": 0, "그레이/L": 5, "그레이/M": 2},
            },
            "P-PADDING-07": {
                "name": "경량 덕다운 패딩",
                "price": 89000,
                "stock": {"블랙/L": 4, "네이비/M": 1},
            },
        },
        "orders": {
            "O-1001": {
                "user_id": "u1",
                "product_id": "P-HOODIE-01",
                "option": "블랙/M",
                "price": 39000,
                "coupon": 5000,
                "points": 2000,
                "card": 32000,
                "status": "delivered",
                "delivered_days_ago": 23,
                "ordered_at": "2026-10-20",
            }
        },
        "drafts": {},
        "card_declined": False,  # 시나리오용: True면 결제 실패
        "next_order_no": 2001,
    },
}


def _path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{name}.json"


def reset(overrides: dict | None = None) -> None:
    for name, seed in SEED.items():
        data = copy.deepcopy(seed)
        for k, v in (overrides or {}).get(name, {}).items():
            data[k] = v
        _path(name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


@contextmanager
def store(name: str, write: bool = False):
    p = _path(name)
    if not p.exists():
        reset()
    with open(p, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        data = json.load(f)
        yield data
        if write:
            f.seek(0)
            f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)
