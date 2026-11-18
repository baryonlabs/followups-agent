"""5단계 시나리오: 장기 사용자 메모리 (동의 칩을 거친 뒤에만 프로필에 기록).

run.py가 시작할 때 db.reset()을 하므로 프로필 저장소도 비어 있는 상태에서 시작한다.
"""
from __future__ import annotations

from followups import db

UTTER = "앞으로 숫자는 반올림하지 말고 소수점 둘째 자리까지 보여줘"


def _prefs(user_id: str = "u1") -> dict:
    with db.store("profile") as d:
        return dict(((d.get("users") or {}).get(user_id) or {}).get("prefs") or {})


def _pref_tasks(log: dict) -> list[dict]:
    return [t for t in (log["turns"][-1]["tasks"] if log["turns"] else []) if t.get("intent") == "set_preference"]


def _expect_saved(log: dict):
    prefs = _prefs()
    row = prefs.get("decimal_places") or {}
    asked = any(c["label"] == "앞으로도 적용" for c in log["turns"][0]["chips"]) if log["turns"] else False
    tasks = _pref_tasks(log)
    ok = (asked and row.get("value") == "2" and bool(row.get("confirmed_at"))
          and row.get("source_thread") == log["session_id"] and any(t["status"] == "done" for t in tasks))
    return ok, f"asked={asked} prefs={ {k: v.get('value') for k, v in prefs.items()} } tasks={[(t['id'], t['status']) for t in tasks]}"


def _expect_nothing_saved(log: dict):
    prefs = _prefs()
    tasks = _pref_tasks(log)
    ok = not prefs and any(t["status"] == "cancelled" for t in tasks)
    return ok, f"prefs={list(prefs)} tasks={[(t['id'], t['status']) for t in tasks]}"


SCENARIOS: dict[str, dict] = {
    "memory_remember": {
        "desc": "영구 선호 발화 → 동의 칩(앞으로도 적용) → memory_writer가 출처와 함께 프로필 기록",
        "steps": [
            ("say", UTTER),
            ("chip", "앞으로도 적용"),
        ],
        "expect": _expect_saved,
    },
    "memory_once": {
        "desc": "영구 선호 발화 → '이번만' → 작업 취소, 프로필에 아무것도 남지 않음",
        "steps": [
            ("say", UTTER),
            ("chip", "이번만"),
        ],
        "expect": _expect_nothing_saved,
    },
}
