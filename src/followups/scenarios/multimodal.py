"""4단계 멀티모달 시나리오 (docs/structured-multiturn.md 6장 턴 1~3).

uv run python -m followups.run mapo_report       # 차트 + DB + 보고서 (출처 칩 → 보고서 → 구별 refine)
uv run python -m followups.run chart_only        # 첨부만: 판독 → '아까 보낸 차트'로 보고서
"""
from __future__ import annotations

CHART = "data/attachments/mapo_sales_2026h1.png"  # uv run python -m followups.make_chart 로 생성


def _tasks(turn: dict, intent: str) -> list[dict]:
    return [t for t in turn["tasks"] if t["intent"] == intent]


def _labels(turn: dict) -> list[str]:
    return [c["label"] for c in turn["chips"]]


def _expect_mapo_report(log: dict):
    t1, t2, t3 = log["turns"][:3]
    att = _tasks(t1, "attachment_analysis")
    rep1 = _tasks(t1, "report_compose")
    rep2, rep3 = _tasks(t2, "report_compose"), _tasks(t3, "report_compose")
    q3 = _tasks(t3, "data_query")
    checks = {
        "턴1 첨부 해석 done": any(t["status"] == "done" for t in att),
        "턴1 출처 칩(상권 DB(월별))": "상권 DB(월별)" in _labels(t1),
        "턴1 보고서는 대기": bool(rep1) and rep1[0]["status"] != "done",
        "턴1 답에 4월 급감": "4월" in t1["answer"],
        "턴2 보고서 done": bool(rep2) and rep2[0]["status"] == "done",
        "턴2 보고서에 차트 수치(870)": "870" in t2["answer"],
        "턴3 data_query group_by에 구": any("구" in (t["slots"].get("group_by") or []) for t in q3),
        "턴3 보고서 재작성": bool(rep3) and rep3[0]["status"] == "done" and t3["answer"] != t2["answer"],
    }
    return all(checks.values()), f"checks={checks}"


def _expect_chart_only(log: dict):
    t1, t2 = log["turns"][:2]
    att = _tasks(t1, "attachment_analysis")
    rep = _tasks(t2, "report_compose")
    checks = {
        "턴1 첨부 해석 done": any(t["status"] == "done" for t in att),
        "턴1 답에 4월·870": "4월" in t1["answer"] and "870" in t1["answer"],
        "턴2 보고서 done": bool(rep) and rep[-1]["status"] == "done",
        "턴2 보고서가 차트 기준(870)": "870" in t2["answer"],
        "턴2 새 이미지 판독 없음(첨부 작업 1개)": len(_tasks(t2, "attachment_analysis")) == 1,
    }
    return all(checks.values()), f"checks={checks}"


SCENARIOS: dict[str, dict] = {
    "mapo_report": {
        "desc": "차트 첨부 + DB 조회 + 보고서 합성 → 출처 칩 → 같은 턴 보고서 → '구별로도' refine 후 재합성",
        "steps": [
            ("attach", {"text": "4월에 왜 떨어졌는지, 같은 기간 업종별 점포 수 DB에서 뽑아서 3분기 보고서 톤으로 정리해줘",
                        "image": CHART}),
            ("chip", "상권 DB(월별)"),
            ("say", "구별로도 나눠줘"),
        ],
        "expect": _expect_mapo_report,
    },
    "chart_only": {
        "desc": "첨부만: 입구에서 한 번 판독 → 다음 턴 '아까 보낸 차트'는 last_attachment로 (이미지 재전송 없음)",
        "steps": [
            ("attach", {"text": "이 차트 요약해줘", "image": CHART}),
            ("say", "아까 보낸 차트 기준으로 보고서 톤으로 정리해줘"),
        ],
        "expect": _expect_chart_only,
    },
}
