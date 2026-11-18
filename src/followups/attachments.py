"""멀티모달: 첨부 리더(입구) · 첨부 해석 worker · 보고서 작성 worker.

계약
- has_attachment(state) -> bool                     # 이번 턴 마지막 HumanMessage에 아직 읽지 않은 이미지가 있나
- build_attachment_reader() -> async node(state, config) -> {"attachments": {id: {...}}, "entity_memory": {"last_attachment": {...}}}
- build_attachment_worker(box) -> async worker(arg, config)   # route="attachment"
- build_report_writer(box) -> async worker(arg, config)       # route="report", arg["inputs"]에 앞 작업 결과
- EXTRACTOR_HINT

설계 (docs/rfp-structured-multiturn.md 5.1)
- 이미지는 입구에서 한 번만 VLM으로 '사실 JSON'으로 바꾼다. 상태에는 사실과 object_key만 두고 원본(base64)은 넣지 않는다.
- 이후 턴은 이미지를 다시 모델에 보내지 않는다. '아까 보낸 차트'는 entity_memory.last_attachment로 푼다.
- LLM 호출(read_image_facts, write_report)은 모듈 함수라 테스트에서 monkeypatch로 바꾼다 (빌더 인자로도 주입 가능).
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from followups.db import DATA_DIR
from followups.state import GUIDELINES

EXTRACTOR_HINT = (
    "- 이미지가 첨부된 발화면 attachment_analysis 의도를 만들고 slots.attachment_id에 '첨부 사실'의 id를 넣는다.\n"
    "- '아까 보낸 차트', '그 그래프'처럼 이전 첨부를 가리키면 최근 엔티티의 last_attachment.attachment_id를 쓴다.\n"
    "- '보고서로 정리', '보고서 톤', '보고서 양식으로'는 report_compose 의도. slots.tone에 말한 문체를 그대로 (예: '3분기 보고서').\n"
    "  첨부를 근거로 하는 보고서면 report_compose의 slots.attachment_id에도 그 첨부 id를 넣는다.\n"
    "- 첨부 사실의 region·period는 같은 발화의 data_query 슬롯(region, period)으로 옮겨도 된다 (사용자가 다른 값을 말하지 않았을 때)."
)

# 원본 이미지 보관소. 서비스에서는 MinIO(오브젝트 스토리지) 버킷이고 object_key가 그 키다.
# 여기서는 DATA_DIR/attachments/uploads/<sha256 앞 16자>.<확장자> 로컬 파일이 그 대역이다.
OBJECT_DIR = DATA_DIR / "attachments" / "uploads"
UNIT_CHOICES = ["백만원", "천원", "원"]


# ---------- 사실 스키마 (VLM 구조화 출력) ----------
class Point(BaseModel):
    x: str = Field(description="x축 값. 월이면 'YYYY-MM' (예: 2026-04), 아니면 축 라벨 그대로")
    y: float | None = Field(None, description="y값. 읽을 수 없으면 null (추정 금지)")


class Series(BaseModel):
    label: str = Field(description="계열 이름. 예: 매출")
    values: list[Point] = Field(default_factory=list)


class Anomaly(BaseModel):
    x: str = Field(description="이상치가 있는 x값. 예: 2026-04")
    note: str = Field(description="무엇이 특이한지 짧게. 예: 전월 대비 급감")


class AttachmentFacts(BaseModel):
    kind: Literal["chart", "table", "document", "photo", "other"]
    title: str | None = Field(None, description="이미지에 적힌 제목 그대로")
    region: str | None = Field(None, description="지역. 예: 마포구")
    period: str | None = Field(None, description="기간 'YYYY-MM~YYYY-MM'. 예: 2026-01~2026-06")
    y_unit: str | None = Field(None, description="y축(값) 단위. 예: 백만원. 이미지에 없거나 못 읽으면 null")
    series: list[Series] = Field(default_factory=list, description="차트·표의 수치. 차트가 아니면 빈 목록")
    anomalies: list[Anomaly] = Field(default_factory=list)
    text: str | None = Field(None, description="문서·사진이면 핵심 내용 1~2문장")
    unreadable: list[str] = Field(default_factory=list, description="읽을 수 없었던 항목. 예: ['y_unit', '2026-03 값']")


READER_PROMPT = (
    "너는 공공 데이터 챗봇의 첨부 이미지 판독기다. 이미지에 보이는 사실만 구조화한다.\n"
    "- 막대·선 위의 숫자 라벨이 있으면 그 값을 그대로 쓴다. 라벨이 없어 눈금으로만 짐작해야 하면 y=null, unreadable에 적는다.\n"
    "- 단위·기간·지역이 이미지에 없으면 null. 사용자 발화에서 추측하지 않는다.\n"
    "- 이상치(급감·급증)는 수치로 확인되는 것만 anomalies에.\n"
    f"- [attachment.units] {GUIDELINES['attachment.units']['text']}"
)

_vlm = None


_ocr_llm = None

OCR_PROMPT = (
    "너는 공공 데이터 챗봇의 첨부 판독기다. 아래는 이미지를 Document Parse(OCR)로 읽은 결과다. 여기 적힌 사실만 구조화한다.\n"
    "- 차트 표(Chart Type 다음 표)의 숫자를 그대로 series에 옮긴다. 'item_01' 같은 자동 라벨이면 제목에서 계열 이름을 짓는다 (예: 매출).\n"
    "- 'segment_' 같은 자동 라벨만 있고 제목과 무관한 작은 표는 범례·장식 오인식이니 버린다.\n"
    "- x가 월이면 제목의 연도를 붙여 'YYYY-MM'으로 (예: 2026년 1월 → 2026-01).\n"
    "- 단위·기간·지역은 OCR 텍스트에 있을 때만. 없으면 null. 사용자 발화에서 추측하지 않는다.\n"
    "- 이상치(급감·급증)는 수치로 확인되는 것만 anomalies에.\n"
    f"- [attachment.units] {GUIDELINES['attachment.units']['text']}"
)


def _decode_data_url(url: str) -> tuple[bytes, str]:
    head, _, b64 = url.partition(",")
    mime = head.split(";")[0].removeprefix("data:") or "image/png"
    return base64.b64decode(b64), mime


async def _document_parse(image_url: str) -> str:
    """사내 Upstage Document Parse로 이미지 OCR (차트는 표로 복원된다)."""
    import httpx

    from followups.llm import DOCPARSE_URL

    if image_url.startswith("data:"):
        data, mime = _decode_data_url(image_url)
    else:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(image_url)
            r.raise_for_status()
            data, mime = r.content, r.headers.get("content-type", "image/png")
    ext = mimetypes.guess_extension(mime) or ".png"
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(DOCPARSE_URL, files={"document": (f"upload{ext}", data, mime)},
                         data={"output_formats": '["text"]', "ocr": "force"})
        r.raise_for_status()
    return (r.json().get("content") or {}).get("text") or ""


async def read_image_facts_ocr(image_url: str, text: str, config) -> dict:
    """VLM이 없는 환경: Document Parse OCR → 텍스트 LLM이 사실 JSON으로 구조화."""
    global _ocr_llm
    if _ocr_llm is None:
        from followups.llm import MODEL_MAIN, llm

        _ocr_llm = llm(MODEL_MAIN).with_structured_output(AttachmentFacts, method="json_schema")
    ocr = await _document_parse(image_url)
    res = await _ocr_llm.ainvoke([
        SystemMessage(content=OCR_PROMPT),
        HumanMessage(content=f"사용자 발화(참고용, 사실 근거 아님): {text or '(없음)'}\n\nOCR 결과:\n{ocr[:6000]}"),
    ], config={**(config or {}), "run_name": "attachment_reader.ocr"})
    if res is None:
        return AttachmentFacts(kind="other", text=ocr[:200] or None, unreadable=["structure"]).model_dump()
    return res.model_dump()


async def read_image_facts(image_url: str, text: str, config) -> dict:
    """VLM 1회 호출: 이미지 → 사실 JSON. 테스트에서는 monkeypatch로 바꾼다.
    ATTACHMENT_READER=docparse면 OCR 경로를 쓴다 (OpenRouter가 아니면 기본값)."""
    from followups.llm import ATTACHMENT_READER

    if ATTACHMENT_READER == "docparse":
        return await read_image_facts_ocr(image_url, text, config)
    global _vlm
    if _vlm is None:
        from followups.llm import MODEL_VISION, VISION_BASE_URL, llm  # 지연 생성: 키 없이도 모듈 import·테스트가 되게

        _vlm = llm(MODEL_VISION, base_url=VISION_BASE_URL).with_structured_output(AttachmentFacts, method="function_calling")
    res: AttachmentFacts = await _vlm.ainvoke([
        SystemMessage(content=READER_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": f"사용자 발화(참고용, 사실 근거 아님): {text or '(없음)'}"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]),
    ], config={**(config or {}), "run_name": "attachment_reader.vlm"})
    return res.model_dump()


# ---------- 입구: 첨부 판독 ----------
def _image_urls(content) -> list[str]:
    """메시지 content에서 이미지 URL(data URI 포함)만 뽑는다. OpenAI image_url 형식과 LangChain image 블록 둘 다."""
    if isinstance(content, str):
        return []
    urls = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "image_url":
            iu = b.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
        elif b.get("type") == "image":
            url = b.get("url")
            if not url and (b.get("base64") or b.get("data")):
                url = f"data:{b.get('mime_type', 'image/png')};base64,{b.get('base64') or b.get('data')}"
        else:
            continue
        if url:
            urls.append(url)
    return urls


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()


def _last_human(state):
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m
    return None


def attachment_id_of(url: str) -> str:
    """같은 이미지는 같은 id (내용 해시). 다시 보내도 다시 읽지 않는다."""
    return "att_" + hashlib.sha256(url.encode()).hexdigest()[:10]


def _new_images(state) -> list[tuple[str, str]]:
    m = _last_human(state)
    if m is None:
        return []
    seen = state.get("attachments") or {}
    return [(attachment_id_of(u), u) for u in _image_urls(m.content) if attachment_id_of(u) not in seen]


def has_attachment(state) -> bool:
    return bool(_new_images(state))


def _store_object(url: str) -> str:
    """원본을 오브젝트 스토리지(여기서는 로컬 파일)에 두고 키만 돌려준다."""
    m = re.match(r"data:([^;,]+)?(;base64)?,(.*)", url, re.S)
    if not m:
        return url  # 외부 URL은 그 자체가 참조 키 (내려받지 않음)
    mime, data = m.group(1) or "image/png", m.group(3)
    raw = base64.b64decode(data) if m.group(2) else data.encode()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    ext = mimetypes.guess_extension(mime) or ".bin"
    OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    path = OBJECT_DIR / f"{sha}{ext}"
    if not path.exists():
        path.write_bytes(raw)
    try:
        return path.relative_to(DATA_DIR).as_posix()
    except ValueError:  # 테스트에서 OBJECT_DIR을 바꾼 경우
        return path.as_posix()


def build_attachment_reader(read=None):
    async def attachment_reader(state, config) -> dict:
        reader = read or read_image_facts
        msg = _last_human(state)
        text = _text_of(msg.content) if msg else ""
        turn = state.get("turn", 0)
        out: dict[str, dict] = {}
        last = None
        for att_id, url in _new_images(state):
            try:
                facts = await reader(url, text, config)
            except Exception as e:  # VLM 실패: 사실 없이 기록하고 worker가 실패로 알린다
                facts = {"kind": "other", "unreadable": ["전체"], "error": f"{type(e).__name__}: {e}"[:200]}
            facts = {k: v for k, v in facts.items() if v not in (None, [], "")} | {"kind": facts.get("kind", "other")}
            out[att_id] = {"kind": facts["kind"], "facts": facts, "object_key": _store_object(url), "turn": turn}
            last = {"attachment_id": att_id, "kind": facts["kind"], "title": facts.get("title")}
        if not out:
            return {}
        return {"attachments": out, "entity_memory": {"last_attachment": last}}

    return attachment_reader


# ---------- 첨부 해석 worker ----------
def _fmt(v: float | None) -> str:
    if v is None:
        return "?"
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.1f}"


def _month(x: str) -> str:
    m = re.fullmatch(r"\d{4}-(\d{2})", x or "")
    return f"{int(m.group(1))}월" if m else x


def series_changes(facts: dict) -> list[dict]:
    """계열별 전 구간 대비 증감 (규칙 계산. 보고서 LLM이 직접 계산하지 않게 미리 준다)."""
    out = []
    for s in facts.get("series") or []:
        pts = [p for p in s.get("values") or [] if p.get("y") is not None]
        for a, b in zip(pts, pts[1:]):
            d = b["y"] - a["y"]
            pct = round(d / a["y"] * 100, 1) if a["y"] else None
            out.append({"series": s.get("label"), "from": a["x"], "to": b["x"], "from_y": a["y"], "to_y": b["y"],
                        "diff": d, "pct": pct})
    return out


def _unit_unreadable(facts: dict) -> bool:
    if facts.get("kind") not in ("chart", "table"):
        return False
    return not facts.get("y_unit") or "y_unit" in (facts.get("unreadable") or [])


def summarize_facts(facts: dict) -> str:
    """사실 → 짧은 한국어 소견 (LLM 없이)."""
    if facts.get("kind") not in ("chart", "table"):
        return facts.get("text") or facts.get("title") or "첨부 내용을 읽었습니다."
    unit = facts.get("y_unit") or "단위 미상"
    head = " ".join(x for x in (facts.get("region"), facts.get("title") and f"'{facts['title']}'") if x) or "첨부 차트"
    period = f" {facts['period']}" if facts.get("period") else ""
    parts = [f"{head}{period} (단위: {unit})."]
    changes = series_changes(facts)
    if changes:
        drop = min(changes, key=lambda c: c["diff"])
        rise = max(changes, key=lambda c: c["diff"])
        if drop["diff"] < 0:
            pct = f"({drop['pct']:+.1f}%)" if drop["pct"] is not None else ""
            parts.append(f"{_month(drop['to'])} {_fmt(drop['to_y'])}로 전월({_fmt(drop['from_y'])}) 대비 "
                         f"{_fmt(-drop['diff'])} 감소{pct}가 가장 큼.")
        if rise["diff"] > 0 and rise is not drop:
            parts.append(f"가장 큰 증가는 {_month(rise['to'])} {_fmt(rise['to_y'])} (+{_fmt(rise['diff'])}).")
    for a in (facts.get("anomalies") or [])[:2]:
        parts.append(f"이상치: {_month(a.get('x', ''))} {a.get('note', '')}.")
    unread = [u for u in facts.get("unreadable") or [] if u != "y_unit"]
    if unread:
        parts.append(f"읽지 못한 항목: {', '.join(unread)}.")
    return " ".join(parts)


def build_attachment_worker(box):
    async def attachment_worker(arg, config) -> dict:
        t = {**arg["task"], "turn": arg["turn"]}
        att_id = t["slots"].get("attachment_id")
        rec = (arg.get("attachments") or {}).get(att_id)
        if not rec:
            t.update(status="failed", error="attachment_not_found", result={"summary": "해당 첨부를 찾을 수 없습니다."})
            return {"task_stack": [t]}
        facts = dict(rec.get("facts") or {})
        if facts.get("error"):
            t.update(status="failed", error="attachment_unreadable",
                     result={"summary": "첨부 이미지를 판독하지 못했습니다.", "facts": facts})
            return {"task_stack": [t]}

        update: dict = {}
        if _unit_unreadable(facts):
            unit = t["slots"].get("y_unit")
            if not unit:
                # 단위를 못 읽으면 추정하지 않고 칩으로 묻는다 (attachment.units). compute_missing은 카탈로그 필수 슬롯만
                # 남기므로, 칩(fill_slots y_unit)을 누르면 missing=[] → pending으로 이 worker가 다시 돈다.
                t.update(status="blocked", missing=["y_unit"],
                         result={"facts": facts, "summary": "차트의 값 단위를 읽지 못했습니다.",
                                 "choices": {"y_unit": [{"label": f"단위: {u}", "value": u} for u in UNIT_CHOICES]}})
                return {"task_stack": [t]}
            # 사용자가 고른 단위를 사실에 반영해 다음 턴(추출기·보고서)도 같은 값을 쓰게 한다
            facts = {**facts, "y_unit": unit, "y_unit_source": "user",
                     "unreadable": [u for u in facts.get("unreadable") or [] if u != "y_unit"]}
            update["attachments"] = {att_id: {**rec, "facts": facts}}

        t.update(status="done", missing=[], error=None, result={"facts": facts, "summary": summarize_facts(facts)})
        return {"task_stack": [t], **update}

    return attachment_worker


# ---------- 보고서 작성 worker ----------
REPORT_PROMPT = (
    "너는 공공기관 내부 보고서 작성기다. 아래 [자료]만 근거로 한국어 보고서를 쓴다.\n"
    "- [자료]에 없는 수치·날짜·고유명사는 쓰지 않는다. 증감은 [자료]의 'changes' 값을 그대로 쓴다.\n"
    "- 원인은 [자료]로 뒷받침되는 범위에서만 '~로 추정됨'으로 쓰고, 확인할 수 없으면 '자료로 확인 불가'라고 적는다.\n"
    "- 빠진 자료(조회 실패 등)가 있으면 마지막에 '한계'로 밝힌다.\n"
    "- 구성: 제목 1줄 → □ 현황(핵심 수치) → □ 분석 → □ 시사점. 각 항목은 '○ ' 개조식.\n"
    "- 이모지·마크다운 표 금지."
)


async def write_report(messages: list, config) -> str:
    """보고서 LLM 1회 호출. 테스트에서는 monkeypatch로 바꾼다."""
    from followups.llm import MODEL_MAIN, llm

    msg = await llm(MODEL_MAIN).ainvoke(messages, config={**(config or {}), "run_name": "report_writer.llm"})
    return msg.content if isinstance(msg.content, str) else _text_of(msg.content)


def collect_materials(arg) -> tuple[list[dict], list[str], list[str]]:
    """보고서 근거 자료 = 의존 작업(inputs) 결과 + 이 작업이 직접 가리킨 첨부. (자료, 출처 작업 id, 빠진 것)"""
    materials, sources, gaps = [], [], []
    seen_att = set()
    for tid, inp in sorted((arg.get("inputs") or {}).items()):
        res = inp.get("result") or {}
        if inp.get("status") != "done":
            gaps.append(f"{tid}({inp.get('intent')}) {inp.get('status')}")
            continue
        if inp.get("intent") == "data_query":
            rows = res.get("rows") or []
            materials.append({"type": "dataset", "task": tid, "source": res.get("source") or inp["slots"].get("source"),
                              "conditions": {k: inp["slots"].get(k) for k in ("metric", "period", "region", "group_by")
                                             if inp["slots"].get(k)},
                              "columns": res.get("columns"), "rows": rows[:50], "row_count": res.get("row_count", len(rows)),
                              **({"truncated": True} if len(rows) > 50 else {})})
            sources.append(tid)
        elif inp.get("intent") == "attachment_analysis" and res.get("facts"):
            facts = res["facts"]
            materials.append({"type": "attachment", "task": tid, "facts": facts, "changes": series_changes(facts)})
            sources.append(tid)
            seen_att.add(inp["slots"].get("attachment_id"))
    att_id = arg["task"]["slots"].get("attachment_id")
    rec = (arg.get("attachments") or {}).get(att_id)
    if att_id and att_id not in seen_att and rec and not (rec.get("facts") or {}).get("error"):
        materials.append({"type": "attachment", "attachment_id": att_id, "facts": rec["facts"],
                          "changes": series_changes(rec["facts"])})
        sources.append(att_id)
    return materials, sources, gaps


def build_report_messages(task: dict, materials: list[dict], gaps: list[str], guidelines_text: str) -> list:
    tone = task["slots"].get("tone") or "내부 보고서"
    sys = SystemMessage(content=f"{REPORT_PROMPT}\n문체·양식: {tone}\n가이드라인:\n{guidelines_text or '- (없음)'}")
    body = json.dumps(materials, ensure_ascii=False, default=str)
    gap_line = f"\n[빠진 자료] {'; '.join(gaps)}" if gaps else ""
    return [sys, HumanMessage(content=f"[자료]\n{body}{gap_line}\n\n위 자료로 '{tone}' 보고서를 작성하라.")]


_NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s*(?:월|분기|년|일|개월|주|차|위|호|번째))?")


def _nums(text: str) -> list[tuple[float, bool]]:
    out = []
    for m in _NUM.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        out.append((v, bool(m.group(2))))
    return out


_DERIVED_KEYS = {"diff", "pct", "from_y", "to_y", "row_count"}  # 이미 계산된 값: 그대로 허용하되 다시 조합하지 않음


def _walk_numbers(obj, raw: set, derived: set, key: str = "") -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        (derived if key in _DERIVED_KEYS else raw).add(abs(float(obj)))
    elif isinstance(obj, str):
        if re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", obj.strip()):
            raw.add(abs(float(obj.replace(",", ""))))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(v, raw, derived, k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, raw, derived, key)


def unsupported_numbers(report: str, materials: list[dict], extra: str = "") -> list[str]:
    """보고서에 나온 수치 중 자료로 설명되지 않는 것 (근거 없는 수치 감시).

    허용: 자료의 수치 값, 미리 계산해 준 증감(changes), 자료 수치 두 개의 차이·비율·증감률.
    """
    raw, derived = set(), set()
    _walk_numbers(materials, raw, derived)
    raw |= {v for v, _ in _nums(extra)}
    allowed = raw | derived
    vals = sorted(raw)[:300]
    for a in vals:
        for b in vals:
            if a != b:
                allowed.add(abs(a - b))
                if b:
                    allowed.update({round(abs(a / b * 100 - 100), 1), round(abs(a / b * 100 - 100)), round(a / b * 100, 1)})
    bad = []
    for v, is_label in _nums(report):
        if is_label or (v.is_integer() and 1900 <= v <= 2100):  # '4월', '3분기', 연도는 수치 주장이 아님
            continue
        if not any(abs(v - a) < 0.051 for a in allowed):
            bad.append(_fmt(v))
    return sorted(set(bad))


def build_report_writer(box, write=None):
    async def report_writer(arg, config) -> dict:
        t = {**arg["task"], "turn": arg["turn"]}
        materials, sources, gaps = collect_materials(arg)
        if not materials:
            # 근거 자료가 하나도 없으면 LLM을 부르지 않는다 (지어낼 위험). blocked로 두면 missing이 없어 planner가
            # 곧바로 다시 돌리므로, done + '무엇이 빠졌는지' 안내로 끝낸다. 자료가 생기면 refine/새 요청으로 다시 돈다.
            why = f" (앞 작업 상태: {'; '.join(gaps)})" if gaps else ""
            t.update(status="done", error=None, result={
                "summary": "보고서에 쓸 근거 자료가 없어 작성하지 않았습니다." + why
                           + " 차트를 첨부하거나 데이터 조회를 먼저 요청해 주세요.",
                "sources": [], "empty": True})
            return {"task_stack": [t]}
        messages = build_report_messages(t, materials, gaps, arg.get("guidelines_text", ""))
        text = (await (write or write_report)(messages, config)).strip()
        bad = unsupported_numbers(text, materials, t["slots"].get("tone") or "")
        if bad:
            text += f"\n※ 검증 필요: 입력 자료에서 확인되지 않는 수치 {', '.join(bad)}"
        t.update(status="done", error=None, result={"summary": text, "sources": sources,
                                                    **({"unverified_numbers": bad} if bad else {}),
                                                    **({"gaps": gaps} if gaps else {})})
        return {"task_stack": [t]}

    return report_writer
