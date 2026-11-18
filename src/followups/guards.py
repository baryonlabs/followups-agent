"""정책 가드 (샌드위치): 입력 가드 · 정책 답변 · 출력 가드.

계약
- build_input_guard()  -> async node(state, config) -> {"guard": {"blocked", "reason", "rule", "alternatives": [label]}, "policy_flags"?, "messages"?}
- build_policy_reply() -> async node(state, config) -> {"answer", "messages": [AIMessage], "followups": [FollowUp type="policy"]}
- build_output_guard() -> async node(state, config) -> {"answer"?, "messages"?, "policy_flags"?}

설계
- 규칙은 아래 데이터 표(BLOCK_RULES, MASK_RULES)로만 정의한다. 감사 시 표만 보면 무엇을 막는지 알 수 있다.
- 전부 정규식이라 LLM 호출·네트워크가 없고 결정적이다 (턴 지연 거의 0).
- 운영에서는 NeMo Guardrails · Llama Guard 같은 분류기를 `classifier` 슬롯에 꽂는다 (기본 None).
  규칙이 먼저 돌고, 규칙에 걸리지 않은 입력만 분류기에 묻는다.

판단
- 사용자 본인의 PII(본인 전화번호·주민번호 등)는 턴을 막지 않는다. 같은 메시지 id로 마스킹한 메시지를 돌려줘
  add_messages가 원문을 덮어쓰게 하므로, 이후 LLM·체크포인트·다음 턴 문맥에는 가려진 값만 남는다. (action="mask")
  단, 그래프 입력 자체(첫 트레이스 span)는 이 노드 이전이라 트레이스 쪽 마스킹은 별도로 필요하다 (Langfuse mask).
- 타인 PII 조회 요청("대표자 주민번호 명단")은 차단하고 대안 칩을 준다. 작업은 만들지 않는다. (action="block")
- 칩 클릭 턴(payload.op가 구조화 명령)은 서버가 만든 선택지라 차단 규칙을 건너뛴다. 마스킹은 그대로 적용한다 (비용 0, 해 없음).
  op="say" 칩은 run.py/화면이 텍스트를 새 사용자 발화로 보내므로(payload=None) 일반 입력과 똑같이 검사된다.
  칩 문구는 LLM이 만든 것(next_steps)일 수도 있으므로 이것이 맞다.
- policy_flags에는 원문 값을 남기지 않는다 (감사 기록이 또 다른 PII 저장소가 되면 안 됨). 규칙 id와 건수만.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage

EXTRACTOR_HINT = "- 발화 속 '*'로 가려진 값은 개인정보 마스킹 결과다. 원래 값을 추측해 슬롯에 채우지 않는다."

# graph.PAYLOAD_OPS와 같은 값 (graph가 이 모듈을 import하므로 순환을 피해 따로 둔다)
TRUSTED_OPS = {"fill_slots", "skip_task", "retry", "refine", "remember"}

_I = re.IGNORECASE


# ---------- 규칙 표 ----------
@dataclass(frozen=True)
class BlockRule:
    id: str
    patterns: tuple[str, ...]          # 모두 맞아야 걸린다 (AND). 각 패턴 안에서는 | 로 OR
    message: str                       # 사용자에게 보일 이유 (한두 문장)
    alternatives: tuple[str, ...] = ()  # 대안 칩 문구 (누르면 새 발화로 보내짐 → 다시 가드를 통과해야 함)
    _compiled: tuple[re.Pattern, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_compiled", tuple(re.compile(p, _I) for p in self.patterns))

    def match(self, text: str) -> bool:
        return all(p.search(text) for p in self._compiled)


_PII_KIND = r"주민\s*(등록)?\s*번호|주민번호|전화\s*번호|휴대\s*(폰|전화)|핸드폰|연락처|계좌\s*(번호)?|카드\s*번호|생년월일|이메일|여권\s*번호|운전\s*면허\s*번호"
_OTHERS = r"대표자|명단|목록|리스트|전체|모든|모두|고객(?!\s*센터)|회원|직원|사람들|타인|남의|다른\s*사람|사업자들|점주"
_RETRIEVE = r"뽑|조회|알려|보여|추출|출력|내려|다운|정리|찾아|모아|엑셀|csv"

BLOCK_RULES: tuple[BlockRule, ...] = (
    BlockRule(
        id="pii.others",
        patterns=(_PII_KIND, _OTHERS, _RETRIEVE),
        message="다른 사람의 개인 식별 정보(주민등록번호·전화번호·계좌 등)는 조회하거나 명단으로 내보낼 수 없습니다. "
                "개인을 알아볼 수 없는 집계나 목록은 도와드릴 수 있습니다.",
        alternatives=("대표자 수만 집계하기", "업종별 점포 목록 보기"),
    ),
    BlockRule(
        id="inject.override",
        patterns=(r"(이전|앞|위|기존|모든|원래)\s*(의|에서\s*)?\s*(지시|지침|명령|규칙|설정|프롬프트)\S*\s*(은|는|을|를|다|모두|전부)?\s*(무시|잊|따르지\s*마)"
                  r"|ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)"
                  r"|disregard\s+(all\s+|the\s+)?(previous|prior|above)",),
        message="이전 지시나 운영 규칙을 바꾸는 요청은 처리할 수 없습니다. 원하시는 업무를 말씀해 주시면 도와드리겠습니다.",
        alternatives=("가능한 기능 안내",),
    ),
    BlockRule(
        id="inject.system_prompt",
        patterns=(r"(시스템|system)\s*(프롬프트|prompt|메시지|지침)|내부\s*(지침|프롬프트)|숨겨진\s*(지시|규칙)",
                  r"보여|알려|출력|공개|말해|복사|print|show|reveal|repeat|그대로"),
        message="시스템 설정이나 내부 지침은 공개할 수 없습니다. 원하시는 업무를 말씀해 주시면 도와드리겠습니다.",
        alternatives=("가능한 기능 안내",),
    ),
    BlockRule(
        id="inject.privilege",
        patterns=(r"관리자\s*(권한|모드)|운영자\s*권한|루트\s*권한|개발자\s*모드|admin\s*(mode|권한)|sudo|탈옥|jailbreak|DAN\s*모드"
                  r"|보안\s*(제한|필터|정책)\S*\s*(없이|풀|해제|끄|우회)",),
        message="권한을 높이거나 보안 제한을 해제하는 요청은 처리할 수 없습니다.",
        alternatives=("가능한 기능 안내",),
    ),
    BlockRule(
        id="sql.write",
        patterns=(r"\bDROP\s+(TABLE|DATABASE|SCHEMA|VIEW|INDEX)\b|\bDELETE\s+FROM\b|\bUPDATE\s+\w+\s+SET\b|\bINSERT\s+INTO\b"
                  r"|\bTRUNCATE\b|\bALTER\s+TABLE\b|\bGRANT\s+\w+|\bREVOKE\s+\w+"
                  r"|(테이블|DB|디비|데이터베이스|스키마)\S*\s*(을|를|에서)?\s*.{0,8}(삭제|지워|드롭|날려|수정해|변경해|비워)",),
        message="데이터를 삭제·변경하는 요청은 처리할 수 없습니다. 이 챗봇은 읽기 전용 조회만 합니다.",
        alternatives=("조회 가능한 데이터 보기",),
    ),
)


@dataclass(frozen=True)
class MaskRule:
    id: str
    pattern: str
    mask: Callable[[re.Match], str]
    min_digits: int = 0                # 숫자 개수가 이보다 적으면 건너뜀 (날짜 2026-04-01 등 오탐 방지)
    _compiled: re.Pattern | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def apply(self, text: str) -> tuple[str, int]:
        n = 0

        def _sub(m: re.Match) -> str:
            nonlocal n
            if sum(c.isdigit() for c in m.group(0)) < self.min_digits:
                return m.group(0)
            n += 1
            return self.mask(m)

        return self._compiled.sub(_sub, text), n


def _keep_last(k: int):
    """숫자만 *로 바꾸고 마지막 k자리는 남긴다. 구분자(-, 공백)는 유지."""
    def f(m: re.Match) -> str:
        s = m.group(0)
        total = sum(c.isdigit() for c in s)
        out, seen = [], 0
        for c in s:
            if c.isdigit():
                seen += 1
                out.append(c if seen > total - k else "*")
            else:
                out.append(c)
        return "".join(out)
    return f


def _mask_rrn(m: re.Match) -> str:  # 900101-1234567 -> 900101-1******
    return f"{m.group(1)}{m.group(2)}{m.group(3)}******"


def _mask_phone(m: re.Match) -> str:  # 010-1234-5678 -> 010-****-5678
    return f"{m.group(1)}{m.group(2)}{'*' * len(m.group(3))}{m.group(4)}{m.group(5)}"


# 순서 중요: 카드 → 주민번호 → 전화 → 계좌. 앞 규칙이 숫자를 *로 바꾸므로 뒤 규칙과 겹쳐 잡히지 않는다.
MASK_RULES: tuple[MaskRule, ...] = (
    MaskRule("pii.card", r"(?<![\d*])\d{4}([- ]?)\d{4}\1\d{4}\1\d{4}(?![\d*])", _keep_last(4)),
    MaskRule("pii.rrn", r"(?<![\d*])(\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(\s?-\s?|)([1-8])\d{6}(?![\d*])", _mask_rrn),
    MaskRule("pii.phone", r"(?<![\d*])(01[016789]|0[2-6]\d?)([- .]?)(\d{3,4})([- .]?)(\d{4})(?![\d*])", _mask_phone),
    MaskRule("pii.account", r"(?<![\d*\-])\d{2,6}-\d{2,6}-\d{2,7}(?:-\d{1,3})?(?![\d*\-])", _keep_last(4), min_digits=10),
)


def mask_text(text: str) -> tuple[str, dict[str, int]]:
    """PII 패턴을 가린다. (가린 문자열, {규칙 id: 건수})."""
    hits: dict[str, int] = {}
    for r in MASK_RULES:
        text, n = r.apply(text)
        if n:
            hits[r.id] = hits.get(r.id, 0) + n
    return text, hits


def match_block(text: str) -> list[BlockRule]:
    return [r for r in BLOCK_RULES if r.match(text)]


# ---------- 메시지 유틸 ----------
def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()


def _mask_content(content):
    """str 또는 멀티모달 블록 목록의 텍스트 부분만 마스킹. 이미지 블록은 그대로 둔다."""
    if isinstance(content, str):
        return mask_text(content)
    hits: dict[str, int] = {}
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            t, h = mask_text(b.get("text", ""))
            for k, v in h.items():
                hits[k] = hits.get(k, 0) + v
            out.append({**b, "text": t})
        else:
            out.append(b)
    return out, hits


def _last(messages: list, cls):
    for m in reversed(messages or []):
        if isinstance(m, cls):
            return m
    return None


# 운영용 분류기 슬롯: async (text) -> None(통과) | {"rule", "reason", "alternatives"}(차단)
# 예: NeMo Guardrails의 self-check input rail, Llama Guard 분류 결과를 이 형태로 감싸 꽂는다.
Classifier = Callable[[str], Awaitable[dict | None]]


# ---------- 노드 ----------
def build_input_guard(classifier: Classifier | None = None):
    async def input_guard(state, config) -> dict:
        turn = state.get("turn", 0)
        msg = _last(state.get("messages", []), HumanMessage)
        if msg is None:
            return {"guard": {"blocked": False, "reason": None, "rule": None, "alternatives": []}}
        out: dict = {}
        flags: list[dict] = []

        # 1) 본인 PII 마스킹: 턴은 계속, 원문은 같은 id로 덮어쓴다
        masked, hits = _mask_content(msg.content)
        if hits:
            for rid, n in hits.items():
                flags.append({"stage": "input", "rule": rid, "action": "mask", "count": n, "turn": turn})
            if msg.id:
                out["messages"] = [msg.model_copy(update={"content": masked})]

        # 2) 차단 규칙: 구조화 칩 턴은 건너뜀 (서버가 만든 선택지)
        payload = state.get("payload") or {}
        guard = {"blocked": False, "reason": None, "rule": None, "alternatives": []}
        if payload.get("op") not in TRUSTED_OPS:
            text = _text_of(masked)
            matched = match_block(text)
            for r in matched:
                flags.append({"stage": "input", "rule": r.id, "action": "block", "turn": turn})
            if matched:
                r = matched[0]
                guard = {"blocked": True, "reason": r.message, "rule": r.id, "alternatives": list(r.alternatives)}
            elif classifier is not None and text:
                verdict = await classifier(text)
                if verdict:
                    rid = verdict.get("rule", "classifier")
                    flags.append({"stage": "input", "rule": rid, "action": "block", "turn": turn})
                    guard = {"blocked": True, "rule": rid,
                             "reason": verdict.get("reason") or "정책상 처리할 수 없는 요청입니다.",
                             "alternatives": list(verdict.get("alternatives") or [])}

        out["guard"] = guard
        if flags:
            out["policy_flags"] = flags
        return out

    return input_guard


def build_policy_reply():
    async def policy_reply(state, config) -> dict:
        g = state.get("guard") or {}
        answer = g.get("reason") or "정책상 처리할 수 없는 요청입니다."
        chips = [{"label": a, "type": "policy", "payload": {"op": "say", "text": a}} for a in g.get("alternatives", [])[:3]]
        return {"answer": answer, "messages": [AIMessage(content=answer)], "followups": chips}

    return policy_reply


def build_output_guard():
    async def output_guard(state, config) -> dict:
        answer = state.get("answer") or ""
        masked, hits = mask_text(answer)
        if not hits:
            return {}
        out: dict = {"answer": masked,
                     "policy_flags": [{"stage": "output", "rule": rid, "action": "mask", "count": n, "turn": state.get("turn", 0)}
                                      for rid, n in hits.items()]}
        msg = _last(state.get("messages", []), AIMessage)
        if msg is not None and msg.id:  # 같은 id로 돌려주면 add_messages가 원문을 덮어쓴다
            content, _ = _mask_content(msg.content)
            out["messages"] = [msg.model_copy(update={"content": content})]
        return out

    return output_guard
