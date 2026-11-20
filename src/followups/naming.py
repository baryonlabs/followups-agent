"""챗봇 이름과 부르는 이름.

기본 이름은 '우리공', 짧게 '공'. 사용자는 "@우리공 마포구 점포 수", "@공 구별로도"처럼 불러도 된다.
부르는 말(@이름, 이름아/야, 쉼표)은 떼고 나머지만 그래프에 넘긴다.

바꾸기: FOLLOWUPS_APP_NAME=새이름 FOLLOWUPS_APP_ALIASES=짧은이름1,짧은이름2 (또는 web --name 새이름)
"""
from __future__ import annotations

import os
import re

DEFAULT_NAME = "우리공"
DEFAULT_ALIASES = ("공",)


def app_name() -> str:
    return (os.getenv("FOLLOWUPS_APP_NAME") or DEFAULT_NAME).strip()


def aliases() -> list[str]:
    raw = os.getenv("FOLLOWUPS_APP_ALIASES")
    names = [a.strip() for a in raw.split(",")] if raw is not None else list(DEFAULT_ALIASES)
    return [a for a in dict.fromkeys([app_name(), *names]) if a]


def call_names() -> list[str]:
    """화면 안내용: ['@우리공', '@공']"""
    return [f"@{a}" for a in aliases()]


def strip_call(text: str) -> tuple[str, bool]:
    """맨 앞의 부르는 말을 뗀다. (남은 글, 불렀는지). '@공장'처럼 이름이 낱말 일부면 떼지 않는다."""
    s = text.strip()
    for name in sorted(aliases(), key=len, reverse=True):  # 긴 이름 먼저 ('우리공'이 '공'보다 먼저)
        m = re.match(rf"@{re.escape(name)}(?:아|야)?(?=$|[\s,.!?~:]|님)(?:님)?[\s,.!?~:]*", s)
        if m:
            return s[m.end():].strip(), True
    return s, False


def icon_svg(color: str = "#2f6fed") -> str:
    """둥근 말풍선 안에 이름 첫 글자. 별도 이미지 파일 없이 쓴다."""
    ch = (aliases()[-1] if aliases() else app_name())[:1]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            f'<path d="M32 4C16.5 4 4 14.9 4 28.5c0 7.6 3.9 14.4 10.1 18.9L12 60l13.6-7.6c2.1.4 4.2.6 6.4.6 '
            f'15.5 0 28-10.9 28-24.5S47.5 4 32 4z" fill="{color}"/>'
            f'<text x="32" y="38" text-anchor="middle" font-size="26" font-weight="700" '
            f'font-family="Pretendard,Malgun Gothic,sans-serif" fill="#fff">{ch}</text></svg>')
