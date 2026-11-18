"""멀티모달 시나리오용 테스트 차트 생성.

uv run python -m followups.make_chart     # data/attachments/mapo_sales_2026h1.png

수치는 상권 DB 시드와 같다 (1~6월 매출, 단위 백만원). 막대 위에 값 라벨을 찍어 VLM이 추정 없이 읽게 한다.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "attachments" / "mapo_sales_2026h1.png"
TITLE = "마포구 월별 매출 (2026년 1~6월, 단위: 백만원)"
DATA = [("1월", 1250), ("2월", 1310), ("3월", 1290), ("4월", 870), ("5월", 1180), ("6월", 1240)]


def _font(size: int):
    for p in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()  # 한글 글꼴이 없으면 기본 글꼴 (한글은 깨질 수 있음)


def make(out: Path = OUT, data=DATA, title: str = TITLE) -> Path:
    W, H = 900, 560
    left, right, top, bottom = 90, 40, 90, 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title, f_axis, f_val = _font(24), _font(16), _font(16)
    d.text((W / 2, 40), title, fill="#1f2328", font=f_title, anchor="mm")

    ymax = 1500
    ph = H - top - bottom
    pw = W - left - right
    y_of = lambda v: top + ph - v / ymax * ph  # noqa: E731
    for v in range(0, ymax + 1, 250):  # 눈금선 + y축 라벨
        y = y_of(v)
        d.line([(left, y), (W - right, y)], fill="#e5e7eb", width=1)
        d.text((left - 10, y), f"{v:,}", fill="#57606a", font=f_axis, anchor="rm")
    d.line([(left, top), (left, top + ph)], fill="#57606a", width=1)
    d.line([(left, top + ph), (W - right, top + ph)], fill="#57606a", width=1)

    slot = pw / len(data)
    bw = slot * 0.55
    for i, (label, v) in enumerate(data):
        cx = left + slot * (i + 0.5)
        color = "#c2410c" if v == min(x for _, x in data) else "#2563eb"  # 최저 막대만 강조
        d.rectangle([cx - bw / 2, y_of(v), cx + bw / 2, top + ph], fill=color)
        d.text((cx, y_of(v) - 8), f"{v:,}", fill="#1f2328", font=f_val, anchor="mb")
        d.text((cx, top + ph + 12), label, fill="#1f2328", font=f_axis, anchor="mt")
    d.text((left, H - 22), "자료: 상권 DB(월별) · 단위: 백만원", fill="#57606a", font=_font(13), anchor="lm")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


if __name__ == "__main__":
    print(make())
