"""목(mock) 공공 데이터 DB 2개: 시드 데이터 → sqlite 파일 (결정적으로 생성, 없거나 버전이 다르면 다시 만든다).

- sangkwon (상권 DB, 월별): stores_monthly, sales_monthly
- bizreg   (사업자 DB, 분기별): business_registrations, active_businesses, business_owners(PII)

두 DB 모두 '점포 수'에 해당하는 데이터가 있어서 "업종별 점포 수"는 출처가 실제로 모호하다 (→ 출처 선택 칩).
서비스에서는 각 DB가 별도 서버(읽기 전용 계정)이고, 이 파일의 SCHEMA가 테이블 정의서 역할을 한다.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path

from followups import db

SEED_VERSION = "2026-11-v1"

SOURCES: dict[str, dict] = {
    "sangkwon": {"label": "상권 DB(월별)", "desc": "상권 분석용 월별 점포·매출 통계", "grain": "월"},
    "bizreg": {"label": "사업자 DB(분기별)", "desc": "사업자 등록 기반 분기별 개업·폐업·영업 사업장 통계와 대표자 명부", "grain": "분기"},
}

GUS = ["마포구", "강남구", "종로구"]
INDUSTRIES = ["음식점", "소매", "서비스"]
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]
QUARTERS = ["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]

# 차트 담당과 맞춘 값: 마포구 업종 합계 월 매출(백만원). 4월 급감
MAPO_SALES_TOTAL = {"2026-01": 1250, "2026-02": 1310, "2026-03": 1290, "2026-04": 870, "2026-05": 1180, "2026-06": 1240}
INDUSTRY_SHARE = {"음식점": 0.5, "소매": 0.3, "서비스": 0.2}
# 다른 구는 완만한 추세 (업종 합계, 백만원)
OTHER_SALES_BASE = {"강남구": 2400, "종로구": 1600}
MONTH_TREND = [1.00, 1.02, 1.03, 1.01, 1.04, 1.05]
# 점포 수 기준값 (구, 업종). 마포구는 4월에 줄었다가 일부 회복
STORE_BASE = {
    ("마포구", "음식점"): 420, ("마포구", "소매"): 260, ("마포구", "서비스"): 180,
    ("강남구", "음식점"): 680, ("강남구", "소매"): 510, ("강남구", "서비스"): 390,
    ("종로구", "음식점"): 350, ("종로구", "소매"): 300, ("종로구", "서비스"): 150,
}
MAPO_STORE_FACTOR = [1.00, 1.01, 1.00, 0.86, 0.92, 0.95]

# 테이블 정의서: 설명·컬럼 설명(한글). 스키마 인덱스와 SQL 생성 프롬프트(DDL)가 모두 여기서 나온다.
#   est_rows: 운영 규모 추정 행 수 (서비스에서는 pg_class.reltuples / EXPLAIN 값). 비용 확인에 쓴다
#   pii: 조회 금지 컬럼
SCHEMA: dict[str, dict[str, dict]] = {
    "sangkwon": {
        "stores_monthly": {
            "desc": "월별 구·업종별 점포 수 통계. 상권 내 영업 점포(가게) 개수",
            "est_rows": 1_800_000,
            "columns": {
                "ym": ("TEXT", "기준 연월 'YYYY-MM' (월별)"),
                "gu": ("TEXT", "자치구 이름. 예: 마포구"),
                "industry": ("TEXT", "업종 대분류: 음식점/소매/서비스"),
                "store_count": ("INTEGER", "점포 수 (개)"),
            },
        },
        "sales_monthly": {
            "desc": "월별 구·업종별 추정 매출액 통계. 상권 매출 규모",
            "est_rows": 1_800_000,
            "columns": {
                "ym": ("TEXT", "기준 연월 'YYYY-MM' (월별)"),
                "gu": ("TEXT", "자치구 이름. 예: 마포구"),
                "industry": ("TEXT", "업종 대분류: 음식점/소매/서비스"),
                "sales_mkrw": ("REAL", "추정 매출액 (백만원)"),
            },
        },
    },
    "bizreg": {
        "business_registrations": {
            "desc": "분기별 구·업종별 사업자 신규 개업 수와 폐업 수",
            "est_rows": 600_000,
            "columns": {
                "quarter": ("TEXT", "기준 분기 'YYYY-Qn' (분기별)"),
                "gu": ("TEXT", "자치구 이름. 예: 마포구"),
                "industry": ("TEXT", "업종 대분류: 음식점/소매/서비스"),
                "new_count": ("INTEGER", "신규 개업 사업자 수"),
                "closed_count": ("INTEGER", "폐업 사업자 수"),
            },
        },
        "active_businesses": {
            "desc": "분기별 구·업종별 영업 중 사업장(점포) 수. 사업자 등록 기준 점포 수",
            "est_rows": 600_000,
            "columns": {
                "quarter": ("TEXT", "기준 분기 'YYYY-Qn' (분기별)"),
                "gu": ("TEXT", "자치구 이름. 예: 마포구"),
                "industry": ("TEXT", "업종 대분류: 음식점/소매/서비스"),
                "active_count": ("INTEGER", "영업 중 사업장 수 (개)"),
            },
        },
        "business_owners": {
            "desc": "사업자 대표자 명부. 사업자별 대표자 정보 (개인정보 포함, 집계만 허용)",
            "est_rows": 2_400_000,
            "pii": ["owner_name", "owner_rrn", "phone"],
            "columns": {
                "biz_id": ("TEXT", "사업자 id"),
                "gu": ("TEXT", "자치구 이름"),
                "industry": ("TEXT", "업종 대분류"),
                "owner_name": ("TEXT", "대표자 이름 (개인정보)"),
                "owner_rrn": ("TEXT", "대표자 주민등록번호 (개인정보)"),
                "phone": ("TEXT", "대표자 전화번호 (개인정보)"),
            },
        },
    },
}


def pii_columns(source: str) -> set[str]:
    return {c for t in SCHEMA[source].values() for c in t.get("pii", [])}


def ddl(source: str, tables: list[str] | None = None) -> str:
    """SQL 생성 프롬프트에 넣을 DDL (컬럼 설명은 주석으로). PII 컬럼은 조회 금지로 표시."""
    out = []
    for name, t in SCHEMA[source].items():
        if tables and name not in tables:
            continue
        pii = set(t.get("pii", []))
        cols = [f"  {c} {typ},  -- {desc}{' [조회 금지]' if c in pii else ''}" for c, (typ, desc) in t["columns"].items()]
        cols[-1] = cols[-1].replace(",  --", "  --", 1)
        out.append(f"-- {t['desc']}\nCREATE TABLE {name} (\n" + "\n".join(cols) + "\n);")
    return "\n".join(out)


# ---------- 시드 행 ----------
def _rows() -> dict[str, dict[str, list[tuple]]]:
    stores, sales = [], []
    for i, ym in enumerate(MONTHS):
        for gu in GUS:
            for ind in INDUSTRIES:
                base = STORE_BASE[(gu, ind)]
                f = MAPO_STORE_FACTOR[i] if gu == "마포구" else 1 + 0.01 * i
                stores.append((ym, gu, ind, round(base * f)))
                total = MAPO_SALES_TOTAL[ym] if gu == "마포구" else OTHER_SALES_BASE[gu] * MONTH_TREND[i]
                sales.append((ym, gu, ind, round(total * INDUSTRY_SHARE[ind], 1)))
    regs, active = [], []
    for qi, q in enumerate(QUARTERS):
        for gu in GUS:
            for ii, ind in enumerate(INDUSTRIES):
                base = STORE_BASE[(gu, ind)]
                new = 12 + (qi * 3 + ii * 5 + len(gu)) % 9 + base // 40
                closed = 8 + (qi * 2 + ii * 3) % 7 + base // 60
                if gu == "마포구" and q == "2026-Q2":
                    closed += 25  # 4월 급감과 맞물린 폐업 증가
                regs.append((q, gu, ind, new, closed))
                active.append((q, gu, ind, round(base * 1.08) + qi * 4 - (20 if gu == "마포구" and q == "2026-Q2" else 0)))
    owners = []
    n = 0
    for gu in GUS:
        for ind in INDUSTRIES:
            for k in range(4):
                n += 1
                # 가짜 개인정보 (형식만 흉내). 실제 조회는 검증기가 막는다
                owners.append((f"B{n:05d}", gu, ind, f"대표자{n:03d}", f"900101-1{n:06d}", f"010-0000-{n:04d}"))
    return {
        "sangkwon": {"stores_monthly": stores, "sales_monthly": sales},
        "bizreg": {"business_registrations": regs, "active_businesses": active, "business_owners": owners},
    }


# ---------- 파일 ----------
def data_dir() -> Path:
    return Path(os.getenv("FOLLOWUPS_DATA_DIR") or db.DATA_DIR)


def db_path(source: str) -> Path:
    if source not in SOURCES:
        raise KeyError(f"알 수 없는 데이터 출처: {source}")
    return data_dir() / f"{source}.sqlite"


def _current(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT value FROM _meta WHERE key='seed_version'").fetchone()
        finally:
            con.close()
        return bool(row) and row[0] == SEED_VERSION
    except sqlite3.Error:
        return False


def build(source: str, force: bool = False) -> Path:
    """시드로 sqlite 파일을 만든다. 이미 같은 버전이면 그대로 둔다. 임시 파일에 쓰고 교체(동시 실행 안전)."""
    p = db_path(source)
    if not force and _current(p):
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex[:8]}.tmp")
    con = sqlite3.connect(tmp)
    try:
        for name, t in SCHEMA[source].items():
            cols = ", ".join(f"{c} {typ}" for c, (typ, _) in t["columns"].items())
            con.execute(f"CREATE TABLE {name} ({cols})")
        for name, rows in _rows()[source].items():
            ph = ", ".join("?" * len(rows[0]))
            con.executemany(f"INSERT INTO {name} VALUES ({ph})", rows)
        con.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO _meta VALUES ('seed_version', ?)", (SEED_VERSION,))
        con.commit()
    finally:
        con.close()
    try:
        os.replace(tmp, p)
    except OSError:  # Windows: 다른 프로세스가 열어 둔 경우. 이미 누가 만들었으면 그것을 쓴다
        tmp.unlink(missing_ok=True)
        if not _current(p):
            raise
    return p


def ensure_all() -> dict[str, Path]:
    return {s: build(s) for s in SOURCES}


def table_row_counts(source: str) -> dict[str, int]:
    """로컬 목 DB의 실제 행 수 (참고용)."""
    return {name: len(rows) for name, rows in _rows()[source].items()}
