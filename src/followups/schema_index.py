"""Text-to-SQL 규칙부: 스키마 인덱스(검색) · SQL 검증기 · 비용 추정. 전부 LLM 없이 테스트하는 순수 함수.

- 스키마 인덱스: 테이블 정의서(테이블 설명 + 컬럼 설명)를 문서로 보고 BM25로 점수를 낸다.
  한국어 형태소 분석 없이 글자 2-gram을 쓴다. 서비스에서는 같은 문서를 임베딩해 pgvector로 바꾼다
  (search_tables의 반환 형식만 유지하면 worker는 그대로).
- 검증기: sqlglot 파싱 → 단일 SELECT, 허용 테이블, PII 컬럼·SELECT * 차단, LIMIT 강제.
- 비용: 테이블 추정 행 수 × WHERE 조건 선택도. 서비스에서는 EXPLAIN 결과로 바꾼다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from followups.sql_data import SCHEMA, SOURCES, pii_columns

MAX_LIMIT = 200
COST_LIMIT_ROWS = 1_000_000  # 이보다 많이 읽을 것으로 추정되면 사용자 확인 (B 칩)
AMBIGUOUS_RATIO = 0.75       # 2등 출처 점수가 1등의 이 비율 이상이면 모호 → 출처 선택 칩


# ---------- 스키마 인덱스 ----------
def tokenize(text: str) -> list[str]:
    """단어 단위로 자르고, 한글 단어는 글자 2-gram으로 (한 글자 단어는 그대로)."""
    out: list[str] = []
    for w in re.findall(r"[0-9A-Za-z_]+|[가-힣]+", (text or "").lower()):
        if re.fullmatch(r"[가-힣]+", w) and len(w) > 1:
            out += [w[i:i + 2] for i in range(len(w) - 1)]
        else:
            out.append(w)
    return out


@dataclass
class TableDoc:
    source: str
    table: str
    tokens: list[str]


def _docs() -> list[TableDoc]:
    docs = []
    for src, tables in SCHEMA.items():
        for name, t in tables.items():
            text = " ".join([name, t["desc"], *(f"{c} {d}" for c, (_, d) in t["columns"].items())])
            docs.append(TableDoc(src, name, tokenize(text)))
    return docs


DOCS = _docs()
_DF = Counter(tok for d in DOCS for tok in set(d.tokens))
_AVGDL = sum(len(d.tokens) for d in DOCS) / len(DOCS)


def search_tables(query: str, k1: float = 1.2, b: float = 0.75) -> list[dict]:
    """BM25 점수순 [{"source", "table", "score"}]. (pgvector로 바꿀 자리)"""
    q = tokenize(query)
    n = len(DOCS)
    out = []
    for d in DOCS:
        tf = Counter(d.tokens)
        s = 0.0
        for tok in set(q):
            if tok not in tf:
                continue
            idf = math.log(1 + (n - _DF[tok] + 0.5) / (_DF[tok] + 0.5))
            s += idf * tf[tok] * (k1 + 1) / (tf[tok] + k1 * (1 - b + b * len(d.tokens) / _AVGDL))
        out.append({"source": d.source, "table": d.table, "score": round(s, 4)})
    return sorted(out, key=lambda r: -r["score"])


def route_source(query: str) -> dict:
    """출처(DB)를 고른다.

    반환: {"source": id | None, "candidates": [id...], "scores": {id: 점수}, "tables": [후보 테이블]}
    - 1등이 확실하면 source를 채운다
    - 점수가 비슷하거나(모호) 아무것도 안 맞으면 source=None, candidates에 고를 후보
    """
    hits = search_tables(query)
    scores: dict[str, float] = {}
    for h in hits:
        scores[h["source"]] = max(scores.get(h["source"], 0.0), h["score"])
    ranked = sorted(scores, key=lambda s: -scores[s])
    top = scores[ranked[0]]
    if top <= 0:
        return {"source": None, "candidates": list(SOURCES), "scores": scores, "tables": []}
    close = [s for s in ranked if scores[s] >= top * AMBIGUOUS_RATIO]
    tables = [h["table"] for h in hits if h["score"] > 0 and h["source"] in close]
    if len(close) > 1:
        return {"source": None, "candidates": close, "scores": scores, "tables": tables}
    return {"source": ranked[0], "candidates": [ranked[0]], "scores": scores, "tables": tables}


# ---------- 검증기 ----------
@dataclass
class Validation:
    ok: bool
    error: str | None = None      # parse_error | multiple_statements | not_select | table_not_allowed | pii_column | select_star
    detail: str = ""
    query: str = ""               # 정규화 + LIMIT 강제된 SQL
    tables: list[str] = field(default_factory=list)
    est_rows: int = 0
    tree: exp.Expression | None = field(default=None, repr=False)


_FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Merge)


def validate_sql(sql: str, source: str, max_limit: int = MAX_LIMIT) -> Validation:
    """생성된 SQL을 실행 전에 검사한다. 통과하면 LIMIT이 강제된 query를 돌려준다."""
    if source not in SCHEMA:
        return Validation(False, "unknown_source", source)
    try:
        stmts = [s for s in sqlglot.parse(sql or "", read="sqlite") if s is not None]
    except sqlglot.errors.SqlglotError as e:
        return Validation(False, "parse_error", str(e)[:200])
    if len(stmts) != 1:
        return Validation(False, "multiple_statements" if stmts else "parse_error", f"문장 {len(stmts)}개")
    tree = stmts[0]
    if not isinstance(tree, exp.Select) or any(isinstance(n, _FORBIDDEN) for n in tree.walk()):
        return Validation(False, "not_select", type(tree).__name__)

    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    tables = sorted({t.name for t in tree.find_all(exp.Table)} - cte_names)
    allowed = set(SCHEMA[source])
    bad = [t for t in tables if t not in allowed]
    if bad:
        return Validation(False, "table_not_allowed", ", ".join(bad), tables=tables)

    pii = pii_columns(source)
    used_pii = sorted({c.name for c in tree.find_all(exp.Column) if c.name in pii})
    if used_pii:
        return Validation(False, "pii_column", ", ".join(used_pii), tables=tables)
    # SELECT * / t.* : 어떤 컬럼이 나갈지 검증할 수 없으므로 금지 (COUNT(*)는 허용)
    for sel in tree.find_all(exp.Select):
        for proj in sel.expressions:
            if isinstance(proj, exp.Star) or (isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star)):
                pii_tables = [t for t in tables if SCHEMA[source][t].get("pii")]
                return Validation(False, "pii_column" if pii_tables else "select_star",
                                  "SELECT *" + (f" ({', '.join(pii_tables)})" if pii_tables else ""), tables=tables)

    limit = tree.args.get("limit")
    n = None
    if limit is not None and isinstance(limit.expression, exp.Literal) and not limit.expression.is_string:
        try:
            n = int(limit.expression.this)
        except ValueError:
            n = None
    if n is None or n > max_limit or n < 0:
        tree = tree.limit(max_limit)
    return Validation(True, query=tree.sql(dialect="sqlite"), tables=tables,
                      est_rows=estimate_rows(tree, source), tree=tree)


# ---------- 비용 ----------
# WHERE에 걸린 컬럼별 선택도 (조건이 있으면 그만큼만 읽는다고 본다)
SELECTIVITY = {"gu": 0.04, "industry": 0.1, "ym": 0.25, "quarter": 0.25, "biz_id": 0.000001}


def estimate_rows(tree: exp.Expression, source: str) -> int:
    """읽을 행 수 추정 = Σ 테이블 추정 행 수 × WHERE 컬럼 선택도의 곱."""
    filtered: set[str] = set()
    for w in tree.find_all(exp.Where):
        filtered |= {c.name for c in w.find_all(exp.Column)}
    total = 0.0
    for t in {t.name for t in tree.find_all(exp.Table)}:
        meta = SCHEMA[source].get(t)
        if not meta:
            continue
        rows = float(meta["est_rows"])
        for c in filtered & set(meta["columns"]):
            rows *= SELECTIVITY.get(c, 0.5)
        total += rows
    return int(total)


def over_budget(v: Validation, limit: int = COST_LIMIT_ROWS) -> bool:
    return v.ok and v.est_rows > limit
