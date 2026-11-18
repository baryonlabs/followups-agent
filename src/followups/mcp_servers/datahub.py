"""MCP Server D: 공공 데이터 조회 (stdio). sql_agent 전용.

- 출처(source)별 sqlite 파일을 읽기 전용(`file:...?mode=ro`, PRAGMA query_only)으로 연다
- 실행 직전에 검증기를 한 번 더 돌린다 (worker 검증 + 서버 검증 이중 통제. 서비스에서는 DB 계정 권한이 세 번째)
- 문장 단위 timeout: sqlite progress handler로 시간이 지나면 중단
서비스에서는 DB마다 서버를 따로 두고(sangkwon_db, bizreg_db) 읽기 전용 계정으로 접속한다.
"""
from __future__ import annotations

import sqlite3
import time

from mcp.server.fastmcp import FastMCP

from followups.schema_index import MAX_LIMIT, validate_sql
from followups.sql_data import SCHEMA, SOURCES, build, ddl

mcp = FastMCP("datahub", log_level="WARNING")

DEFAULT_TIMEOUT_MS = 3000


def execute(source: str, sql: str, max_rows: int = MAX_LIMIT, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict:
    """검증 → 읽기 전용 실행. 실패해도 예외 대신 {"error", "detail"}를 돌려준다."""
    if source not in SOURCES:
        return {"error": "unknown_source", "detail": source}
    v = validate_sql(sql, source, max_limit=min(max_rows, MAX_LIMIT))
    if not v.ok:
        return {"error": v.error, "detail": v.detail}
    path = build(source)  # 없으면 시드로 생성
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
    try:
        con.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + timeout_ms / 1000
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cur = con.execute(v.query)
        columns = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchmany(max_rows + 1)]
    except sqlite3.OperationalError as e:
        msg = str(e)
        return {"error": "timeout" if "interrupted" in msg else "execution_error", "detail": msg[:200]}
    except sqlite3.Error as e:
        return {"error": "execution_error", "detail": str(e)[:200]}
    finally:
        con.close()
    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    return {"source": source, "columns": columns, "rows": rows, "row_count": len(rows),
            "truncated": truncated, "query": v.query, "tables": v.tables}


@mcp.tool()
def run_query(source: str, sql: str, max_rows: int = MAX_LIMIT, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict:
    """데이터 출처(source: sangkwon|bizreg)에 읽기 전용 SELECT 한 문장을 실행한다. 결과 행은 max_rows로 제한된다."""
    return execute(source, sql, max_rows, timeout_ms)


@mcp.tool()
def describe_source(source: str) -> dict:
    """데이터 출처의 테이블 정의(DDL과 설명)를 돌려준다."""
    if source not in SOURCES:
        return {"error": "unknown_source", "detail": source}
    return {"source": source, **SOURCES[source], "tables": list(SCHEMA[source]), "ddl": ddl(source)}


if __name__ == "__main__":
    mcp.run()
