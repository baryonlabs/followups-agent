"""로컬 데모 UI 서버: 채팅 + 후속 질문 칩 + 이미지 첨부 + 작업 스택/가드/메모리 패널.

uv run python -m followups.web          # http://localhost:8765
"""
from __future__ import annotations

import os
import time
import traceback
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import uvicorn
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from followups import db, memory
from followups import llm as _llm  # noqa: F401  (.env 로드)
from followups.graph import build_graph
from followups.tools import open_toolbox

STATIC = Path(__file__).parent / "web_static"
USER_ID = "u1"
APP: dict = {}


def _callbacks() -> list:
    """Langfuse가 떠 있으면 트레이스를 남긴다 (없어도 UI는 동작)."""
    if APP.get("langfuse") is None:
        try:
            from langfuse import get_client

            APP["langfuse"] = bool(get_client().auth_check())
        except Exception:
            APP["langfuse"] = False
    if not APP["langfuse"]:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


@asynccontextmanager
async def lifespan(_app):
    async with AsyncExitStack() as stack:
        box = await open_toolbox(stack)
        APP["graph"] = build_graph(box)
        APP["chips"] = {}
        yield


def _snapshot(values: dict) -> dict:
    tasks = values.get("task_stack", [])
    atts = values.get("attachments") or {}
    return {
        "tasks": [{k: t.get(k) for k in ("id", "intent", "mode", "status", "phase", "missing", "depends_on", "slots",
                                          "guidelines", "error", "turn", "route")} for t in tasks],
        "results": {t["id"]: t.get("result") for t in tasks},
        "sql": {t["id"]: t.get("sql") for t in tasks if t.get("sql")},
        "attachments": {k: {kk: v.get(kk) for kk in ("kind", "facts", "object_key", "turn")} for k, v in atts.items()},
        "profile": values.get("profile") or {},
        "policy_flags": values.get("policy_flags") or [],
        "entity_memory": values.get("entity_memory") or {},
        "turn": values.get("turn", 0),
    }


async def turn(request: Request) -> JSONResponse:
    body = await request.json()
    thread = body.get("thread_id") or uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": thread}}
    app = APP["graph"]
    text = (body.get("text") or "").strip()
    image = body.get("image")
    chip = body.get("chip")

    if chip:
        p = chip["payload"]
        shown = f"[칩] {chip['label']}"
        if p["op"] == "confirm":
            inp = Command(resume=p, update={"messages": [HumanMessage(content=shown)]})
        elif p["op"] == "say":
            shown = p["text"]
            inp = {"messages": [HumanMessage(content=p["text"])], "user_id": USER_ID, "payload": None}
        else:
            inp = {"messages": [HumanMessage(content=shown)], "user_id": USER_ID, "payload": p}
    elif image:
        content = [{"type": "text", "text": text or "이 이미지 봐줘"}, {"type": "image_url", "image_url": {"url": image}}]
        inp = {"messages": [HumanMessage(content=content)], "user_id": USER_ID, "payload": None}
        shown = text
    else:
        inp = {"messages": [HumanMessage(content=text)], "user_id": USER_ID, "payload": None}
        shown = text

    t0 = time.perf_counter()
    try:
        out = await app.ainvoke(inp, config={**cfg, "callbacks": _callbacks()})
    except Exception as e:
        traceback.print_exc()
        snap = await app.aget_state(cfg)
        msg = str(e)
        hint = "LLM 키가 거부되었습니다 (.env의 OPENROUTER_API_KEY 확인)" if "401" in msg else None
        return JSONResponse({"thread_id": thread, "error": f"{type(e).__name__}: {msg[:300]}", "hint": hint,
                             "shown": shown, "seconds": round(time.perf_counter() - t0, 2),
                             **_snapshot(snap.values if snap else {})})

    intr = out.get("__interrupt__")
    if intr:
        v = intr[0].value
        answer, chips = v["message"], v["chips"]
    else:
        answer, chips = out.get("answer", ""), list(out.get("followups", []))
        try:
            chips += await memory.after_turn(app, cfg, out) or []
        except Exception:
            traceback.print_exc()
    snap = await app.aget_state(cfg)
    return JSONResponse({"thread_id": thread, "shown": shown, "answer": answer, "chips": chips,
                         "interrupt": bool(intr), "seconds": round(time.perf_counter() - t0, 2),
                         **_snapshot(snap.values)}, headers={"Cache-Control": "no-store"})


async def reset(request: Request) -> JSONResponse:
    db.reset()
    return JSONResponse({"thread_id": uuid.uuid4().hex[:8]})


async def sample_image(request: Request):
    p = db.ROOT / "data" / "attachments" / "mapo_sales_2026h1.png"
    return FileResponse(p) if p.exists() else JSONResponse({"error": "no sample"}, status_code=404)


async def index(request: Request):
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


routes = [
    Route("/", index),
    Route("/api/turn", turn, methods=["POST"]),
    Route("/api/reset", reset, methods=["POST"]),
    Route("/api/sample-image", sample_image),
]
server = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(server, host="127.0.0.1", port=int(os.getenv("PORT", "8765")), log_level="info")
