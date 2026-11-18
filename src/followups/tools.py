"""MCP 클라이언트: 서버별 세션을 앱 수명 동안 유지하고, 도구를 서버 단위로 묶어 둔다."""
from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from followups.db import ROOT

SERVERS = {
    name: {
        "command": sys.executable,
        "args": ["-m", f"followups.mcp_servers.{name}"],
        "transport": "stdio",
        "cwd": str(ROOT),
        "env": {**os.environ, "FASTMCP_LOG_LEVEL": "WARNING"},
    }
    for name in ("payment", "member", "policy", "datahub")
}


@dataclass
class Toolbox:
    by_server: dict[str, list[BaseTool]] = field(default_factory=dict)

    @property
    def by_name(self) -> dict[str, BaseTool]:
        return {t.name: t for ts in self.by_server.values() for t in ts}

    def subset(self, *servers: str, names: list[str] | None = None) -> list[BaseTool]:
        """동적 도구 바인딩: 필요한 서버(와 도구)만 골라서 노출."""
        tools = [t for s in servers for t in self.by_server[s]]
        return [t for t in tools if names is None or t.name in names]

    async def call(self, name: str, args: dict, config: RunnableConfig | None = None) -> dict:
        out = await self.by_name[name].ainvoke(args, config=config)
        return parse_tool_output(out)


def parse_tool_output(out) -> dict:
    if isinstance(out, tuple):  # (content, artifact)
        out = out[0]
    if isinstance(out, list):
        out = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"text": out}
    return out if isinstance(out, dict) else {"value": out}


async def open_toolbox(stack: AsyncExitStack) -> Toolbox:
    client = MultiServerMCPClient(SERVERS)
    box = Toolbox()
    for name in SERVERS:
        session = await stack.enter_async_context(client.session(name))
        box.by_server[name] = await load_mcp_tools(session)
    return box
