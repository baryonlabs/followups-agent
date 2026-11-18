"""그래프 플러그인 계약.

graph.py의 배선(노드·엣지)은 고정하고, 각 기능 모듈은 아래 이름의 함수만 제공한다.
모듈끼리는 서로 import하지 않는다. 공유하는 것은 state.py의 State/Task 스키마뿐이다.

Worker (planner가 Send로 부르는 작업 처리기)
    async def worker(arg: WorkerArg, config) -> dict   # {"task_stack": [task], "entity_memory"?: {...}}
    - arg["task"]는 복사본. status를 done/blocked/failed 중 하나로 바꿔 돌려준다.
    - 막히면 status="blocked", missing=[slot], result={"choices": {slot: [{"label", "value"}]}} → 규칙이 칩을 만든다.

Entry/exit node (State 전체를 받는 노드)
    async def node(state: State, config) -> dict       # State 부분 갱신
"""
from __future__ import annotations

from typing import Any, TypedDict

from followups.state import Task


class WorkerArg(TypedDict, total=False):
    task: Task
    user_id: str
    turn: int
    utterance: str              # 이번 턴 사용자 발화 (텍스트 부분)
    inputs: dict[str, dict]     # depends_on 작업들: {task_id: {"intent", "status", "slots", "result", "sql"}}
    attachments: dict[str, Any] # State.attachments
    profile: dict               # State.profile (확정 선호)
    guidelines_text: str        # render_guidelines(task["guidelines"], profile)
