"""도메인 플러그인 레지스트리.

그래프 배선(노드·엣지)은 그대로 두고, 특정 서비스 도메인(외부 도구를 부르는 질의응답)을 플러그인으로 붙인다.
플러그인은 의도 카탈로그 항목, 추출기 안내문, worker, 의도 보정 규칙을 한 묶음으로 등록한다.

    from followups.domains import DomainPlugin, register
    register(DomainPlugin(name="retail", route="retail", intents={"sales_lookup": {...}}, build_worker=build_retail_worker, ...))

불러오기
- 환경변수 FOLLOWUPS_PLUGINS=경로1.py,경로2.py
- 저장소의 local/plugins/*_domain.py (git 제외 폴더: 조직·과제 전용 플러그인을 공개 코드와 분리해 둔다)

프로필
- 등록하면 INTENT_PROFILES[name] = 플러그인 의도 + 공통 의도(첨부·보고서·선호·other)가 생긴다.
- build_graph(profile=name) 이면 추출기는 그 의도만 고르고, 잡담·범위 밖(other)도 handles_other면 플러그인 worker가 답한다.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from followups import state

ROOT = Path(__file__).resolve().parents[2]
COMMON_INTENTS = ["attachment_analysis", "report_compose", "set_preference", "other"]


@dataclass
class DomainPlugin:
    name: str                                   # 프로필 이름 (build_graph(profile=name))
    route: str                                  # CATALOG route 값. worker 노드는 f"{route}_worker"
    intents: dict[str, dict[str, Any]]          # 의도 → 카탈로그 항목 (route는 자동으로 채운다)
    build_worker: Callable[[], Callable]        # async worker(arg, config) -> {"task_stack": [task]}
    extractor_hint: str = ""                    # 의도 추출기 시스템 프롬프트에 덧붙일 안내
    domain_label: str = ""                      # "너는 {domain_label} 챗봇의 의도 추출기다"
    # 규칙 보정 (추출기 결과를 고친다). correct_intent(intent, text, prev_intent) -> intent
    correct_intent: Callable[[str, str, str | None], str] | None = None
    # 'other'로 추출된 짧은 후속을 진행 중인 도메인 의도로 잇는다. continue_intent(text, prev_intent) -> intent | None
    continue_intent: Callable[[str, str | None], str | None] | None = None
    handles_other: bool = True                  # 잡담·범위 밖과 근거 없는 보고서 요청도 이 worker가 답한다
    samples: list[str] = field(default_factory=list)   # 데모 UI 예시 질문
    judge_context: str = ""                     # 정답지 채점관에게 줄 서비스 설명 한 줄
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def worker_node(self) -> str:
        return f"{self.route}_worker"


PLUGINS: dict[str, DomainPlugin] = {}


def register(p: DomainPlugin) -> DomainPlugin:
    for intent, entry in p.intents.items():
        state.CATALOG[intent] = {"required": [], "risk": "low", "force_action": True, **entry, "route": p.route}
        state.INTENT_SLOTS.setdefault(intent, [])
    state.INTENT_PROFILES[p.name] = [*p.intents, *COMMON_INTENTS]
    state.INTENT_PROFILES["all"] = list(state.CATALOG)
    PLUGINS[p.name] = p
    return p


def get(profile: str | None) -> DomainPlugin | None:
    return PLUGINS.get(profile or "")


def routes() -> set[str]:
    return {p.route for p in PLUGINS.values()}


_loaded = False


def load_plugins() -> list[str]:
    """FOLLOWUPS_PLUGINS와 local/plugins/*_domain.py를 한 번만 불러온다. 불러온 파일 목록을 돌려준다."""
    global _loaded
    if _loaded:
        return []
    _loaded = True
    paths = [Path(p.strip()) for p in os.getenv("FOLLOWUPS_PLUGINS", "").split(",") if p.strip()]
    local = ROOT / "local" / "plugins"
    if local.is_dir():
        paths += sorted(local.glob("*_domain.py"))
    done = []
    for path in paths:
        if not path.is_file():
            continue
        if str(path.parent) not in sys.path:  # 플러그인 옆의 보조 모듈을 import할 수 있게
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(f"followups_plugin_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # 모듈이 import될 때 register()를 부른다
        done.append(str(path))
    return done
