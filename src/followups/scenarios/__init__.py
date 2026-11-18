"""시나리오 묶음. 각 모듈은 SCENARIOS: dict[str, dict]를 내보낸다 (run.py 형식과 같음).

기능별 담당이 자기 파일만 추가한다: sql.py, policy.py, multimodal.py, memory.py ...
"""
from __future__ import annotations

import importlib
import pkgutil


def load_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{m.name}")
        out.update(getattr(mod, "SCENARIOS", {}))
    return out
