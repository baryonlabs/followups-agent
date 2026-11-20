"""도메인 플러그인 레지스트리: 등록하면 카탈로그·프로필·그래프 노드가 붙는다."""
import copy

from followups import domains, state
from followups.graph import build_graph, extractor_hints


def _worker():
    async def worker(arg, config):
        return {"task_stack": []}
    return worker


def test_register_adds_profile_and_worker(monkeypatch):
    # 전역 레지스트리를 테스트 안에서만 바꾼다 (다른 테스트로 새지 않게)
    monkeypatch.setattr(state, "CATALOG", copy.deepcopy(state.CATALOG))
    monkeypatch.setattr(state, "INTENT_SLOTS", copy.deepcopy(state.INTENT_SLOTS))
    monkeypatch.setattr(state, "INTENT_PROFILES", copy.deepcopy(state.INTENT_PROFILES))
    monkeypatch.setattr(domains, "PLUGINS", dict(domains.PLUGINS))
    p = domains.register(domains.DomainPlugin(
        name="dummy", route="dummy", intents={"dummy_lookup": {}}, build_worker=_worker,
        extractor_hint="dummy_lookup: 더미 조회", domain_label="더미 서비스"))
    assert state.CATALOG["dummy_lookup"]["route"] == "dummy"
    assert state.INTENT_PROFILES["dummy"][0] == "dummy_lookup"
    assert "other" in state.INTENT_PROFILES["dummy"]
    assert domains.get("dummy") is p and "dummy" in domains.routes()
    assert "더미 조회" in extractor_hints("dummy")
    class Box:  # 커머스 하위 에이전트 도구 자리
        def subset(self, *a, **k):
            return []

    app = build_graph(Box(), profile="dummy")
    assert "dummy_worker" in app.get_graph().nodes
