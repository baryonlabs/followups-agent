"""OpenRouter 연결 (OpenAI 호환 엔드포인트)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # 셸의 관리용 키보다 프로젝트 .env의 추론 키를 우선

MODEL_MAIN = os.getenv("MODEL_MAIN", "anthropic/claude-sonnet-5.5")
MODEL_FAST = os.getenv("MODEL_FAST", "anthropic/claude-haiku-5.5")
MAIN_REASONING_EFFORT = os.getenv("MAIN_REASONING_EFFORT", "low")  # 판단 호출의 사고량을 낮춰 지연 단축


def llm(model: str | None = None, **kw) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or MODEL_MAIN,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        timeout=90,
        max_retries=2,
        max_tokens=kw.pop("max_tokens", int(os.getenv("MAX_TOKENS", "2048"))),  # 기본값(모델 최대치)이면 OpenRouter가 크레딧 선확보로 402를 냄
        default_headers={"X-Title": "followups-agent-local"},
        **kw,
    )
