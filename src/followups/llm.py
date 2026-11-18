"""LLM 연결 (OpenAI 호환 엔드포인트).

기본은 OpenRouter. 공공 폐쇄망에서는 내부 추론 서버(vLLM 등 OpenAI 호환)로 바꾼다:
    LLM_BASE_URL=http://gpu-01.internal:8000/v1  LLM_API_KEY=...  MODEL_MAIN=...  MODEL_FAST=...
코드 변경 없이 환경변수만 바꾸면 된다.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # 셸의 관리용 키보다 프로젝트 .env의 추론 키를 우선

BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_MAIN = os.getenv("MODEL_MAIN", "anthropic/claude-sonnet-5.5")
MODEL_FAST = os.getenv("MODEL_FAST", "anthropic/claude-haiku-5.5")
MAIN_REASONING_EFFORT = os.getenv("MAIN_REASONING_EFFORT", "low")  # 판단 호출의 사고량을 낮춰 지연 단축


def _api_key() -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY 또는 OPENROUTER_API_KEY가 필요합니다 (.env)")
    return key


IS_OPENROUTER = "openrouter.ai" in BASE_URL
# 이미지 판독 모델. 내부 Solar처럼 이미지를 받지 않는 모델이면 VLM이 있는 엔드포인트를 따로 준다
VISION_BASE_URL = os.getenv("LLM_VISION_BASE_URL", BASE_URL)
MODEL_VISION = os.getenv("MODEL_VISION", MODEL_MAIN)
# 이미지 판독: vlm(이미지 입력 모델) | docparse(사내 Upstage Document Parse OCR → 텍스트 LLM 구조화)
ATTACHMENT_READER = os.getenv("ATTACHMENT_READER", "vlm" if "openrouter.ai" in BASE_URL else "docparse")
DOCPARSE_URL = os.getenv("DOCPARSE_URL", BASE_URL.rstrip("/") + "/document-parse")


def llm(model: str | None = None, **kw) -> ChatOpenAI:
    if not IS_OPENROUTER:
        kw.pop("reasoning_effort", None)  # vLLM·Solar 등 OpenAI 호환 서버는 이 인자를 모른다
    base_url = kw.pop("base_url", BASE_URL)
    return ChatOpenAI(
        model=model or MODEL_MAIN,
        base_url=base_url,
        api_key=_api_key(),
        temperature=0,
        timeout=90,
        max_retries=2,
        max_tokens=kw.pop("max_tokens", int(os.getenv("MAX_TOKENS", "2048"))),  # 기본값(모델 최대치)이면 OpenRouter가 크레딧 선확보로 402를 냄
        default_headers={"X-Title": "followups-agent-local"},
        **kw,
    )
