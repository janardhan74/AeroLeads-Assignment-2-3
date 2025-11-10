import os
from typing import Any

LLM = Any


def load_llm(provider: str | None = None) -> LLM:
    """Instantiate an LLM via LangChain wrappers, based on env/provider."""
    provider = (provider or os.getenv("AI_PROVIDER", "gemini")).lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("MODEL_NAME_GEMINI", "gemini-2.0-flash-001")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.6)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.6)

    if provider == "perplexity":
        # Community wrapper; requires PPLX_API_KEY
        from langchain_community.chat_models import ChatPerplexity
        model_name = os.getenv("MODEL_NAME_PPLX", "sonar-pro")
        api_key = os.getenv("PPLX_API_KEY")
        if not api_key:
            raise RuntimeError("PPLX_API_KEY missing")
        # NOTE: Perplexity is research-focused; set search_disabled=True for pure writing.
        return ChatPerplexity(model=model_name, temperature=0.6, pplx_api_key=api_key, search_disabled=True)

    raise ValueError(f"Unknown AI_PROVIDER: {provider}")
