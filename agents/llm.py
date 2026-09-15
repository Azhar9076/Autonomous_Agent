"""
Central place that builds the chat model used by every agent.

Reads LLM_PROVIDER / LLM_MODEL from the environment (see .env.example)
so students can swap providers without touching any agent code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "grok": "grok-4.6",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}

PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, "openai/gpt-oss-120b"))


def get_llm(temperature: float = 0.4):
    """Return a LangChain chat model configured from the environment.

    Supports four providers via LLM_PROVIDER in .env: "groq" (default),
    "grok", "anthropic", "openai". Groq and Grok both have OpenAI-compatible
    APIs, so both just use ChatOpenAI pointed at the right base URL — don't
    mix them up, they're different companies (see README).
    """
    if PROVIDER == "groq":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "Groq API key (get one at https://console.groq.com)."
            )
        return ChatOpenAI(
            model=MODEL,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    if PROVIDER == "grok":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY is not set. Copy .env.example to .env and add your "
                "Grok (xAI) API key (get one at https://console.x.ai)."
            )
        return ChatOpenAI(
            model=MODEL,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return ChatOpenAI(model=MODEL, temperature=temperature, api_key=api_key)

    # anthropic
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatAnthropic(model=MODEL, temperature=temperature, api_key=api_key)
