import os
from baselines.agentic.litellm_model import LitellmModel


def _create_litellm_model(model_id: str, provider: str):
    """Helper to create LitellmModel with the right API key."""
    api_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together_ai": "TOGETHERAI_API_KEY",
    }
    api_key = os.getenv(api_key_map[provider])
    return LitellmModel(model=model_id, api_key=api_key)


class MODELS:
    """Your specific model constants."""

    class OPENAI:
        # OpenAI models are just strings
        GPT_5 = "gpt-5"
        GPT_5_MINI = "gpt-5-mini"
        GPT_4_1 = "gpt-4.1"
        GPT_4_1_MINI = "gpt-4.1-mini"

        # Utility models (for internal functions)
        GPT_4O_MINI = "gpt-4o-mini"

    class ANTHROPIC:
        # Anthropic models are LitellmModel instances
        CLAUDE_4_OPUS = _create_litellm_model(
            "anthropic/claude-opus-4-1-20250805", "anthropic"
        )
        CLAUDE_4_SONNET = _create_litellm_model(
            "anthropic/claude-sonnet-4-20250514", "anthropic"
        )

    class GOOGLE:
        # Google models are LitellmModel instances
        GEMINI_25_PRO = _create_litellm_model("gemini/gemini-2.5-pro", "google")
        GEMINI_25_FLASH = _create_litellm_model("gemini/gemini-2.5-flash", "google")

    class TOGETHER_AI:
        # Together AI models are LitellmModel instances
        GPT_OSS_20B = _create_litellm_model(
            "together_ai/openai/gpt-oss-20b", "together_ai"
        )
        GPT_OSS_120B = _create_litellm_model(
            "together_ai/openai/gpt-oss-120b", "together_ai"
        )
        GLM_45_AIR = _create_litellm_model(
            "together_ai/zai-org/GLM-4.5-Air-FP8", "together_ai"
        )
        KIMI_K2 = _create_litellm_model(
            "together_ai/moonshotai/Kimi-K2-Instruct-0905", "together_ai"
        )
        QWEN_25_72B = _create_litellm_model(
            "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo", "together_ai"
        )
        DEEPSEEK_V31 = _create_litellm_model(
            "together_ai/deepseek-ai/DeepSeek-V3.1", "together_ai"
        )
        DEEPSEEK_R1 = _create_litellm_model(
            "together_ai/deepseek-ai/DeepSeek-R1", "together_ai"
        )


__all__ = ["MODELS"]
