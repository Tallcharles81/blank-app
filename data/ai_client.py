import os

import anthropic

# The AI-powered layers this backs (data/news_scanner.py's continuous news
# scan, models/lineup_anomaly_review.py's post-build sanity check) are a
# SEPARATE, real, billed Anthropic API account from whatever session or
# product is running this code - never assumed to be free or already
# authenticated. A real ANTHROPIC_API_KEY is required (see .env.example);
# never hardcoded here per this project's own secret-handling convention.
DEFAULT_MODEL = "claude-opus-5"


def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set - the AI-powered news scanner and "
            "lineup anomaly reviewer both need a real Anthropic API key (see "
            ".env.example). This is unrelated to any API access the "
            "environment running this code might already have."
        )
    return anthropic.Anthropic(api_key=api_key)
