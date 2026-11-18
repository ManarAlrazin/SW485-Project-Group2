"""Settings for the Generative AI part (Phase 2).

This file has no secret in it. The API key is read from the environment.
Put the key in a file called .env in the repo root (see .env.example).
The .env file is in .gitignore, so it is never uploaded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# provider and model
# no Llama chat model was available on our Groq account, so we use this open-weight model
PROVIDER = "Groq"
MODEL_NAME = "openai/gpt-oss-120b"

# low temperature so the briefings stay stable between runs
TEMPERATURE = 0.2

# this model thinks before it answers, low effort keeps it fast and short
REASONING_EFFORT = "low"

# token limit for thinking and answer together (the answer itself is about 100 words)
MAX_TOKENS = 600

# stop waiting after this many seconds
TIMEOUT_SECONDS = 30

# name of the environment variable that holds the key
API_KEY_NAME = "GROQ_API_KEY"

# repo root (one level above this folder)
ROOT = Path(__file__).resolve().parents[1]


def get_api_key():
    """Return the key from the environment, or None if it is missing."""
    # load .env from the repo root, do not overwrite real environment variables
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv(API_KEY_NAME)
    if key is None or key.strip() == "" or key.strip() == "your_key_here":
        return None
    return key.strip()


def key_is_set():
    """True or False only, so the key itself is never shown."""
    return get_api_key() is not None
