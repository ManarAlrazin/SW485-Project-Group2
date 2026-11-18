"""Executive briefing with an open-weight model on Groq.

Main function: build_briefing(ctx, template=None) -> str

ctx is the same dict the dashboard sends in Dashboard/genai_panel.py:
    probability, threshold, risk, cluster, cluster_rate,
    drivers (list of dicts with "label" and "value")
Other keys (like "model" or "candidate") are ignored and never sent to the model.
"""
try:
    from .api_config_template import (MAX_TOKENS, MODEL_NAME, REASONING_EFFORT,
                                      TEMPERATURE, TIMEOUT_SECONDS, get_api_key)
except ImportError:
    from api_config_template import (MAX_TOKENS, MODEL_NAME, REASONING_EFFORT,
                                     TEMPERATURE, TIMEOUT_SECONDS, get_api_key)

# every error message starts with this, so tests can find failed calls
ERROR_PREFIX = "Briefing not available:"

# how many drivers we send to the model
MAX_DRIVERS = 5

SYSTEM_MESSAGE = (
    "You write short briefings for an HR executive about one candidate's risk of "
    "looking for a job change. "
    "Use only the numbers and facts given in the message. Do not invent any facts, "
    "numbers, names or reasons. If something is not given, do not guess it. "
    "Each driver shows the candidate's real value. Describe the value exactly as given; "
    "do not infer it from the direction. "
    "Keep it short and clear: at most 120 words, plain business English. "
    "Never use gender, or any other personal trait like age, religion or nationality, "
    "as a reason or a recommendation. "
    "The prediction is a support tool, not a final decision."
)

# DRAFT - for testing the framework only, to be replaced by the final templates
DRAFT_TEMPLATE = (
    "Write an executive briefing for this candidate.\n"
    "Predicted chance of looking for a job change: {probability} "
    "(decision threshold {threshold}).\n"
    "Risk level: {risk}.\n"
    "Segment: {cluster}, where {cluster_rate} of candidates looked for a job change.\n"
    "Main drivers of the score:\n{drivers}\n"
    "Give: 1) the assessment in one sentence, 2) the main reasons, "
    "3) one suggested HR action."
)


class _KeepMissing(dict):
    """Leaves unknown placeholders as they are instead of failing."""

    def __missing__(self, key):
        return "{" + key + "}"


def _format_drivers(drivers):
    """Turn the drivers list into short text lines.

    The label is used as given (it can already hold the candidate's real value),
    and "value" is the effect on the score, so its sign gives the direction.
    """
    lines = []
    for d in drivers or []:
        label = str(d.get("label", ""))
        # safety check: never pass gender to the model
        if label.lower().startswith("gender"):
            continue
        value = float(d.get("value", 0.0))
        direction = "raises risk" if value > 0 else "lowers risk"
        lines.append(f"- {label} -> {direction} (effect {value:+.2f})")
        if len(lines) == MAX_DRIVERS:
            break
    if not lines:
        return "- no drivers given"
    return "\n".join(lines)


def fill_template(ctx, template=None):
    """Put the ctx values into the template placeholders."""
    if template is None:
        template = DRAFT_TEMPLATE
    values = _KeepMissing(
        # one decimal, so 31.9% is not rounded up to the 0.32 threshold
        probability=f"{float(ctx['probability']):.1%}",
        threshold=f"{float(ctx['threshold']):.2f}",
        risk=str(ctx["risk"]),
        cluster=str(ctx["cluster"]),
        cluster_rate=f"{float(ctx['cluster_rate']):.0%}",
        drivers=_format_drivers(ctx.get("drivers")),
    )
    return template.format_map(values)


def _clean_error(err, key):
    """Short error text, with the key removed just in case."""
    text = f"{type(err).__name__}: {err}"
    if key:
        text = text.replace(key, "[hidden]")
    return text[:300]


def build_briefing(ctx, template=None, model=None):
    """Fill the template, call openai/gpt-oss-120b on Groq and return the briefing text.

    Never raises. On any problem it returns a message that starts with ERROR_PREFIX.
    """
    # build the prompt first, so a bad ctx gives a clear message
    try:
        user_message = fill_template(ctx, template)
    except (KeyError, TypeError, ValueError) as err:
        return f"{ERROR_PREFIX} the input is missing a value ({type(err).__name__}: {err})."

    key = get_api_key()
    if key is None:
        return (f"{ERROR_PREFIX} GROQ_API_KEY is not set. "
                "Create a .env file in the repo root with the line GROQ_API_KEY=<your key>.")

    try:
        from groq import Groq
    except ImportError:
        return f"{ERROR_PREFIX} the groq package is not installed (pip install groq)."

    try:
        client = Groq(api_key=key, timeout=TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            # only return the answer, not the model's thinking
            include_reasoning=False,
        )
        text = response.choices[0].message.content
    except Exception as err:
        return f"{ERROR_PREFIX} the API call failed ({_clean_error(err, key)})."

    if not text or not text.strip():
        return f"{ERROR_PREFIX} the model returned an empty answer."
    return _plain_text(text).strip()


def _plain_text(text):
    """Replace special hyphens and spaces with normal ones."""
    for odd, normal in {"‑": "-", "‐": "-", "–": "-", "—": "-",
                        " ": " ", " ": " "}.items():
        text = text.replace(odd, normal)
    return text


def is_error(text):
    """True if build_briefing returned an error message."""
    return str(text).startswith(ERROR_PREFIX)
