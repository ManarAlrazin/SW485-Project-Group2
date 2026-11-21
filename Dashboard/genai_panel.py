"""AI briefing panel: the connection between the dashboard and the Generative AI module.

    generate_briefing(ctx, use_ai=True) -> (text, source)

`ctx` holds the model prediction, the segment and the main drivers (same keys that
Generative_AI/briefing.py expects). With use_ai=True the team's module is called
(Groq, see Generative_AI/api_config_template.py). If the key is missing or the call
fails, a clearly labelled template briefing is returned, so the panel is never empty.

API keys are never stored in the repository: locally they come from a `.env` file in the
repo root, on Streamlit Cloud from the app's Secrets (GROQ_API_KEY).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def template_briefing(ctx: dict) -> str:
    """Rule-based fallback briefing (no AI call)."""
    drivers = ", ".join(f"{d['label']} ({'raises' if d['value'] > 0 else 'lowers'} risk)" for d in ctx["drivers"][:3])
    action = {
        "High": "Prioritise a retention conversation and review the development plan for this candidate.",
        "Elevated": "Keep the candidate on the watch list and check in at the next review cycle.",
        "Low": "No immediate action; continue the normal development path.",
    }[ctx["risk"]]
    return (
        f"**Executive briefing**\n\n"
        f"- **Assessment:** the model estimates a **{ctx['probability']:.0%}** chance that this candidate "
        f"is looking for a job change (decision threshold {ctx['threshold']:.2f}) -> **{ctx['risk']} risk**.\n"
        f"- **Segment:** {ctx['cluster']}, where {ctx['cluster_rate']:.0%} of candidates looked for a change.\n"
        f"- **Main drivers:** {drivers}.\n"
        f"- **Suggested action:** {action}"
    )


def _load_streamlit_secret() -> None:
    """On Streamlit Cloud the key is in st.secrets; copy it to the environment for the GenAI module."""
    if os.getenv("GROQ_API_KEY"):
        return
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file: the module will look for .env instead


def generate_briefing(ctx: dict, use_ai: bool = True):
    """Return (briefing_text, source)."""
    fallback = template_briefing(ctx)
    if not use_ai:
        return fallback, "Template preview"

    _load_streamlit_secret()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from Generative_AI.api_config_template import MODEL_NAME, PROVIDER
        from Generative_AI.briefing import build_briefing, is_error
    except Exception as err:  # package missing (e.g. groq not installed)
        return fallback, f"Template (GenAI module unavailable: {type(err).__name__})"

    text = build_briefing(ctx)
    if is_error(text):
        reason = text.split(":", 1)[-1].strip()
        return fallback, f"Template (AI call not available: {reason[:140]})"
    return text, f"Generative AI ({MODEL_NAME} on {PROVIDER})"
