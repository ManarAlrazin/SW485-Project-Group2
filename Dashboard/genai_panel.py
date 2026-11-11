"""AI briefing panel.

The Generative AI module (Phase2_Generative_AI.ipynb / Generative_AI/) is being built by
other team members. This file is the single connection point:

    generate_briefing(context) -> (text, source)

`context` holds the model prediction, the segment and the main drivers. When the team's
GenAI function is available it is called here; until then a clearly labelled template
briefing is returned so the panel is never empty.

API keys must never be committed: the GenAI module reads them from a `.env` file.
"""


def _template_briefing(ctx: dict) -> str:
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


def generate_briefing(ctx: dict):
    """Return (briefing_text, source). Swap the body for the team's GenAI call when it is ready."""
    # TODO (GenAI integration): replace with the function from Generative_AI/, for example:
    #   from Generative_AI.briefing import build_briefing
    #   return build_briefing(ctx), "Generative AI"
    return _template_briefing(ctx), "Template (GenAI module not connected yet)"
