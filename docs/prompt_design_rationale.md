# Prompt Design Rationale

This document explains the two prompt templates we designed for the Generative AI part of the system.
Both templates turn the model results for one candidate into a short text for HR decision-makers.

## 1. Context

**Model:** `openai/gpt-oss-120b` on Groq (temperature 0.2). Setup details are in `Generative_AI/Phase2_Generative_AI.ipynb`, Section 1.

**System message:** both templates are sent with the same system message (defined in `Generative_AI/briefing.py`). It tells the model to use only the given numbers and facts, to keep the answer under 120 words, and to never use gender or any other personal trait as a reason. Because these rules are already in the system message, the templates focus only on the purpose, structure and style of the answer.

**Placeholders available to the templates** (filled by `fill_template()` in `briefing.py`):

| Placeholder | Meaning | Example value |
|---|---|---|
| `{probability}` | XGBoost predicted chance of looking for a job change | 63.5% |
| `{threshold}` | decision threshold of the final model | 0.32 |
| `{risk}` | risk level (High, Elevated or Low) | High |
| `{cluster}` | K-Means segment of the candidate | Segment 4 |
| `{cluster_rate}` | job-change rate of that segment | 50% |
| `{drivers}` | up to 5 main drivers of the score, with the candidate's value and direction | City development index: 0.62 (below average, average 0.83) -> raises risk (effect +0.85) |

## 2. Template 1: Decision Card

### Template Name & ID
**T1_decision_card** - Decision Card

### Intended Use Case
A busy HR executive who needs a quick answer for one candidate: is this person at risk, why, and what should HR do now. It fits a dashboard view or a short list of candidates that the executive scans quickly.

### Design Rationale
- **Fixed format:** the answer always has the same three labeled lines (Decision, Why, Action). The executive always knows where to find each piece of information, and outputs for different candidates are easy to compare.
- **Decision first:** the first line turns the probability into a clear decision ("act now" or "keep monitoring"), because a leader needs a decision, not only a number.
- **Strongest drivers only:** it asks for only the one or two strongest drivers, so the reason stays short and focused.
- **Very short:** under 50 words, much shorter than the 120-word limit, so it can be read in a few seconds.
- **No segment information:** this template does not use `{cluster}` or `{cluster_rate}` on purpose, to keep it simple and fast. This also makes it clearly different from Template 2.

### Prompt Structure
```
You are preparing a quick decision card for a busy HR executive.
Candidate result:
- Chance of looking for a job change: {probability} (alert threshold {threshold})
- Risk level: {risk}
- Main drivers:
{drivers}
Reply with exactly three lines and nothing else:
Decision: one sentence that states the risk level and whether HR should act now or keep monitoring.
Why: one sentence with the one or two strongest drivers.
Action: one concrete HR step.
Keep the whole card under 50 words. No headings, no bullet points, no extra notes.
```

### Example Input/Output
**Input:** test case `high_row651` from `Generative_AI/test_results/test_results.csv`: probability 63.5%, High risk, Segment 4 (segment rate 50%), top drivers: City development index: 0.62 (below average, average 0.83), Company information: missing.

**Output** (26 words):
```
High risk; HR should act now.
City development index low (0.62) and missing company information are the strongest risk drivers.
Initiate outreach to discuss retention opportunities.
```
Note: in this case the model kept the three lines but left out the "Decision:", "Why:" and "Action:" labels (see Section 7).

### Assumptions & Limitations
- Assumes the drivers are already sorted from strongest to weakest, so the model picks the right ones.
- Assumes the reader only needs a quick decision, not a full explanation.
- Not personalized to the candidate's segment, because it does not use the cluster information.
- Only one action is suggested, which may be too little for complex cases.
- The model may not always follow the exact three-line format or the 50-word limit.

## 3. Template 2: Segment-Aware Advisor

### Template Name & ID
**T2_segment_advisor** - Segment-Aware Advisor

### Intended Use Case
The HR leadership team when they plan retention actions. They need to know how risky the candidate is compared with similar candidates, the reasons behind the score, and a retention plan with a reason.

### Design Rationale
- **Role-based prompt:** the model acts as a "senior HR retention advisor", so the answer uses HR language and focuses on keeping the candidate.
- **Personalization with the segment:** the first part compares the candidate's chance with the job-change rate of their K-Means segment. This shows if the candidate is more or less at risk than similar candidates, which helps HR set priorities.
- **No new numbers:** the model is asked to say only "above" or "below" the segment rate and not to calculate anything, because language models can make arithmetic mistakes. This supports factual accuracy.
- **More detail:** it asks for two or three drivers and two HR actions, so it gives more information than Template 1.
- **Strategic rationale:** it asks the model to explain why the actions fit the candidate, so leaders understand the reason behind the advice, not only the advice.
- **Three labeled parts:** the answer stays organized even though it is longer.

### Prompt Structure
```
You are a senior HR retention advisor writing a risk summary for the HR leadership team.
Candidate result:
- Chance of looking for a job change: {probability} (alert threshold {threshold})
- Risk level: {risk}
- Segment: {cluster}, where {cluster_rate} of candidates looked for a job change
- Main drivers:
{drivers}
Write three short parts with these labels:
Segment comparison: say whether this candidate's chance is above or below the segment rate, and what that means for priority. Do not calculate any new numbers.
Key drivers: explain the two or three strongest drivers in plain words.
Retention plan: suggest two HR actions that fit this candidate's drivers and segment, and explain in one sentence why they fit.
```

### Example Input/Output
**Input:** the same test case as Template 1 (`high_row651`), so the two outputs can be compared: probability 63.5%, High risk, Segment 4 (segment rate 50%), top drivers: City development index: 0.62 (below average, average 0.83), Company information: missing.

**Output** (106 words):
```
Segment comparison:
The candidate's 63.5% chance of looking for a new job is above the Segment 4 rate of 50%, indicating a higher priority for retention focus.

Key drivers:
- Low city development index (0.62) - a below-average environment that increases turnover risk.
- Missing company information - lack of data signals uncertainty and raises risk.
- Very limited experience (< 1 year) - early-career stage contributes to higher mobility.

Retention plan:
1. Offer a structured onboarding and mentorship program to address the candidate's limited experience and build engagement.
2. Provide clear, personalized communication about company details and growth opportunities to mitigate uncertainty from missing information.
```

### Assumptions & Limitations
- `{cluster}` is only a segment number (for example "Segment 4") with no description, so the personalization is based only on the segment's job-change rate.
- Assumes the segment rate is a fair comparison for the candidate.
- The answer is longer, so it takes more time to read than Template 1.
- The suggested actions are general HR ideas from the model, not company-specific policies.
- The model may still add details that are not in the input; this is checked in the testing section.

## 4. Comparison of the Two Templates

| | T1: Decision Card | T2: Segment-Aware Advisor |
|---|---|---|
| Reader | busy HR executive | HR leadership team |
| Goal | quick decision | retention planning |
| Length | under 50 words | up to 120 words |
| Prompt style | fixed output format | role-based with segment comparison |
| Uses the segment | no | yes |
| Actions | one | two, with a reason |

## 5. Thought Process Behind Each Template

**Template 1 (Decision Card):** For the first template, I focused on what an HR executive needs to know quickly: whether the candidate is at risk, why, and what action HR should take. My original idea was similar to the draft template in the notebook, so I changed it into a fixed three-line format to make it different and easier to read. I also kept it under 50 words and removed the segment information to keep it simple and direct.

**Template 2 (Segment-Aware Advisor):** For the second template, I wanted to provide more details and make the advice more personalized. At first, I considered asking the model to explain its steps, but since the response has to be under 120 words, I decided to focus more on the actual advice. Instead, I used the segment information to compare the candidate with similar candidates. I also asked the model to use only "above" or "below" rather than calculate new numbers to avoid mistakes. Finally, I asked it to explain the reason behind each HR action so leaders can understand why each recommendation is suitable.

## 6. How Domain Knowledge Influenced the Design

- **Executives have little time:** HR leaders usually need a decision and a next step, not a long report. This led to the short, fixed format of Template 1.
- **Retention is the real goal:** the system is meant to help HR keep candidates, so both templates end with an HR action, and Template 2 asks for a retention plan.
- **Findings from Phase 1:** the EDA showed that city development index, experience and missing company information are among the strongest signals. The `{drivers}` placeholder brings these signals into the prompt, so the explanation is based on the same factors we found in the data.
- **Segments from Phase 2:** the K-Means segments have very different job-change rates (from about 14% to about 50%), so comparing a candidate with their segment gives HR useful context. This led to the segment comparison in Template 2.
- **Fairness in HR decisions:** HR decisions must not be based on personal traits. The system message forbids using gender or other personal traits, and the code removes gender from the drivers before the prompt is built.

## 7. Lessons Learned During Prompt Testing

Both templates were tested on four test cases (`high_row651`, `elevated_row242`, `low_row1668`, `borderline_row817`). The full results are in `Generative_AI/test_results/test_results.csv` and in Sections 4 and 5 of `Phase2_Generative_AI.ipynb`.

- **Format instructions are not always followed:** T1 asked for three lines starting with "Decision:", "Why:" and "Action:". The model kept the three lines in all four cases, but it left out the labels in two of them (`high_row651` and `elevated_row242`). A stricter instruction or a short example of the format could fix this.
- **Word limits worked well:** T1 stayed between 26 and 31 words (limit 50), and T2 stayed between 91 and 106 words (limit 120).
- **"Do not calculate any new numbers" worked:** in all T2 cases, the model compared the candidate with the segment using only "above" or "below", and the comparisons were correct, even in the close case (31.9% vs 30%).
- **"Plain words" was not fully followed:** T2 still showed the raw effect values (for example "+1.05"), which are not meaningful for an HR executive. A clearer instruction like "do not show effect values" would help.
- **Shorter is not always easier to read:** T1 is much shorter, but its readability grade (11.77) is almost the same as T2 (12.32), because the short sentences are packed with information.
- **Segment context made the advice more useful:** in `elevated_row242`, T2 showed that the candidate (55.3%) is far above their segment rate (14%), which T1 could not show. This supports choosing T2 as the final template.

## 8. References to Prompt Engineering Best Practices

- OpenAI, *Prompt engineering guide*: https://platform.openai.com/docs/guides/prompt-engineering (clear instructions, giving the model a role, specifying the output format and length).
- Anthropic, *Prompt engineering overview*: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview (being clear and direct, using role prompting, structuring the output).
- White, J. et al. (2023). *A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT*. arXiv:2302.11382 (persona pattern used in Template 2, template pattern used in Template 1).
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv:2201.11903 (we did not ask for step-by-step reasoning in the output, because the model already reasons internally and the answer must stay under 120 words).
