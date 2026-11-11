"""Executive dashboard: job-change risk for trained data scientists.

Run from the repository root:
    streamlit run Dashboard/app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from genai_panel import generate_briefing
from utils import (COMPANY_SIZE, COMPANY_TYPES, EDUCATION, GENDERS, MAJORS, UNIVERSITY, assign_cluster,
                   candidate_frame, cluster_name, cluster_table, encode, historical_hit_rate, key_drivers,
                   load_artifacts, predict_proba, risk_level, sweep, test_metrics, threshold_metrics)

st.set_page_config(page_title="HR Retention Dashboard", page_icon=":bar_chart:", layout="wide")

RISK_COLORS = {"Low": "#2E8B57", "Elevated": "#E0A100", "High": "#C0392B"}
CLUSTER_COLORS = px.colors.qualitative.Set2

art = load_artifacts()


@st.cache_data(show_spinner=False)
def cached_test_metrics(model_name: str):
    return test_metrics(load_artifacts(), model_name)


@st.cache_data(show_spinner=False)
def cached_pca_sample():
    a = load_artifacts()
    idx = np.random.RandomState(42).choice(len(a["full"]), 4000, replace=False)
    pts = a["pca"].transform(a["full"][a["feats"]].iloc[idx])
    return pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1],
                         "cluster": a["kmeans"].labels_[idx]})


def pretty(feature: str) -> str:
    """Readable label for an encoded feature name."""
    base = {"city_development_index": "City development index", "experience": "Years of experience",
            "last_new_job": "Years since last job change", "training_hours_log": "Training hours",
            "company_size": "Company size", "education_level": "Education level",
            "relevent_experience": "Has relevant experience", "company_size_unknown": "Company size unknown",
            "education_unknown": "Education unknown", "company_info_missing": "No company information"}
    if feature in base:
        return base[feature]
    for prefix in ("gender", "enrolled_university", "major_discipline", "company_type"):
        if feature.startswith(prefix + "_"):
            return f"{prefix.replace('_', ' ').capitalize()}: {feature[len(prefix) + 1:]}"
    return feature


# ---------------------------------------------------------------- sidebar: candidate + model
with st.sidebar:
    st.header("Candidate profile")
    cdi = st.slider("City development index", 0.45, 0.95, 0.80, 0.01)
    experience = st.slider("Years of experience (21 = more than 20)", 0, 21, 5)
    last_job = st.slider("Years since last job change (5 = more than 4, 0 = never)", 0, 5, 1)
    hours = st.slider("Training hours", 1, 336, 50)
    company_size = st.selectbox("Company size", list(COMPANY_SIZE) + ["Unknown"], index=2)
    company_type = st.selectbox("Company type", COMPANY_TYPES, index=0)
    education = st.selectbox("Education level", list(EDUCATION) + ["Unknown"], index=2)
    relevant = st.checkbox("Has relevant experience", value=True)
    major = st.selectbox("Major discipline", MAJORS, index=0)
    university = st.selectbox("Enrolled university", UNIVERSITY, index=0)
    gender = st.selectbox("Gender", GENDERS, index=0)

    st.header("Model")
    model_name = st.selectbox("Prediction model", list(art["models"]))
    default_thr = art["models"][model_name]["threshold"]
    threshold = st.slider("Decision threshold", 0.10, 0.90, float(default_thr), 0.01,
                          help="Candidates at or above this probability are flagged as likely to look for a job change.")

candidate = {"city_development_index": cdi, "experience": experience, "last_new_job": last_job,
             "training_hours": hours, "company_size": company_size, "company_type": company_type,
             "education_level": education, "relevent_experience": relevant, "major_discipline": major,
             "enrolled_university": university, "gender": gender}

prob = predict_proba(art, model_name, candidate)
risk = risk_level(prob, threshold)
cluster_id = assign_cluster(art, candidate)
ctable = cluster_table(art)
cluster_rate = float(ctable.loc[ctable["cluster"] == cluster_id, "job_change_rate"].iloc[0])
drivers = key_drivers(art, model_name, candidate)
tm = cached_test_metrics(model_name)
tmt = threshold_metrics(tm, threshold)

st.title("Data Scientist Retention: Executive Dashboard")
st.caption("Predicts which trained candidates are likely to look for a job change, groups candidates into "
           "segments, and explains the result for HR leadership. Decision support only.")

tab_sum, tab_pred, tab_seg, tab_scn, tab_ai = st.tabs(
    ["Executive summary", "Candidate prediction", "Segments", "Scenario analysis", "AI briefing"])

# ---------------------------------------------------------------- tab 1: executive summary
with tab_sum:
    raw = art["raw"]
    top = ctable.sort_values("job_change_rate", ascending=False).iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Candidates analysed", f"{len(raw):,}")
    k2.metric("Observed job-change rate", f"{raw['target'].mean():.1%}")
    k3.metric("Highest-risk segment", cluster_name(top["cluster"]), f"{top['job_change_rate']:.0%} job-change rate",
              delta_color="inverse")
    k4.metric("Model ROC-AUC (test set)", f"{tm['auc']:.3f}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Recall at threshold", f"{tmt['recall']:.0%}", help="Share of real job-changers the model catches.")
    k6.metric("Precision at threshold", f"{tmt['precision']:.0%}", help="Share of flagged candidates who really look for a change.")
    k7.metric("Candidates flagged", f"{tmt['flagged_share']:.0%}", help="Share of the test set flagged at this threshold.")
    k8.metric("Decision threshold", f"{threshold:.2f}")
    st.caption(f"Active model: **{model_name}**. Metrics are computed on the held-out test set "
               f"({len(art['test']):,} candidates).")

    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(ctable.assign(segment=[cluster_name(c) for c in ctable["cluster"]]), x="segment",
                     y="job_change_rate", color="segment", color_discrete_sequence=CLUSTER_COLORS,
                     labels={"job_change_rate": "Job-change rate", "segment": ""},
                     title="Job-change rate by segment")
        fig.add_hline(y=raw["target"].mean(), line_dash="dash", annotation_text="overall rate")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False, height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.pie(ctable.assign(segment=[cluster_name(c) for c in ctable["cluster"]]), names="segment",
                     values="candidates", hole=0.5, color_discrete_sequence=CLUSTER_COLORS,
                     title="Candidates per segment")
        fig.update_layout(height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 2: candidate prediction
with tab_pred:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of looking for a change", f"{prob:.1%}")
    c2.metric("Risk level", risk)
    c3.metric("Model decision", "Flag as at-risk" if prob >= threshold else "Not flagged")
    c4.metric("Segment", cluster_name(cluster_id), f"{cluster_rate:.0%} of the segment changed jobs",
              delta_color="off")

    g, d = st.columns(2)
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[risk]},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}}))
        gauge.update_layout(height=320, margin=dict(t=60, b=10))
        st.plotly_chart(gauge, width="stretch")
        rate, n = historical_hit_rate(tm, prob)
        if rate is None:
            st.info("Too few similar candidates in the test set to estimate how reliable this score is.")
        else:
            st.info(f"**Prediction confidence:** among {n} test-set candidates with a similar score, "
                    f"**{rate:.0%}** actually looked for a job change.")
    with d:
        dr = drivers.assign(label=drivers["feature"].map(pretty))
        dr["direction"] = np.where(dr["contribution"] > 0, "Raises risk", "Lowers risk")
        fig = px.bar(dr.iloc[::-1], x="contribution", y="label", orientation="h", color="direction",
                     color_discrete_map={"Raises risk": "#C0392B", "Lowers risk": "#2E8B57"},
                     labels={"contribution": "Effect on the score", "label": ""}, title="Key drivers for this candidate")
        fig.update_layout(height=380, margin=dict(t=50, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 3: segments
with tab_seg:
    st.markdown("Candidates are grouped by K-Means (k = 5) from the Phase 2 notebook. "
                "The star marks the candidate selected in the sidebar.")
    pts = cached_pca_sample().assign(segment=lambda d: d["cluster"].map(cluster_name))
    fig = px.scatter(pts, x="PC1", y="PC2", color="segment", opacity=0.45,
                     color_discrete_sequence=CLUSTER_COLORS, title="Segments projected to 2D (PCA)")
    xy = art["pca"].transform(encode(art, candidate))[0]
    fig.add_trace(go.Scatter(x=[xy[0]], y=[xy[1]], mode="markers", name="Selected candidate",
                             marker=dict(symbol="star", size=24, color="gold", line=dict(color="black", width=2))))
    fig.update_layout(height=480, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    show = ctable.assign(segment=[cluster_name(c) for c in ctable["cluster"]]).set_index("segment")
    st.dataframe(show[["candidates", "share", "job_change_rate", "avg_experience", "avg_cdi", "avg_training_hours"]]
                 .rename(columns={"candidates": "Candidates", "share": "Share", "job_change_rate": "Job-change rate",
                                  "avg_experience": "Avg. experience (yrs)", "avg_cdi": "Avg. city dev. index",
                                  "avg_training_hours": "Avg. training hours"})
                 .style.format({"Share": "{:.1%}", "Job-change rate": "{:.1%}", "Avg. experience (yrs)": "{:.1f}",
                                "Avg. city dev. index": "{:.2f}", "Avg. training hours": "{:.0f}"}),
                 width="stretch")
    st.caption("Note: the clusters overlap (silhouette about 0.14), so segments are a guide for prioritising "
               "attention, not sharp categories.")

# ---------------------------------------------------------------- tab 4: scenario analysis
with tab_scn:
    st.markdown("See how the risk changes when one factor changes and everything else stays as in the sidebar.")
    options = {"Training hours": ("training_hours", np.arange(1, 337, 5)),
               "Years of experience": ("experience", np.arange(0, 22)),
               "City development index": ("city_development_index", np.round(np.arange(0.45, 0.951, 0.01), 2)),
               "Years since last job change": ("last_new_job", np.arange(0, 6))}
    choice = st.selectbox("Factor to vary", list(options), index=2)
    var, values = options[choice]
    curve = sweep(art, model_name, candidate, var, values)
    fig = px.line(curve, x=var, y="probability", labels={var: choice, "probability": "Probability"})
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="threshold")
    fig.add_trace(go.Scatter(x=[candidate[var]], y=[prob], mode="markers", name="Current",
                             marker=dict(size=12, color="black")))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=380, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("What-if comparison")
    s1, s2, s3 = st.columns(3)
    new_hours = s1.slider("Scenario: training hours", 1, 336, hours, key="s_hours")
    new_cdi = s2.slider("Scenario: city development index", 0.45, 0.95, cdi, 0.01, key="s_cdi")
    new_exp = s3.slider("Scenario: years of experience", 0, 21, experience, key="s_exp")
    scenario = dict(candidate, training_hours=new_hours, city_development_index=new_cdi, experience=new_exp)
    p2 = predict_proba(art, model_name, scenario)
    cl2 = assign_cluster(art, scenario)
    a, b, c_ = st.columns(3)
    a.metric("Current probability", f"{prob:.1%}")
    b.metric("Scenario probability", f"{p2:.1%}", f"{(p2 - prob) * 100:+.1f} pts", delta_color="inverse")
    c_.metric("Scenario segment", cluster_name(cl2), "same segment" if cl2 == cluster_id else "segment changes",
              delta_color="off")

# ---------------------------------------------------------------- tab 5: AI briefing
with tab_ai:
    ctx = {"probability": prob, "threshold": threshold, "risk": risk, "cluster": cluster_name(cluster_id),
           "cluster_rate": cluster_rate, "model": model_name,
           "drivers": [{"label": pretty(r.feature), "value": float(r.contribution)} for r in drivers.itertuples()],
           "candidate": candidate}
    text, source = generate_briefing(ctx)
    st.markdown(text)
    st.caption(f"Source: {source}")

st.divider()
st.caption("Predictions describe patterns in historical data and should support, not replace, human judgement. "
           "Gender is an input of the Phase 1 model but is not shown among the key drivers; review its influence "
           "before acting on a score.")
