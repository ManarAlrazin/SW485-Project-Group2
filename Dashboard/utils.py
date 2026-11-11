"""Data, model and scoring helpers for the executive dashboard.

Everything the app needs is loaded here: the preprocessor and the supervised
model from Phase 1, the K-Means segmentation from Phase 2 (rebuilt with the
same settings as the notebook, which gives identical labels), and the
functions that turn a candidate profile into a prediction.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "Supervised_Learning" / "models"
RANDOM_STATE = 42
K_FINAL = 5  # chosen in Phase2_Unsupervised_Learning.ipynb

# Same mappings as Section 5.2 of Phase1_Data_Exploration.ipynb
COMPANY_SIZE = {"<10": 0, "10-49": 1, "50-99": 2, "100-500": 3, "500-999": 4,
                "1000-4999": 5, "5000-9999": 6, "10000+": 7}
EDUCATION = {"Primary School": 0, "High School": 1, "Graduate": 2, "Masters": 3, "Phd": 4}
GENDERS = ["Male", "Female", "Other", "Unknown"]
UNIVERSITY = ["no_enrollment", "Full time course", "Part time course", "Unknown"]
MAJORS = ["STEM", "Business Degree", "Humanities", "Arts", "Other", "No Major", "Unknown"]
COMPANY_TYPES = ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other", "Unknown"]

# Fill these in when the cluster profiles (cluster_profiles.txt) are ready.
# Key = cluster id, value = short executive name. Missing ids fall back to "Segment <id>".
CLUSTER_NAMES = {}

# Final XGBoost decision threshold agreed by the team (used when that model is available)
XGB_THRESHOLD = 0.32


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load preprocessor, data, K-Means segmentation and the available models."""
    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    train = pd.read_csv(ROOT / "Dataset" / "preprocessed_train.csv")
    test = pd.read_csv(ROOT / "Dataset" / "preprocessed_test.csv")
    full = pd.read_csv(ROOT / "Dataset" / "preprocessed_data.csv")
    raw = pd.read_csv(ROOT / "Dataset" / "clustered_data.csv")  # readable columns + cluster labels

    feats = [c for c in full.columns if c != "target"]
    kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=RANDOM_STATE).fit(full[feats])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(full[feats])

    models = {"Logistic Regression": {"model": joblib.load(MODELS_DIR / "model1_logistic_regression.joblib"),
                                      "threshold": 0.50}}
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        if "xgb" in p.name.lower() or "xgboost" in p.name.lower():
            models["XGBoost"] = {"model": joblib.load(p), "threshold": XGB_THRESHOLD}

    return {"pre": pre, "feats": feats, "train": train, "test": test, "full": full, "raw": raw,
            "kmeans": kmeans, "pca": pca, "models": models}


def cluster_name(cid: int) -> str:
    return CLUSTER_NAMES.get(int(cid), f"Segment {int(cid)}")


# ---------------------------------------------------------------- candidate -> features
def candidate_frame(c: dict) -> pd.DataFrame:
    """Turn a candidate profile (dict from the sidebar) into the columns the preprocessor expects.

    Mirrors the fixed steps in Section 5.2 of Phase1_Data_Exploration.ipynb.
    """
    size = COMPANY_SIZE.get(c["company_size"], np.nan)
    edu = EDUCATION.get(c["education_level"], np.nan)
    size_unknown = int(np.isnan(size))
    return pd.DataFrame([{
        "city_development_index": c["city_development_index"],
        "experience": c["experience"],
        "last_new_job": c["last_new_job"],
        "training_hours_log": np.log1p(c["training_hours"]),
        "company_size": size,
        "education_level": edu,
        "relevent_experience": int(c["relevent_experience"]),
        "company_size_unknown": size_unknown,
        "education_unknown": int(np.isnan(edu)),
        "company_info_missing": int(size_unknown == 1 and c["company_type"] == "Unknown"),
        "gender": c["gender"],
        "enrolled_university": c["enrolled_university"],
        "major_discipline": c["major_discipline"],
        "company_type": c["company_type"],
    }])


def encode(art, c: dict) -> pd.DataFrame:
    pre = art["pre"]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return pd.DataFrame(pre.transform(candidate_frame(c)), columns=names)[art["feats"]]


def predict_proba(art, model_name: str, c: dict) -> float:
    return float(art["models"][model_name]["model"].predict_proba(encode(art, c))[0, 1])


def assign_cluster(art, c: dict) -> int:
    return int(art["kmeans"].predict(encode(art, c))[0])


def sweep(art, model_name: str, c: dict, variable: str, values) -> pd.DataFrame:
    """Probability of leaving while one input changes and the rest stay fixed."""
    rows = []
    for v in values:
        cc = dict(c); cc[variable] = v
        rows.append({variable: v, "probability": predict_proba(art, model_name, cc)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- explanations & confidence
def key_drivers(art, model_name: str, c: dict, top: int = 8) -> pd.DataFrame:
    """Signed contribution of each feature to this candidate's score (log-odds).

    Gender is left out of the displayed drivers: it is a sensitive attribute and should not be
    presented to HR as a reason for a decision (it is still an input of the Phase 1 model).
    """
    x = encode(art, c)
    model = art["models"][model_name]["model"]
    if hasattr(model, "coef_"):  # linear model: coefficient x value
        contrib = pd.Series(model.coef_[0] * x.iloc[0].values, index=x.columns)
    else:  # tree model: native per-feature contributions
        try:
            import xgboost as xgb
            vals = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0][:-1]
            contrib = pd.Series(vals, index=x.columns)
        except Exception:
            contrib = pd.Series(getattr(model, "feature_importances_", np.zeros(x.shape[1])), index=x.columns)
    contrib = contrib[(contrib.abs() > 1e-9) & ~contrib.index.str.startswith("gender")]
    out = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top]
    return out.rename("contribution").reset_index().rename(columns={"index": "feature"})


def test_metrics(art, model_name: str) -> dict:
    """ROC-AUC on the held-out test set, plus precision/recall at the model's threshold."""
    info = art["models"][model_name]
    X, y = art["test"].drop(columns="target"), art["test"]["target"]
    p = info["model"].predict_proba(X)[:, 1]
    return {"auc": roc_auc_score(y, p), "proba": p, "y": y.values}


def threshold_metrics(m: dict, threshold: float) -> dict:
    pred = (m["proba"] >= threshold).astype(int)
    return {"precision": precision_score(m["y"], pred, zero_division=0),
            "recall": recall_score(m["y"], pred, zero_division=0),
            "flagged_share": pred.mean()}


def historical_hit_rate(m: dict, prob: float, width: float = 0.05):
    """Among test candidates with a similar score, the share who really looked for a job change."""
    mask = np.abs(m["proba"] - prob) <= width
    return (float(m["y"][mask].mean()), int(mask.sum())) if mask.sum() >= 20 else (None, int(mask.sum()))


def risk_level(prob: float, threshold: float) -> str:
    if prob >= max(threshold * 1.5, 0.6):
        return "High"
    return "Elevated" if prob >= threshold else "Low"


def cluster_table(art) -> pd.DataFrame:
    raw = art["raw"]
    t = raw.groupby("cluster_kmeans").agg(candidates=("target", "size"), job_change_rate=("target", "mean"),
                                          avg_experience=("experience", lambda s: s.replace({"<1": 0, ">20": 21}).astype(float).mean()),
                                          avg_cdi=("city_development_index", "mean"),
                                          avg_training_hours=("training_hours", "mean"))
    t["share"] = t["candidates"] / t["candidates"].sum()
    t.index.name = "cluster"
    return t.reset_index()
