import kagglehub

# Download latest version of the dataset (preserve existing behavior)
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")

print("Path to dataset files:", path)


# --- New: Streamlit radar chart UI for patient risk profile ---
try:
    import streamlit as st
    import plotly.graph_objects as go
except Exception:
    # if Streamlit or plotly aren't installed, don't break the original behavior
    st = None


def make_radar_figure(labels, values, title="Health Risk Radar"):
    # Ensure the radar closes (first value repeated at end)
    vals = list(values)
    if len(vals) > 0:
        vals.append(vals[0])
    labs = list(labels)
    if len(labs) > 0:
        labs.append(labs[0])

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=vals, theta=labs, fill='toself', name='Risk Profile', marker=dict(color='crimson'))
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=title
    )
    return fig


def streamlit_radar_ui():
    if st is None:
        print("Streamlit or plotly not available; radar UI skipped.")
        return

    st.set_page_config(page_title="Health Risk Radar", layout="centered")
    st.title("Health Risk Radar — Patient Risk Dashboard")

    st.markdown("Visualize the patient's risk profile across key metrics.")

    with st.sidebar:
        st.header("Patient Metrics (0-100)")
        age_score = st.slider("Age risk score", 0, 100, 40)
        cholesterol_score = st.slider("Cholesterol risk score", 0, 100, 50)
        ecg_score = st.slider("ECG abnormality score", 0, 100, 20)
        bp_score = st.slider("Blood pressure risk score", 0, 100, 60)
        # model predicted probability as a decimal 0.0-1.0 but visualize as 0-100
        model_prob = st.slider("Model predicted probability", 0.0, 1.0, 0.27, step=0.01)

    labels = [
        "Age risk",
        "Cholesterol risk",
        "ECG abnormality",
        "Blood pressure",
        "Model probability"
    ]

    values = [age_score, cholesterol_score, ecg_score, bp_score, model_prob * 100]

    st.subheader("Patient Radar Chart")
    fig = make_radar_figure(labels, values)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interpretation:** Higher values indicate higher relative risk for that axis.\n\nUse the sidebar sliders to simulate or inspect a patient's profile.")


if __name__ == "__main__":
    # If user runs this file directly, run the Streamlit UI when possible.
    try:
        streamlit_radar_ui()
    except Exception as e:
        # keep original behavior intact; print error but don't raise
        print("Streamlit radar UI failed to start:", e)
# streamlit_upgrade.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import requests
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Predictive Disease Detector", layout="wide")

# -------------------------
# CONFIG
# -------------------------
API_URL = "http://127.0.0.1:8000/predict"   # optional; we call local model directly for better UX
MODEL_DIR = "models"
HISTORY_PATH = "data/historical_treatments.csv"  # should contain features + 'treatment' + 'outcome' columns
DEFAULT_DISEASES = {
    "Heart (UCI Cleveland)": {
        "model": os.path.join(MODEL_DIR, "xgb_heart_pipeline.joblib"),
        "shap_bg": os.path.join(MODEL_DIR, "shap_bg_heart.joblib")
    },
    # Add more diseases here when you have models:
    # "Diabetes (Pima)": {"model": "models/xgb_diabetes_pipeline.joblib", "shap_bg": "models/shap_bg_diabetes.joblib"},
}

# -------------------------
# Helpers
# -------------------------
def load_model_and_info(disease_cfg):
    model_path = disease_cfg["model"]
    shap_bg_path = disease_cfg["shap_bg"]
    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"
    pipeline = joblib.load(model_path)
    bg = None
    if os.path.exists(shap_bg_path):
        bg = joblib.load(shap_bg_path)
    # try fetch feature names (imputer or scaler)
    feature_names = None
    try:
        feature_names = list(pipeline.named_steps['imputer'].feature_names_in_)
    except Exception:
        try:
            feature_names = list(pipeline.named_steps['scaler'].feature_names_in_)
        except Exception:
            feature_names = None
    return pipeline, bg, feature_names

def ensure_history(feature_names):
    """
    Ensure a historical treatments CSV exists. If none is present, create a synthetic one
    with columns matching feature_names plus 'treatment' and 'outcome'.
    """
    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        return hist
    # create synthetic history for demo
    np.random.seed(42)
    n = 200
    data = {}
    for i, fn in enumerate(feature_names):
        # make plausible ranges
        if fn in ["age"]:
            data[fn] = np.random.randint(30, 85, size=n)
        elif fn in ["sex","fbs","exang","ca"]:
            data[fn] = np.random.randint(0, 2, size=n)
        elif fn in ["cp","restecg","slope","thal"]:
            data[fn] = np.random.randint(0, 4, size=n)
        else:
            data[fn] = np.random.normal(loc=100, scale=30, size=n)
    # Add treatments (three example treatments) and outcomes (1=good outcome)
    treatments = ["MedA + Lifestyle", "MedB", "Surgery", "Lifestyle only", "MedA + MedB"]
    data["treatment"] = np.random.choice(treatments, size=n)
    # outcome correlated with random feature to simulate signal
    data["outcome"] = (np.random.rand(n) + (np.array(data.get("age", np.zeros(n))) < 60).astype(float) * 0.2) > 0.5
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)
    return df

def recommend_treatments(history_df, input_row, feature_names, k=10):
    """
    Returns a ranked list of recommended treatments based on nearest neighbors
    in feature space and aggregated historical outcomes.
    """
    # Ensure columns
    X_hist = history_df[feature_names].copy()
    # numeric conversion (coerce non-numeric)
    X_hist = X_hist.apply(pd.to_numeric, errors="coerce").fillna(0)
    input_vec = np.array([input_row[fn] if fn in input_row else 0 for fn in feature_names], dtype=float).reshape(1, -1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_hist)
    xi = scaler.transform(input_vec)
    # k-NN
    nn = NearestNeighbors(n_neighbors=min(k, len(X_hist)), metric="euclidean")
    nn.fit(Xs)
    dists, idx = nn.kneighbors(xi)
    neighbors = history_df.iloc[idx[0]].copy()
    # aggregate treatments by average outcome among neighbors
    agg = neighbors.groupby("treatment").agg(
        count=("outcome", "count"),
        success_rate=("outcome", lambda x: float(np.mean(x)))
    ).reset_index().sort_values(["success_rate", "count"], ascending=[False, False])
    return agg, neighbors

def create_pdf_report(inputs, disease_name, prediction, prob, shap_series, treatment_table):
    """
    Build a PDF bytes object with the report summary
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Predictive Report — {disease_name}")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Prediction: {prediction}   Probability: {prob:.3f}")
    y -= 20

    c.drawString(margin, y, "Input features:")
    y -= 18
    for k, v in inputs.items():
        c.drawString(margin + 10, y, f"{k}: {v}")
        y -= 14
        if y < 100:
            c.showPage()
            y = height - margin

    y -= 6
    c.drawString(margin, y, "Top SHAP feature contributions:")
    y -= 18
    for k, v in shap_series.head(8).items():
        c.drawString(margin + 10, y, f"{k}: {v:.4f}")
        y -= 14
        if y < 100:
            c.showPage()
            y = height - margin

    y -= 6
    c.drawString(margin, y, "Top recommended treatments (from similar patients):")
    y -= 18
    # treatment_table: DataFrame with treatment, count, success_rate
    for _, row in treatment_table.head(8).iterrows():
        c.drawString(margin + 10, y, f"{row['treatment']} — success_rate: {row['success_rate']:.2f} ({int(row['count'])} cases)")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - margin

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# UI
# -------------------------
st.markdown("<h1 style='text-align:center'>Predictive Disease Detection Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Early detection + personalized treatment suggestions (demo)</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([2,1])
with col1:
    disease_choice = st.selectbox("Choose disease model", list(DEFAULT_DISEASES.keys()))
    disease_cfg = DEFAULT_DISEASES[disease_choice]
    pipeline, shap_bg, feature_names = load_model_and_info(disease_cfg)
    if pipeline is None:
        st.error(shap_bg)  # message returned when model missing
        st.stop()
    st.success(f"Loaded model for: {disease_choice}")
    st.write("Model features detected:" , feature_names)

with col2:
    st.metric("Dataset size (rows)", value="297 (example)")
    st.metric("Demo mode", value="Not clinical")

st.write("## Patient input")

# Build input form dynamically from feature_names (fall back to common set)
if feature_names is None:
    feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

input_vals = {}
cols = st.columns(3)
for i, fname in enumerate(feature_names):
    col = cols[i % 3]
    if fname in ["sex","fbs","exang","ca"]:
        input_vals[fname] = int(col.selectbox(fname, options=[0,1], index=0))
    elif fname in ["cp","restecg","slope","thal"]:
        # small-range integers
        input_vals[fname] = int(col.number_input(fname, min_value=0, max_value=4, value=0))
    elif fname in ["age", "trestbps", "chol", "thalach"]:
        input_vals[fname] = float(col.number_input(fname, min_value=0, max_value=300, value=50))
    else:
        input_vals[fname] = float(col.number_input(fname, min_value=0.0, max_value=1000.0, value=0.0))

st.write("---")

if st.button("Get prediction & recommendations"):
    with st.spinner("Computing prediction..."):
        # Prepare DataFrame and predict locally (faster and no network needed)
        X = pd.DataFrame([input_vals], columns=feature_names)
        # pipeline may expect exact columns & preprocessing
        try:
            prob = float(pipeline.predict_proba(X)[:,1][0])
            pred = int(prob >= 0.5)
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.stop()

        # SHAP
        try:
            imputer = pipeline.named_steps['imputer']
            scaler = pipeline.named_steps['scaler']
            model = pipeline.named_steps['model']
            X_t = scaler.transform(imputer.transform(X))
            bg = shap_bg if shap_bg is not None else X_t
            explainer = shap.TreeExplainer(model, data=bg)
            shap_values = explainer.shap_values(X_t)
            shap_series = pd.Series(shap_values[0], index=feature_names).sort_values(ascending=False)
        except Exception as e:
            shap_series = pd.Series([0]*len(feature_names), index=feature_names)
            st.warning("Could not compute SHAP: " + str(e))

        # Load or synthesize history and compute recommendations
        history_df = ensure_history(feature_names)
        agg, neighbors = recommend_treatments(history_df, input_vals, feature_names, k=30)

    # Display results nicely
    left, right = st.columns([2,1])
    with left:
        st.subheader("Prediction")
        st.write(f"**Risk:** {pred}   —   **Probability:** {prob:.3f}")
        st.subheader("Top feature contributions (SHAP)")
        st.table(shap_series.head(8).reset_index().rename(columns={"index":"feature", 0:"shap_value"}))
    with right:
        st.subheader("Recommended treatments (from similar patients)")
        if agg.shape[0] == 0:
            st.write("No treatment history available.")
        else:
            st.table(agg.head(6).reset_index(drop=True))
        st.info("Recommendations show treatments used by clinically similar patients and their historical success rates (demo).")

    # Save PDF and provide download
    pdf_buffer = create_pdf_report(input_vals, disease_choice, pred, prob, shap_series, agg)
    st.download_button("Download PDF report", data=pdf_buffer, file_name="report.pdf", mime="application/pdf")

    # Optional: show nearest neighbor examples (de-identified)
    with st.expander("Show similar patient examples (de-identified)"):
        if neighbors is not None and not neighbors.empty:
            st.dataframe(neighbors.head(10))
        else:
            st.write("No neighbors to show.")

st.write("---")
st.caption("Note: This demo is for educational purposes only and is not a medical device.")
