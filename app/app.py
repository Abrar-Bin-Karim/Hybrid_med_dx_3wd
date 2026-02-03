import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
from pathlib import Path

ARTIFACT_DIR = Path("artifacts")

# ---------- 3-Way Decision logic (kept inside app to avoid import issues) ----------
def apply_three_way(p_benign: pd.Series, alpha: float, beta: float) -> pd.Series:
    def decide(p):
        if p >= alpha:
            return "Confirm_Benign"
        elif p <= beta:
            return "Confirm_Malignant"
        else:
            return "Uncertain"
    return p_benign.apply(decide)

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    feature_names = json.load(open(ARTIFACT_DIR / "feature_names.json"))
    return model, scaler, feature_names

# ---------- SHAP waterfall helper ----------
def shap_waterfall(explainer, X_scaled, row_idx: int):
    shap_values = explainer(X_scaled)
    shap.plots.waterfall(shap_values[row_idx], max_display=12, show=False)
    fig = plt.gcf()
    return fig

def main():
    st.set_page_config(page_title="Medical Diagnosis Support (3WD + SHAP)", layout="wide")
    st.title("Hybrid Medical Diagnosis Support System")
    st.caption("Baseline ML + Three-Way Decisions (3WD) + SHAP Explainability")

    # Load artifacts
    if not ARTIFACT_DIR.exists():
        st.error("Missing /artifacts folder. Run: python src/train_and_save_artifacts.py")
        return

    model, scaler, feature_names = load_artifacts()

    # Sidebar thresholds
    st.sidebar.header("Three-Way Decision Thresholds")
    alpha = st.sidebar.slider("Alpha (Confirm Benign if p ≥ α)", 0.50, 0.99, 0.91, 0.01)
    beta  = st.sidebar.slider("Beta (Confirm Malignant if p ≤ β)", 0.01, 0.50, 0.19, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.write(f"Rule:")
    st.sidebar.write(f"• p ≥ {alpha:.2f} → Confirm_Benign")
    st.sidebar.write(f"• p ≤ {beta:.2f} → Confirm_Malignant")
    st.sidebar.write(f"• otherwise → Uncertain")

    # Choose input method
    st.header("Input Method")
    mode = st.radio("Choose input method", ["Upload CSV", "Manual input (single case)"], horizontal=True)

    # Build SHAP explainer background (lightweight)
    # For best practice, background should be training data; for demo, use small random baseline.
    # We'll create background from zeros to keep it self-contained.
    background = np.zeros((50, len(feature_names)))
    explainer = shap.LinearExplainer(model, background, feature_names=feature_names)

    # ---------- Mode A: CSV Upload ----------
    if mode == "Upload CSV":
        st.subheader("A) Upload CSV")
        st.write("CSV must contain the same 30 feature columns (same names) as your WDBC features.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to start (1 row or multiple rows).")
            return

        df = pd.read_csv(uploaded)

        # Validate
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error("CSV is missing required columns:")
            st.write(missing)
            return

        X = df[feature_names].copy()

        # Predict
        X_scaled = scaler.transform(X)
        p_benign = model.predict_proba(X_scaled)[:, 1]

        df_out = df.copy()
        df_out["p_benign"] = p_benign
        df_out["decision"] = apply_three_way(df_out["p_benign"], alpha=alpha, beta=beta)

        st.subheader("Predictions")
        st.dataframe(df_out[["p_benign", "decision"]], use_container_width=True)

        st.download_button(
            "Download Results CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="streamlit_predictions.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("SHAP Explanation (select a row)")
        row_idx = st.number_input("Row index", min_value=0, max_value=len(df_out)-1, value=0, step=1)

        st.write(
            f"Row **{row_idx}** | p(benign) = **{df_out.loc[row_idx,'p_benign']:.4f}** "
            f"| Decision = **{df_out.loc[row_idx,'decision']}**"
        )

        fig = shap_waterfall(explainer, X_scaled, int(row_idx))
        st.pyplot(fig, clear_figure=True)

    # ---------- Mode B: Manual Input (Single Case) ----------
    else:
        st.subheader("B) Manual Input (single case)")
        st.write("Fill the 30 feature values and click **Predict**.")

        # Optional: load example values from a file if you want (not required)
        example_path = Path("data/sample_input.csv")
        example_row = None
        if example_path.exists():
            try:
                ex = pd.read_csv(example_path)
                if all(c in ex.columns for c in feature_names) and len(ex) > 0:
                    example_row = ex.iloc[0][feature_names].to_dict()
            except:
                example_row = None

        use_example = st.checkbox("Use example values (if available)", value=False)

        with st.form("manual_form"):
            # Create 2-column layout to reduce scrolling pain
            col1, col2 = st.columns(2)
            inputs = {}

            half = len(feature_names) // 2
            left_feats = feature_names[:half]
            right_feats = feature_names[half:]

            with col1:
                for f in left_feats:
                    default = float(example_row[f]) if (use_example and example_row is not None) else 0.0
                    inputs[f] = st.number_input(f, value=default, format="%.6f")

            with col2:
                for f in right_feats:
                    default = float(example_row[f]) if (use_example and example_row is not None) else 0.0
                    inputs[f] = st.number_input(f, value=default, format="%.6f")

            submitted = st.form_submit_button("Predict")

        if not submitted:
            return

        X_one = pd.DataFrame([inputs], columns=feature_names)
        X_scaled = scaler.transform(X_one)

        p_benign = float(model.predict_proba(X_scaled)[:, 1][0])
        decision = apply_three_way(pd.Series([p_benign]), alpha=alpha, beta=beta).iloc[0]

        st.success(f"p(benign) = {p_benign:.4f} | Decision: {decision}")

        st.markdown("---")
        st.subheader("SHAP Explanation (this case)")
        fig = shap_waterfall(explainer, X_scaled, 0)
        st.pyplot(fig, clear_figure=True)

if __name__ == "__main__":
    main()
