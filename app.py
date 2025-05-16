import streamlit as st
import pandas as pd
import joblib

# ——————————————
# 1) Load model and threshold once
# ——————————————
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("rf_response_pipeline.pkl")
    threshold = float(open("response_threshold.txt").read())
    return pipeline, threshold

pipeline, threshold = load_artifacts()

# ——————————————
# 2) App header
# ——————————————
st.title("🎯 Marketing Response Predictor")
st.markdown("Predict which customers will respond to the campaign.")

# ——————————————
# 3) Choose mode: single or batch
# ——————————————
mode = st.radio("Select input mode", ["Single customer", "Batch via CSV"])

# ——————————————
# 4) Single customer input
# ——————————————
if mode == "Single customer":
    st.subheader("Enter customer data")

    # Only ask for a handful of key features
    Recency             = st.number_input("Recency",               min_value=0.0, value=10.0)
    NumCatalogPurchases = st.number_input("NumCatalogPurchases",   min_value=0,   value=2)
    NumWebVisitsMonth   = st.number_input("NumWebVisitsMonth",     min_value=0,   value=5)
    Spending            = st.number_input("Spending",              min_value=0.0, value=3.2)
    # … add more inputs here as needed …

    if st.button("Predict"):
        # Retrieve the exact training feature names
        rf = pipeline.named_steps['rf']
        feature_names = list(rf.feature_names_in_)

        # Create a zero-filled record for all features
        record = {feat: 0 for feat in feature_names}

        # Overwrite only the features we collected
        record['Recency']             = Recency
        record['NumCatalogPurchases'] = NumCatalogPurchases
        record['NumWebVisitsMonth']   = NumWebVisitsMonth
        record['Spending']            = Spending
        # … and so on for any other inputs …

        # Build a single-row DataFrame in the correct order
        df = pd.DataFrame([record], columns=feature_names)

        # Predict probability and class label
        proba = pipeline.predict_proba(df)[:, 1][0]
        label = int(proba >= threshold)

        # Display results
        st.write("**Will respond?**", "✅ Yes" if label else "❌ No")
        st.write(f"**Probability:** {proba:.3f}")
        st.write(f"**Threshold used:** {threshold:.3f}")

# ——————————————
# 5) Batch CSV scoring
# ——————————————
else:
    st.subheader("Upload a CSV for batch scoring")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        X_new = pd.read_csv(uploaded_file)

        # Align columns: add missing ones as zero
        rf = pipeline.named_steps['rf']
        expected = list(rf.feature_names_in_)
        for c in expected:
            if c not in X_new.columns:
                X_new[c] = 0
        X_new = X_new[expected]

        # Score
        probs = pipeline.predict_proba(X_new)[:, 1]
        preds = (probs >= threshold).astype(int)

        # Prepare results
        result = X_new.copy()
        result['will_respond'] = preds
        result['response_proba'] = probs

        # Show and allow download
        st.dataframe(result.head(20))
        st.markdown(f"**Threshold:** {threshold:.3f}")
        st.download_button(
            "Download results as CSV",
            data=result.to_csv(index=False),
            file_name="scored_customers.csv",
            mime="text/csv"
        )
