import streamlit as st
import pandas as pd
import joblib
import json


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Lead Scoring",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Lead Scoring System")
st.write("Enter the lead details below to predict the lead score.")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

try:
    model = joblib.load("lead_scoring_model.pkl")

    with open("model_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)

    st.success("✅ Lead scoring model loaded successfully!")

except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# GET MODEL INFORMATION
# ---------------------------------------------------------

feature_columns = model_info["feature_columns"]
numeric_features = model_info["numeric_features"]
categorical_features = model_info["categorical_features"]
category_options = model_info["category_options"]


# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------

st.header("📋 Lead Information")

user_input = {}


with st.form("lead_form"):

    st.subheader("Basic Lead Information")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_columns):

        # -------------------------
        # NUMERIC FEATURES
        # -------------------------

        if feature in numeric_features:

            if index % 2 == 0:
                with col1:
                    user_input[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        step=1.0
                    )
            else:
                with col2:
                    user_input[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=0.0,
                        step=1.0
                    )

        # -------------------------
        # CATEGORICAL FEATURES
        # -------------------------

        else:

            options = category_options.get(feature, [])

            if not options:
                options = ["Unknown"]

            if index % 2 == 0:
                with col1:
                    user_input[feature] = st.selectbox(
                        feature,
                        options
                    )
            else:
                with col2:
                    user_input[feature] = st.selectbox(
                        feature,
                        options
                    )

    st.divider()

    submitted = st.form_submit_button(
        "🎯 Score Lead",
        use_container_width=True
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if submitted:

    try:

        # Create dataframe
        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        st.subheader("📊 Lead Details")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # Prediction
        st.write("Model classes:", model.classes_)

        prediction = model.predict(input_df)[0]

        # Prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            max_probability = max(probabilities)

            score = round(max_probability * 100, 2)

        else:

            score = None


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.header("🎯 Lead Score")

        if score is not None:

            st.metric(
                label="AI Lead Score",
                value=f"{score}%"
            )

        st.write("### Prediction")

        st.success(
            f"Predicted Result: **{prediction}**"
        )


        # -------------------------------------------------
        # LEAD QUALITY
        # -------------------------------------------------

        if score is not None:

            if score >= 80:

                st.success(
                    "🔥 HIGH-QUALITY LEAD"
                )

                st.write(
                    "This lead has a high predicted probability. "
                    "It should receive priority follow-up."
                )

            elif score >= 50:

                st.warning(
                    "🟡 MEDIUM-QUALITY LEAD"
                )

                st.write(
                    "This lead has moderate potential. "
                    "Consider additional engagement."
                )

            else:

                st.error(
                    "🔴 LOW-QUALITY LEAD"
                )

                st.write(
                    "This lead has relatively low predicted potential."
                )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.exception(e)