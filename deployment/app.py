Predictive Maintenance - Streamlit Deployment Application

Final MLOps deployment application.

Features:
    - Loads the trained Random Forest model from Hugging Face Model Hub
    - Accepts six operational sensor inputs
    - Recreates the four engineered features used during training
    - Creates the prediction DataFrame using the exact training feature order
    - Displays prediction and confidence
    - Displays engineered features and submitted inputs

Model Repository:
    hiteshsharma/predictive-maintenance-model
"""

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


# ==========================================================
# CONFIGURATION
# ==========================================================

HF_USERNAME = os.getenv(
    "HF_USERNAME",
    "hiteshsharma"
)

HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    f"{HF_USERNAME}/predictive-maintenance-model"
)

MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "best_model.pkl"
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)


# ==========================================================
# APPLICATION HEADER
# ==========================================================

st.title(
    "⚙️ Predictive Maintenance System"
)

st.markdown(
    """
    **Machine Learning Based Engine Condition Prediction**

    Enter the current engine sensor readings below to predict
    the engine condition and support proactive maintenance decisions.
    """
)


# ==========================================================
# LOAD MODEL FROM HUGGING FACE
# ==========================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = joblib.load(
        model_path
    )

    return model


try:

    model = load_model()

    st.success(
        "✅ Model loaded successfully from Hugging Face Model Hub."
    )

except Exception as error:

    st.error(
        "❌ Unable to load the trained model from Hugging Face."
    )

    st.info(
        f"Model repository: {HF_MODEL_REPO}"
    )

    st.exception(error)

    st.stop()


# ==========================================================
# SENSOR INPUTS
# ==========================================================

st.subheader(
    "1. Enter Engine Sensor Parameters"
)

col1, col2 = st.columns(2)


with col1:

    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0.0,
        value=1500.0,
        step=10.0
    )

    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure",
        min_value=0.0,
        value=6.0,
        step=0.1
    )


with col2:

    coolant_pressure = st.number_input(
        "Coolant Pressure",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

    lub_oil_temp = st.number_input(
        "Lub Oil Temperature",
        value=80.0,
        step=0.5
    )

    coolant_temp = st.number_input(
        "Coolant Temperature",
        value=85.0,
        step=0.5
    )


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

temperature_difference = (
    coolant_temp
    - lub_oil_temp
)

total_pressure = (
    fuel_pressure
    + lub_oil_pressure
    + coolant_pressure
)

pressure_ratio = (
    fuel_pressure
    / lub_oil_pressure
    if lub_oil_pressure != 0
    else 0.0
)

rpm_temperature_interaction = (
    engine_rpm
    * coolant_temp
)


# ==========================================================
# CREATE PREDICTION DATAFRAME
# ==========================================================

input_df = pd.DataFrame(
    [
        {
            "Engine_RPM": engine_rpm,

            "Lub_Oil_Pressure":
                lub_oil_pressure,

            "Fuel_Pressure":
                fuel_pressure,

            "Coolant_Pressure":
                coolant_pressure,

            "Lub_Oil_Temp":
                lub_oil_temp,

            "Coolant_Temp":
                coolant_temp,

            "Temperature_Difference":
                temperature_difference,

            "Total_Pressure":
                total_pressure,

            "Pressure_Ratio":
                pressure_ratio,

            "RPM_Temperature_Interaction":
                rpm_temperature_interaction,
        }
    ]
)


# ==========================================================
# VERIFY FEATURE ORDER
# ==========================================================

EXPECTED_FEATURES = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temp",
    "Coolant_Temp",
    "Temperature_Difference",
    "Total_Pressure",
    "Pressure_Ratio",
    "RPM_Temperature_Interaction"
]


input_df = input_df[
    EXPECTED_FEATURES
]


# ==========================================================
# ENGINEERED FEATURE DISPLAY
# ==========================================================

st.subheader(
    "2. Engineered Features"
)

with st.expander(
    "View calculated engineered features"
):

    engineered_df = input_df[
        [
            "Temperature_Difference",
            "Total_Pressure",
            "Pressure_Ratio",
            "RPM_Temperature_Interaction"
        ]
    ]

    st.dataframe(
        engineered_df,
        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

st.subheader(
    "3. Engine Condition Prediction"
)

predict_clicked = st.button(
    "🔍 Predict Engine Condition",
    type="primary",
    use_container_width=True
)


if predict_clicked:

    try:

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]

        prediction_value = int(
            prediction
        )

        # --------------------------------------------------
        # Prediction Probability
        # --------------------------------------------------

        confidence = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_df
                )[0]
            )

            confidence = (
                float(
                    max(probabilities)
                )
                * 100
            )

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        # The project source defines Engine_Condition as
        # the binary target but does not explicitly document
        # a semantic mapping of 0/1 to business labels.
        # Therefore the UI keeps the actual predicted class
        # visible and makes the business interpretation
        # configurable.

        class_0_label = os.getenv(
            "CLASS_0_LABEL",
            "Engine Condition Class 0"
        )

        class_1_label = os.getenv(
            "CLASS_1_LABEL",
            "Engine Condition Class 1"
        )

        if prediction_value == 0:

            st.success(
                f"🟢 Prediction: {class_0_label}"
            )

        else:

            st.warning(
                f"🟠 Prediction: {class_1_label}"
            )

        st.metric(
            "Predicted Class",
            str(prediction_value)
        )

        if confidence is not None:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        # --------------------------------------------------
        # Maintenance decision support
        # --------------------------------------------------

        st.markdown(
            """
            **Business Decision Support**

            The prediction can be used as an early-warning signal
            for condition-based maintenance. A positive maintenance
            indicator should trigger inspection of the relevant
            pressure, temperature and RPM readings before a potential
            equipment issue develops into unplanned downtime.
            """
        )

        # --------------------------------------------------
        # Display input DataFrame
        # --------------------------------------------------

        with st.expander(
            "View Prediction Input DataFrame"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(
            error
        )


# ==========================================================
# APPLICATION INFORMATION
# ==========================================================

st.divider()

with st.expander(
    "About this application"
):

    st.markdown(
        f"""
        **Model:** Random Forest Classifier

        **Model Repository:** `{HF_MODEL_REPO}`

        **Model Artifact:** `{MODEL_FILENAME}`

        **Input Sensors:** 6

        **Engineered Features:** 4

        **MLflow:** Experiment tracking implemented

        **Hugging Face:** Dataset and model registration implemented

        **Deployment Framework:** Streamlit
        """
    )

st.caption(
    "Predictive Maintenance | ML + MLOps | "
    "Hugging Face Model Hub | Streamlit"
)
'''

app_path = deployment_dir / "app.py"

app_path.write_text(
    app_code,
    encoding="utf-8"
)

print("✅ app.py created successfully")
print(f"Location: {app_path}")
