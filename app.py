import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chest X-Ray AI System",
    page_icon="🫁",
    layout="wide"
)

# ---------------- UI THEME ----------------
st.markdown("""
<style>

.main {
    background-color: #0b1220;
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

[data-testid="stSidebar"] {
    background-color: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
MODEL_URL = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    model_file = BytesIO(response.content)
    return tf.keras.models.load_model(model_file)

model = load_model()

# ---------------- LABELS ----------------
CLASS_NAMES = ["Normal", "Pneumonia"]

# ---------------- PREPROCESS ----------------
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img)

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- SIDEBAR ----------------
st.sidebar.title("🫁 AI X-Ray System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🤖 AI Chat", "📊 Model Info"]
)

st.sidebar.write("---")
st.sidebar.caption("⚠️ Educational system only — not medical diagnosis")

# ---------------- MODEL INFO ----------------
if page == "📊 Model Info":
    st.title("📊 Model Information")
    st.write("Deep Learning CNN model loaded from Hugging Face.")
    st.code(MODEL_URL)

# ---------------- CHATBOT (LEVEL 4 FEATURE) ----------------
elif page == "🤖 AI Chat":
    st.title("🤖 AI Medical Assistant (Simulation)")

    st.write("Ask questions about X-ray results:")

    user_input = st.text_input("Enter question")

    if user_input:
        st.info("This is a simulated medical explanation layer.")

        st.write("💬 Response:")
        st.write("""
        Chest X-rays are interpreted based on lung opacity, structure, and abnormalities.

        If pneumonia is detected, it usually indicates:
        - Infection in lungs
        - Fluid or inflammation
        - Reduced air transparency
        """)

# ---------------- MAIN DASHBOARD ----------------
else:

    st.title("🫁 Chest X-Ray AI Diagnostic System")
    st.caption("Advanced AI-powered medical imaging assistant")

    col1, col2 = st.columns([1, 1])

    # ---------------- UPLOAD ----------------
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    # ---------------- ANALYSIS ----------------
    with col2:
        st.subheader("🧠 Analysis Panel")

        if uploaded_file:
            st.success("Image ready for analysis")
        else:
            st.warning("Upload image first")

    # ---------------- PREDICTION ----------------
    if uploaded_file:

        if st.button("🔍 Run AI Analysis", use_container_width=True):

            with st.spinner("Running deep learning model..."):

                processed = preprocess(image)
                prediction = model.predict(processed)

                predicted_class = np.argmax(prediction)
                confidence = float(np.max(prediction) * 100)
                label = CLASS_NAMES[predicted_class]

            st.markdown("---")

            # ---------------- RESULT ----------------
            st.subheader("🧾 Diagnosis Result")

            if label == "Normal":
                st.success(f"Result: {label}")
            else:
                st.error(f"Result: {label}")

            st.metric("Confidence Score", f"{confidence:.2f}%")

            # ---------------- CONFIDENCE TABLE (LEVEL 4 UPGRADE) ----------------
            st.subheader("📊 Confidence Analysis")

            df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": prediction[0]
            })

            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, prediction[0])
            ax.set_ylim([0, 1])
            st.pyplot(fig)

            # ---------------- AI EXPLANATION ----------------
            st.subheader("🤖 AI Explanation Engine")

            if label == "Normal":
                st.info("""
                The lung image appears normal with no strong abnormal opacity patterns detected.
                """)
            else:
                st.warning("""
                The model detected patterns consistent with pneumonia-like infection.

                Possible indicators:
                - Lung opacity increase
                - Inflammation patterns
                - Reduced air transparency
                """)

            # ---------------- RECOMMENDATIONS ----------------
            st.subheader("🧾 Clinical Recommendations")

            if label == "Normal":
                st.success("""
                - No abnormal findings detected
                - Maintain healthy lifestyle
                - Consult doctor if symptoms exist
                """)
            else:
                st.error("""
                - Seek medical consultation
                - Further imaging may be required
                - Monitor symptoms (fever, breathing difficulty)
                """)

            # ---------------- GRAD-CAM PLACEHOLDER (LEVEL 4 CORE FEATURE) ----------------
            st.subheader("🔥 Explainable AI (Grad-CAM)")

            st.info("""
            Next upgrade: Heatmap visualization showing which lung regions influenced the prediction.
            (Model interpretability layer)
            """)

            # ---------------- DOWNLOAD REPORT ----------------
            st.subheader("📄 Generate Report")

            report_text = f"""
Chest X-Ray AI Report

Diagnosis: {label}
Confidence: {confidence:.2f}%

Disclaimer: Educational tool only, not medical diagnosis.
"""

            st.download_button(
                label="Download Report",
                data=report_text,
                file_name="xray_report.txt"
            )
