import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chest X-Ray AI System",
    page_icon="🫁",
    layout="wide"
)

# ---------------- CUSTOM CSS (PRO UI THEME) ----------------
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
    font-size: 16px;
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
    model = tf.keras.models.load_model(model_file)
    return model

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
st.sidebar.title("🫁 Chest X-Ray AI")

page = st.sidebar.radio("Navigation", ["🏠 Home", "ℹ️ About", "📊 Model Info"])

st.sidebar.write("---")
st.sidebar.caption("⚠️ Educational use only. Not medical diagnosis.")

# ---------------- ABOUT PAGE ----------------
if page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.write("""
    This AI system analyzes chest X-ray images using a deep learning model.

    Features:
    - Image classification
    - Confidence scoring
    - AI explanation (Level 3)
    - Future: chatbot medical assistant

    ⚠️ Not a medical tool.
    """)

# ---------------- MODEL INFO PAGE ----------------
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.write("Model loaded from Hugging Face dataset repository.")
    st.code(MODEL_URL)

# ---------------- MAIN APP ----------------
else:

    st.title("🫁 Chest X-Ray AI Diagnostic System")
    st.caption("AI-powered assistant for educational medical imaging analysis")

    col1, col2 = st.columns([1, 1])

    # ---------------- UPLOAD ----------------
    with col1:
        st.subheader("📤 Upload X-Ray")

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # ---------------- ANALYSIS PANEL ----------------
    with col2:
        st.subheader("🧠 Analysis Panel")

        if uploaded_file:
            st.info("Ready for AI analysis")
        else:
            st.warning("Please upload an image")

    # ---------------- PREDICTION ----------------
    if uploaded_file:

        if st.button("🔍 Analyze X-Ray", use_container_width=True):

            with st.spinner("Analyzing X-Ray using AI model..."):

                processed = preprocess(image)
                prediction = model.predict(processed)

                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                label = CLASS_NAMES[predicted_class]

            st.markdown("---")

            # ---------------- RESULT ----------------
            st.subheader("🧾 Prediction Result")

            if label == "Normal":
                st.success(f"Result: {label}")
            else:
                st.error(f"Result: {label}")

            st.metric("Confidence Score", f"{confidence:.2f}%")

            # ---------------- CONFIDENCE CHART ----------------
            st.subheader("📊 Confidence Breakdown")

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, prediction[0])
            ax.set_ylim([0, 1])
            st.pyplot(fig)

            # ---------------- AI EXPLANATION (LEVEL 3) ----------------
            st.subheader("🤖 AI Explanation")

            if label == "Normal":
                st.info("""
                The model predicts a normal chest X-ray.

                No major abnormal lung opacity patterns detected.
                """)
            else:
                st.warning("""
                The model detects patterns consistent with Pneumonia.

                Possible indicators:
                - Lung opacity
                - Inflammation patterns
                - Reduced air space visibility
                """)

            # ---------------- RECOMMENDATIONS ----------------
            st.subheader("🧾 Recommendations")

            if label == "Normal":
                st.success("""
                - No immediate concerns detected
                - Maintain healthy lifestyle
                - Consult doctor if symptoms exist
                """)
            else:
                st.error("""
                - Consult a healthcare professional
                - Further diagnostic tests may be required
                - Seek medical attention if symptoms worsen
                """)

            # ---------------- ADVANCED PLACEHOLDER (GRAD-CAM SLOT) ----------------
            st.subheader("🔥 Advanced Visualization (Grad-CAM)")

            st.info("Coming next upgrade: Heatmap showing affected lung regions")
