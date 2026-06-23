import streamlit as st
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chest X-Ray AI System",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🫁 Chest X-Ray AI Diagnostic System")
st.write("Upload a chest X-ray image to analyze using AI (Phase 1 UI only).")

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About This App")

st.sidebar.write("""
This application is an AI-powered system for analyzing chest X-ray images.

⚠️ Disclaimer:
This tool is for educational purposes only.
It is NOT a medical diagnosis system.
Always consult a medical professional.
""")

st.sidebar.divider()

st.sidebar.subheader("How to Use")
st.sidebar.write("""
1. Upload a chest X-ray image  
2. View image preview  
3. AI prediction will appear in next phase  
""")

st.sidebar.divider()

st.sidebar.subheader("Model Info")
st.sidebar.write("Hugging Face Deep Learning Model (to be connected in Phase 2)")

# ---------------- UPLOAD SECTION ----------------
st.header("📤 Upload X-Ray Image")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

# ---------------- IMAGE PREVIEW ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_container_width=True)

# ---------------- PLACEHOLDER SECTIONS ----------------
st.header("🧠 Prediction Result")
st.info("Prediction will appear here after connecting AI model (Phase 2)")

st.header("🤖 AI Explanation")
st.warning("AI explanation chatbot will be added in Phase 3")

st.header("🧾 Recommendations")
st.success("Patient guidance and recommendations will appear here in Phase 3")
