import streamlit as st
from PIL import Image
import numpy as np
import requests
import base64
from openai import OpenAI
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Chest X-Ray AI", layout="wide")

# ---------------- KEYS ----------------
HF_API_KEY = st.secrets["HF_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

CLASSES = ['covid', 'normal', 'pneumonia']

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/datasets/yamram/xray-model/resolve/main/best_model_final_fixed%20(1).keras"
    response = requests.get(url)
    with open("model.keras", "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model("model.keras", compile=False)

model = load_model()

# ---------------- UI ----------------
st.title("🫁 Chest X-Ray AI System")
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])

# ---------------- HF PREDICTION ----------------
def query_hf(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image).astype(np.float32)
    img_array = resnet_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    confidence = f"{preds[0][idx]*100:.2f}%"
    return label, confidence

# ---------------- OPENAI EXPLANATION ----------------
def explain_result(label, confidence):
    prompt = f"""
A chest X-ray AI model predicted:
- Result: {label}
- Confidence: {confidence}
Explain in simple medical terms:
1. Meaning of result
2. Possible interpretation
3. What patient should do next
4. No prescriptions
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ---------------- MAIN ----------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray")

    if st.button("Analyze"):
        with st.spinner("Analyzing with AI..."):
            try:
                label, confidence = query_hf(image)
                st.success(f"Prediction: **{label.upper()}** — Confidence: **{confidence}**")
            except Exception as e:
                st.error(f"Model error: {e}")
                label = "Unknown"
                confidence = "N/A"

        if label != "Unknown":
            with st.spinner("Generating medical explanation..."):
                try:
                    explanation = explain_result(label, confidence)
                    st.subheader("🤖 AI Explanation")
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Explanation error: {e}")
