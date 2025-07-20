import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# 🌊 Custom background and styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #f0f8ff;
        }
        .block-container {
            background-color: rgba(0, 51, 102, 0.75);
            padding: 2rem;
            border-radius: 12px;
        }
        .centered-title h1 {
            text-align: center;
            font-size: 3em;
            color: #ffc0cb;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# 🎯 Title
st.markdown('<div class="centered-title"><h1>🐠 Multiclass Fish Classifier</h1></div>', unsafe_allow_html=True)

# 🧠 Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\HP\Downloads\mobilenet_model.h5")

# 🏷️ Load class labels
with open(r"C:\Users\HP\Downloads\class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

model = load_model()
img_size = (224, 224)

# Upload instruction
st.markdown("📥 Upload a fish image to classify its species:")

# Custom label
st.markdown(
    "<p style='color: #aee4ff; font-weight: bold; font-size: 18px;'>📸 Upload image (JPG, JPEG, PNG):</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)



    # 🔁 Preprocess image
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # 🔍 Predict
    preds = model.predict(image_array)
    predicted_index = np.argmax(preds)
    predicted_class = idx_to_class[predicted_index]
    confidence = float(np.max(preds)) * 100

    # ✅ Display result
    st.markdown(
        f"""
        <div style='
            background-color: #e8f5e9;
            padding: 15px;
            border-left: 5px solid #00695c;
            border-radius: 8px;
            color: #004d40;
            font-size: 18px;
            font-weight: bold;
        '>
        🎯 Predicted Class: {predicted_class} ({confidence:.2f}% confidence)
        </div>
        """,
        unsafe_allow_html=True
    )

    # 📊 Top-3 predictions
    st.subheader("🔍 Top-3 Predictions:")
    top_3 = preds[0].argsort()[-3:][::-1]
    for i in top_3:
        st.write(f"- {idx_to_class[i]}: {preds[0][i] * 100:.2f}%")

    st.markdown("---")
    st.markdown("**📌 Model: MobileNet | Trained on Multiclass Fish Dataset | Built with 💙 using Streamlit**")
