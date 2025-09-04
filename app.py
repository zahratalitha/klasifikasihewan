import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Prediksi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Kasifikasi Gambar: Anjing vs Kucing")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/klasifikasihewan",   # ganti dengan repo HuggingFace kamu
        filename="kucinganjing.h5"         # nama file model
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()
uploaded_file = st.file_uploader("Upload gambar anjing atau kucing", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    img = img.resize((180, 180))   
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    label = "🐶 Anjing" if prob > 0.5 else "🐱 Kucing"

    st.success(f"Hasil Prediksi: {label} (Probabilitas: {prob:.2f})")
