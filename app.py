import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (160, 160)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("furniture_cnn.keras", compile=False)
    names = json.load(open("class_names.json"))
    return model, names

model, class_names = load_model()

st.title("Klasifikasi Furniture (Custom CNN)")
st.write("Upload foto furniture (Seperti Kursi, Meja, Laptop dan Meja), model akan memprediksi kelasnya.")

file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Gambar diupload", use_container_width=True)

    x = np.array(img.resize(IMG_SIZE), dtype="float32")   # tanpa /255, sudah ada Rescaling di model
    probs = model.predict(x[None, ...], verbose=0)[0]

    pred = int(np.argmax(probs))
    st.success(f"Prediksi: **{class_names[pred]}** ({probs[pred]*100:.1f}%)")

    st.bar_chart({c: float(p) for c, p in zip(class_names, probs)})
