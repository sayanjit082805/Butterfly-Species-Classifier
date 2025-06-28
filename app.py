import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import random

st.set_page_config(
    page_title="Butterfly Classifier",
    page_icon=":butterfly:",
    initial_sidebar_state="expanded",
)

butterfly_samples = [
    {
        "name": "Southern Dogface",
        "image": "public/southern.jpg",
    },
    {
        "name": "Adonis",
        "image": "public/adonis.jpg",
    },
    {
        "name": "Appollo",
        "image": "public/appollo.jpg",
    },
    {
        "name": "Malachite",
        "image": "public/malachite.jpg",
    },
    {
        "name": "Poppinjay",
        "image": "public/Poppinjay.jpg",
    },
    {
        "name": "Eastern Dapple White",
        "image": "public/eastern.jpg",
    },
    {
        "name": "Red Admiral",
        "image": "public/admiral.jpg",
    },
    {
        "name": "Monarch",
        "image": "public/monarch.jpg",
    },
    {
        "name": "Question Mark",
        "image": "public/question.jpg",
    },
]

model = tf.keras.models.load_model("classifier.keras")
classes = joblib.load("classes.pkl")

st.title("Butterfly Image Classifier")

st.subheader(
    "Classify different butterfly species from images using the EfficientNet architecture.",
    divider="gray",
)

st.write(
    """
   A deep learning project to classify different butterfly species into nearly 75 distinct types. The model is trained on the Butterfly Image Classification dataset from Kaggle and uses the EfficientNet architecture for classification.     
"""
)

st.info(
    "Some classifications may be inaccurate. The model may not generalize perfectly to all images."
)

uploaded_file = st.file_uploader("Select an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Classify"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preprocessing image...")
        progress_bar.progress(25)

        status_text.text("Running model inference...")
        progress_bar.progress(50)

        predictions = model.predict(img_array)
        progress_bar.progress(75)

        status_text.text("Processing results...")
        predicted_class = np.argmax(predictions, axis=1)[0]
        progress_bar.progress(100)

        status_text.text("Complete!")

        progress_bar.empty()
        status_text.empty()

        st.success(f"Prediction: **{classes[predicted_class].title()}**")

st.divider()

if "butterfly_samples" not in st.session_state:
    st.session_state.butterfly_samples = random.choice(butterfly_samples)

img = st.session_state.butterfly_samples
st.sidebar.image(img["image"], caption=img["name"])

st.sidebar.header("Dataset Information", divider="gray")

st.sidebar.markdown(
    """ 
    The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification). It contains more than 1000 images of different butterflies.

**Statistics** :

- **Total Images**: ~1,000 images
- **Training Set**: ~800 images
- **Validation Set**: ~200 images
- **Classes**: 75 distinct species
"""
)

st.sidebar.header("Model Metrics", divider="gray")

st.sidebar.markdown(
    """ 
The model achieves an overall accuracy of ~99% on the training data and ~91% on the validation data.
"""
)

st.sidebar.markdown(
    """> The model is not intended for real-world applications and should not be used for any commercial or operational purposes."""
)
