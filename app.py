import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Titre de l'application
st.title("🧬 Détection de catégories de sperme")
st.markdown("Cette application permet de classifier une image de sperme comme **normal** , **anormal** ou non sperme à l'aide d'un modèle de deep learning.")

# Chargement du modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sperm_classifier.h5")
    return model

model = load_model()

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléversez une image microscopique au format BMP", type=["bmp"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_column_width=True)

    # Prétraitement de l'image
    size = (224, 224)  # Adapter à la taille d'entrée du modèle
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajout de la dimension batch

    # Prédiction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    label = "Normal" if confidence < 0.5 else "Anormal"
    score = 1 - confidence if label == "Normal" else confidence

    # Affichage du résultat
    st.markdown(f"### 🧾 Résultat de la classification : **{label}**")
    st.markdown(f"**Score de confiance :** {score:.2%}")
