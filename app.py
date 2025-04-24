import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Titre de l'application
st.title("🧬 Détection de catégories de sperme")
st.markdown("Cette application permet de classifier une image de sperme comme **non sperme** , **normal** ou **anormal** à l'aide d'un modèle de deep learning.")
st.markdown("Il est à noter que nous traitons que les images au format **bmp** .")


# Chargement du modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sperm_classifier.h5")
    return model

model = load_model()

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléversez une image microscopique au format JPG ou PNG", type=["bmp"])

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
    print('pred',prediction)
    confidence = float(prediction[0][0])
    print('conf',confidence)

    # Affichage du résultat
    pred_index = np.argmax(prediction[0])  # Index de la classe la plus probable
    confidence = float(prediction[0][pred_index])  # Confiance associée

    # Labels des classes (adapter si nécessaire)
    labels = [ "Anormal", "Non Sperm","Normal"]
    label = labels[pred_index]

    # Affichage du résultat
    st.write(f"### 🏷️ Classe prédite : **{label}**")
    st.write(f"### 🔍 Confiance : **{confidence:.4f}**")
