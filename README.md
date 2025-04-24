# 🧪 Détection de Sperme via Deep Learning – Projet SMIDS

## 📌 Introduction

Ce projet vise à automatiser la classification d’images microscopiques dans le cadre de l’analyse de la fertilité masculine. En combinant l’analyse d’image, le Deep Learning et une interface utilisateur simple, notre objectif est de détecter avec précision le type de cellule présent sur une image.

---

## 🗂️ Dataset – SMIDS

Le dataset **SMIDS (Sperm Morphology Image Dataset for Segmentation)** est composé d’images biomédicales classées en 3 catégories :

- 🟢 **Normal_Sperm**  
- 🔵 **Abnormal_Sperm**  
- 🟠 **Non-Sperm**

Les images sont en niveaux de gris ou en couleur, avec des résolutions variables. Un important travail de **prétraitement** a été mené (redimensionnement, histogrammes, PCA) pour uniformiser les données et améliorer la performance du modèle.

---

## 🧠 Modèle de Deep Learning

Nous utilisons **MobileNetV2**, un modèle léger et puissant, pré-entraîné sur ImageNet. Ce modèle est intégré à une architecture personnalisée :

- 🔒 Mode *feature extractor* (couches gelées)
- 🧠 Ajout d’un classificateur dense (GlobalAveragePooling + Dense)
- 🧪 Optimisation avec l’algorithme **Adam** et la fonction **categorical_crossentropy**
- ✅ Précision atteinte : **~86%**

Des techniques de régularisation (Dropout, EarlyStopping) ont été mises en place pour éviter le surapprentissage. Un fine-tuning est envisageable pour affiner davantage les performances.

---

## 📊 Analyse exploratoire

Avant la modélisation, nous avons exploré les données à travers :

- 📉 Distribution des classes
- 📏 Étude des dimensions (largeur/hauteur)
- 🌈 Histogrammes de niveaux de gris
- 📐 Réduction de dimension avec **PCA**
- 🔍 Clustering et visualisation des patterns par classe

Ces analyses ont permis de mieux comprendre les caractéristiques discriminantes des images.

---

## 🌐 Application Web

Une **application web** Streamlit a été développée pour permettre :

- 🖼️ L’upload d’une image microscopique
- 🔍 La prédiction de la classe (normal, anormal, non-sperme)
- 📋 Un affichage clair et ergonomique des résultats

Cette application est pensée comme un **outil d’aide au diagnostic médical**, simple à utiliser dans un contexte réel de laboratoire.




