# 🚗 Driver Drowsiness & Distraction Detection (Edge AI via TRM)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

Un système de surveillance conducteur (DMS) ultra-léger conçu pour tourner sur des architectures Edge (Raspberry Pi, Jetson Nano). Ce projet remplace les lourds réseaux de neurones classiques (VGG, ResNet + LSTM) par une approche innovante : le **Tiny Recursive Model (TRM)**.

## 🧠 Pourquoi le TRM ? (Concept Scientifique)

Ce projet est une implémentation appliquée du papier de recherche *"Less is More: Recursive Reasoning with Tiny Networks"*.

Au lieu d'empiler des centaines de couches, nous utilisons un **réseau minuscule (Tiny Network) de seulement 2 couches** qui "réfléchit" de manière récursive sur l'image.

* **Raisonnement Récursif :** Le modèle met à jour un état latent $z$ (le raisonnement) et sa prédiction $y$ (la réponse) sur plusieurs itérations pour une même image.
* **Deep Supervision :** L'entraînement calcule la perte à chaque étape de récursion, forçant le modèle à converger plus vite et à être plus robuste.
* **Avantage Edge :** Moins de paramètres = Inférence plus rapide et consommation mémoire réduite, idéal pour l'embarqué.

## 🎯 Fonctionnalités

* **Détection Multi-Classes :**
    * ✅ **Alert :** Conduite normale.
    * 😴 **Drowsy :** Signes de fatigue (yeux fermés).
    * 📱 **Distracted :** Usage du téléphone, radio, ou regard détourné.
* **Interface Web (Streamlit) :** Dashboard interactif pour tester via Image, Vidéo ou Webcam en temps réel.
* **Pipeline Automatisé :** Téléchargement automatique des datasets Kaggle (State Farm & Eye Dataset).
* **Robustesse :** Augmentation de données simulant des conditions de nuit ou de tunnel.

## 📂 Structure du Projet

```bash
Driver-Drowsiness-TRM/
├── notebook.ipynb         # Pipeline complet (Data, Train, Eval)
├── app.py                 # Application Frontend (Streamlit)
├── requirements.txt       # Dépendances Python
├── README.md              # Documentation
└── best_trm_model.pth     # Le meilleur modèle entraîné (Apparait après l'entraînement)