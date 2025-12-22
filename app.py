import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import tempfile

# ==========================================
# 1. DÉFINITION DE L'ARCHITECTURE (Nécesaire pour charger le .pth)
# ==========================================

# Configuration (Doit matcher celle de l'entraînement)
CONFIG = {
    'img_size': 64,
    'embed_dim': 128,
    'num_classes': 3,
    'recursion_depth': 4,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- REMPLACEMENT DES CLASSES ---

class SpatialVisualEmbedding(nn.Module):
    """
    Encodeur qui conserve la dimension spatiale (4x4 patches) 
    au lieu de tout aplatir. Permet l'attention.
    (Code provenant du notebook)
    """
    def __init__(self, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            # Étape 1 : 64x64 -> 32x32 (Sortie 32 canaux)
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            
            # Étape 2 : 32x32 -> 16x16 (Sortie 64 canaux)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(),
            
            # Étape 3 : 16x16 -> 8x8 (Sortie output_dim canaux)
            nn.Conv2d(64, output_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU6(),
            
            # Force une grille de 4x4 (16 tokens) peu importe la taille d'entrée
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
    
    def forward(self, x):
        # Sortie: [Batch, Dim, 4, 4] -> Flatten -> [Batch, Dim, 16] -> Permute -> [Batch, 16, Dim]
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) 
        return x

class AttentionTRMBlock(nn.Module):
    """
    Bloc récursif utilisant le Self-Attention (Transformer).
    (Code provenant du notebook)
    """
    def __init__(self, embed_dim, num_classes, num_heads=4):
        super().__init__()
        # Couche d'attention
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        
        # Réseau Feed-Forward
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Tête de classification (Moyenne des patches)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_feat, z_prev):
        # 1. Fusion de l'Input visuel et du Raisonnement précédent
        z_curr = z_prev + x_feat
        
        # 2. Self-Attention : Le modèle compare les patches entre eux
        z_norm = self.attn_norm(z_curr)
        attn_out, _ = self.attn(z_norm, z_norm, z_norm)
        z_curr = z_curr + attn_out # Connexion résiduelle
        
        # 3. Feed-Forward
        z_curr = z_curr + self.ffn(self.ffn_norm(z_curr))
        
        # 4. Prédiction : On fait la moyenne des 16 patches pour décider
        z_pooled = z_curr.mean(dim=1) 
        y_logits = self.head(z_pooled)
        
        return z_curr, y_logits

class DriverTRM(nn.Module):
    """
    Classe principale mise à jour pour utiliser l'Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Utilisation des nouveaux blocs définis ci-dessus
        self.embedding = SpatialVisualEmbedding(config['embed_dim'])
        self.trm_block = AttentionTRMBlock(config['embed_dim'], config['num_classes'])
        
    def forward(self, img):
        # x_feat est maintenant une séquence : [Batch, 16, Dim]
        x_feat = self.embedding(img)
        
        # z_curr est aussi une séquence (mémoire spatiale)
        z_curr = torch.zeros_like(x_feat).to(img.device)
        
        outputs_list = []
        
        # Boucle de Raisonnement Récursif
        for _ in range(self.config['recursion_depth']):
            z_curr, y_logits = self.trm_block(x_feat, z_curr)
            outputs_list.append(y_logits)
            
        return outputs_list

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================

@st.cache_resource
def load_model(model_path):
    try:
        model = DriverTRM(CONFIG)
        # map_location permet de charger sur CPU si pas de CUDA disponible
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        return model
    except FileNotFoundError:
        return None

def get_preprocessing():
    # Mêmes transformations que pour la validation
    return T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_frame(model, frame_rgb, transform):
    # Conversion numpy -> PIL -> Tensor
    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        outputs = model(img_tensor)
        final_output = outputs[-1] # Dernière étape du TRM
        probs = F.softmax(final_output, dim=1)
        score, pred = torch.max(probs, 1)
        
    return pred.item(), score.item(), probs[0].cpu().numpy()

# Classes
CLASSES = {0: "ALERT (Normal)", 1: "DROWSY (Somnolent)", 2: "DISTRACTED (Distrait)"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)} # RGB pour affichage

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(page_title="Driver Monitor AI", layout="wide")

st.title("🚗 Driver Drowsiness & Distraction Detection (TRM)")
st.sidebar.header("Configuration")

# Chargement du modèle
model_path = "best_trm_model.pth" # Assurez-vous que le fichier est à côté de app.py
model = load_model(model_path)

if model is None:
    st.error(f"Modèle introuvable : {model_path}. Veuillez placer le fichier .pth dans le dossier.")
    st.stop()
else:
    st.sidebar.success("Modèle TRM chargé avec succès !")

# Choix du mode
mode = st.sidebar.radio("Choisissez une source :", ("🖼️ Image", "🎥 Vidéo", "📷 Webcam (Live)"))

transform = get_preprocessing()

# --- MODE IMAGE ---
if mode == "🖼️ Image":
    uploaded_file = st.file_uploader("Uploader une image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption="Image originale", use_container_width=True)
            
        # Inférence
        img_tensor = transform(image).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs[-1], dim=1)
            score, pred = torch.max(probs, 1)
            
        class_idx = pred.item()
        
        with col2:
            st.subheader("Résultat")
            if class_idx == 0:
                st.success(f"**{CLASSES[0]}**")
            elif class_idx == 1:
                st.error(f"**{CLASSES[1]}**")
            else:
                st.warning(f"**{CLASSES[2]}**")
                
            st.metric("Confiance", f"{score.item():.2%}")
            
            st.write("Détails des probabilités :")
            st.bar_chart({k: v for k, v in zip(CLASSES.values(), probs[0].cpu().numpy())})

# --- MODE VIDÉO ---
elif mode == "🎥 Vidéo":
    uploaded_video = st.file_uploader("Uploader une vidéo", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.image([])
        
        stop_button = st.button("Arrêter la vidéo")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV est BGR, on convertit en RGB pour le modèle et l'affichage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pred_idx, conf, _ = predict_frame(model, frame_rgb, transform)
            
            # Dessin sur l'image
            label_text = f"{CLASSES[pred_idx]} ({conf:.0%})"
            color = COLORS[pred_idx]
            
            cv2.putText(frame_rgb, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2)
            
            # Affichage dans Streamlit
            st_frame.image(frame_rgb, caption="Analyse vidéo TRM", use_container_width=True)
            
        cap.release()

# --- MODE WEBCAM ---
elif mode == "📷 Webcam (Live)":
    st.write("L'analyse se fera en temps réel sur le flux de votre webcam.")
    run = st.checkbox('Activer la Webcam')
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0) # 0 est l'index par défaut de la webcam
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur d'accès à la webcam")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inférence TRM
            pred_idx, conf, probs_array = predict_frame(model, frame, transform)
            
            # Affichage Overlay
            label = CLASSES[pred_idx]
            color = COLORS[pred_idx]
            
            cv2.putText(frame, f"{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Barre latérale dynamique
            with st.sidebar:
                st.markdown(f"### État actuel : **{label}**")
                st.progress(int(conf * 100))
            
            FRAME_WINDOW.image(frame)
        
        cap.release()
    else:
        st.write("Cochez la case pour démarrer.")