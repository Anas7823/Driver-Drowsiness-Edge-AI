import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import tempfile
import torchvision.models as models

# ==========================================
# 1. DÉFINITION DE L'ARCHITECTURE (Doit correspondre au notebook)
# ==========================================

# Configuration de l'entraînement
CONFIG = {
    'img_size': 224,
    'embed_dim': 128,
    'num_classes': 3,
    'recursion_depth': 3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- Architecture du notebook (MobileNetV3 + Attention) ---

class PretrainedVisualEmbedding(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = models.mobilenet_v3_small(weights=weights)
        self.features = self.backbone.features
        
        for param in self.features.parameters():
            param.requires_grad = False # Geler les poids pour l'inférence
            
        self.adapter = nn.Sequential(
            nn.Conv2d(576, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU6()
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.features(x)
        x = self.adapter(x)
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        return x

class AttentionTRMBlock(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, attn_dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, 
                                          dropout=attn_dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.gate = nn.Parameter(torch.tensor([2.0])) 

    def forward(self, tokens, z_prev):
        g = torch.sigmoid(self.gate) 
        z_curr = z_prev + g * tokens
        z_norm = self.attn_norm(z_curr)
        attn_out, attn_weights = self.attn(z_norm, z_norm, z_norm, average_attn_weights=False)
        z_curr = z_curr + attn_out
        z_curr = z_curr + self.ffn(self.ffn_norm(z_curr))
        cls_tok = z_curr[:, 0, :] 
        y_logits = self.head(cls_tok)
        return z_curr, y_logits, attn_weights

class DriverTRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.H = self.W = 8
        self.embedding = PretrainedVisualEmbedding(self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.H*self.W, self.embed_dim))
        self.trm_block = AttentionTRMBlock(self.embed_dim, config['num_classes'])
        
    def forward(self, img):
        B = img.size(0)
        x_feat = self.embedding(img)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, x_feat], dim=1)
        tokens = tokens + self.pos_embed
        z_curr = torch.zeros_like(tokens).to(img.device)
        outputs_list = []
        last_attn = None
        for _ in range(self.config['recursion_depth']):
            z_curr, y_logits, last_attn = self.trm_block(tokens, z_curr)
            outputs_list.append(y_logits)
        return outputs_list, last_attn

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================

@st.cache_resource
def load_model(model_path):
    try:
        model = DriverTRM(CONFIG)
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.error("Assurez-vous que l'architecture dans `app.py` correspond à celle du modèle entraîné.")
        return None


def get_preprocessing():
    # Transformations pour la validation/inférence (doit correspondre au notebook)
    return T.Compose([
        T.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_frame(model, frame_rgb, transform):
    pil_img = Image.fromarray(frame_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        # Le modèle du notebook retourne (outputs_list, last_attn)
        outputs, _ = model(img_tensor)
        final_output = outputs[-1] # On prend la dernière prédiction
        probs = F.softmax(final_output, dim=1)
        score, pred = torch.max(probs, 1)
        
    return pred.item(), score.item(), probs[0].cpu().numpy()

# Classes et couleurs
CLASSES = {0: "ALERT (Normal)", 1: "DROWSY (Somnolent)", 2: "DISTRACTED (Distrait)"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)} # BGR pour OpenCV

# ==========================================
# 3. INTERFACE STREAMLIT (inchangée)
# ==========================================

st.set_page_config(page_title="Driver Monitor AI", layout="wide")

st.title("🚗 Driver Drowsiness & Distraction Detection (TRM)")
st.sidebar.header("Configuration")

# Chargement du modèle
model_path = "best_trm_model.pth"
model = load_model(model_path)

if model is None:
    st.error(f"Modèle introuvable : {model_path}. Veuillez placer le fichier .pth dans le dossier.")
    st.stop()
else:
    st.sidebar.success("Modèle TRM chargé avec succès !")

mode = st.sidebar.radio("Choisissez une source :", ("🖼️ Image", "🎥 Vidéo", "📷 Webcam (Live)"))

transform = get_preprocessing()

if mode == "🖼️ Image":
    uploaded_file = st.file_uploader("Uploader une image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption="Image originale", use_container_width=True)
            
        pred_idx, score, probs = predict_frame(model, np.array(image), transform)
        
        with col2:
            st.subheader("Résultat")
            if pred_idx == 0:
                st.success(f"**{CLASSES[0]}**")
            elif pred_idx == 1:
                st.error(f"**{CLASSES[1]}**")
            else:
                st.warning(f"**{CLASSES[2]}**")
            st.metric("Confiance", f"{score:.2%}")
            st.bar_chart({k: v for k, v in zip(CLASSES.values(), probs)})

elif mode == "🎥 Vidéo":
    uploaded_video = st.file_uploader("Uploader une vidéo", type=['mp4', 'avi', 'mov'])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.image([])
        stop_button = st.button("Arrêter la vidéo")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_idx, conf, _ = predict_frame(model, frame_rgb, transform)
            
            # Affichage sur l'image (OpenCV utilise BGR)
            label_text = f"{CLASSES[pred_idx]} ({conf:.0%})"
            color_bgr = tuple(reversed(COLORS[pred_idx])) # Convertir RGB en BGR pour OpenCV
            cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_bgr, 3)
            
            st_frame.image(frame, channels="BGR", caption="Analyse vidéo TRM")
            
        cap.release()
        tfile.close()

elif mode == "📷 Webcam (Live)":
    st.write("L'analyse se fera en temps réel sur le flux de votre webcam.")
    run = st.checkbox('Activer la Webcam')
    
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur d'accès à la webcam")
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_idx, conf, _ = predict_frame(model, frame_rgb, transform)
            
            label = CLASSES[pred_idx]
            color_bgr = tuple(reversed(COLORS[pred_idx]))
            
            cv2.putText(frame, f"{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 3)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            
            with st.sidebar:
                st.markdown(f"### État actuel : **{label}**")
                st.progress(int(conf * 100))
            
            FRAME_WINDOW.image(frame, channels="BGR")
        
        cap.release()
    else:
        st.write("Cochez la case pour démarrer.")
