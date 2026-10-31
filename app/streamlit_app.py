import json
import numpy as np
import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm

CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
MODEL_PATH = "models/best_model_ft_v2.pt"
KB_PATH = "knowledge_base/diseases.json"
THRESHOLD = 0.70

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 4)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return m.eval()

@st.cache_data
def load_kb():
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)

def gradcam(model, x, idx):
    store = {}
    layer = model.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: store.update(a=o))
    out = model(x)
    store["a"].retain_grad()
    out[0, idx].backward()
    h1.remove()
    a, g = store["a"], store["a"].grad
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True)..., :3]
    st.image(overlay, caption="Grad-CAM: model ne kahan dekha", width=300)

    st.info(kb["disclaimer_hi"]))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()

st.set_page_config(page_title="मक्का रोग पहचान", page_icon="🌽")
st.title("🌽 मक्का रोग पहचान / Maize Disease Detector")
st.write("पत्ती की साफ़ फोटो अपलोड करें / Upload a clear leaf photo")

file = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
lang = st.radio("Language / भाषा", ["हिंदी", "English"], horizontal=True)

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded", width=300)

    model, kb = load_model(), load_kb()
    x = tf(img).unsqueeze(0)
    x.requires_grad_(True)

    with torch.enable_grad():
        out = model(x)
        probs = out.softmax(1)[0].detach().numpy()
        idx = int(probs.argmax())
        conf = float(probs[idx])
        cam = gradcam(model, tf(img).unsqueeze(0), idx)

    entropy = float(-(probs * np.log(probs + 1e-9)).sum())
    not_maize = entropy > 0.8
    if not_maize:
        st.error("यह मक्का की पत्ती नहीं लग रही। कृपया मक्का की पत्ती की साफ़ फोटो डालें। / This does not look like a maize leaf.")
    elif conf < THRESHOLD:
        st.warning("पक्का नहीं कह सकते। कृपया पत्ती की ज़्यादा साफ़ फोटो डालें। / Not sure, please upload a clearer photo.")
    else:
        info = kb[CLASSES[idx]]
        st.subheader(f"{info['name_hi']}  ({conf:.0%})")
        st.caption(info["name_en"])
        st.markdown(f"**लक्षण:** {info['symptoms_hi']}")
        st.markdown(f"**कारण:** {info['cause_hi']}")
        st.markdown(f"**उपचार:** {info['remedy_hi']}")
        st.markdown(f"**रोकथाम:** {info['prevention_hi']}")

    base = np.array(img.resize((224, 224))) / 255.0
    overlay = 0.5 * base + 0.5 * cm.jet(cam)