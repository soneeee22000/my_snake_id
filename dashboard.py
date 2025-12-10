import os
import streamlit as st
import onnxruntime as ort
import json
import numpy as np
from PIL import Image

# Load ONNX model
sess = ort.InferenceSession("artifacts/model_fp32.onnx", providers=["CPUExecutionProvider"])

# Load class map
with open("artifacts/class_map.json", "r") as f:
    cm = json.load(f)
idx_to_class = {int(k): v for k, v in cm["idx_to_class"].items()}

# Load safety guidelines
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAFETY_PATH = os.path.join(BASE_DIR, "safety_cards", "en.json")

with open(SAFETY_PATH, "r") as f:
    safety_data = json.load(f)


def preprocess(img, size=320):
    # Ensure RGB
    img = img.convert("RGB")

    # Resize
    img = img.resize((size, size))

    # Convert to numpy (H, W, C)
    arr = np.array(img).astype("float32") / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    arr = (arr - mean) / std

    # HWC -> CHW
    arr = np.transpose(arr, (2,0,1))

    # Add batch dimension -> (1,3,H,W)
    arr = arr[np.newaxis, :, :, :]

    return arr.astype(np.float32)


def predict(img):
    x = preprocess(img)

    # ONNX inference
    logits = sess.run(["logits"], {"input": x})[0]

    # Softmax
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # Highest probability
    idx = int(np.argmax(probs))
    return idx, float(probs[0, idx])


# Streamlit UI
st.title("🐍 Snake Safety Advisor")
st.write("Upload an image of a snake to get identification and Burmese safety tips.")

img_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    idx, prob = predict(img)
    species = idx_to_class[idx]

    st.subheader(f"Prediction: **{species}**")
    st.write(f"Confidence: **{prob:.2f}**")

    # safety lookup
    # Normalize species output
    species_normalized = species.replace("_"," ").replace("-"," ").title()

    # Normalize JSON keys as well
    safety_data_norm = {
        k.replace("_"," ").replace("-"," ").title(): v
        for k, v in safety_data.items()
    }

    # Lookup
    tips = safety_data_norm.get(species_normalized)


    if tips:
        st.markdown("### 🛡 Burmese Safety Guidance")
        st.write(tips["description"])
        st.write("#### ✔ First aid / What to do")
        for step in tips["actions"]:
            st.write(f"- {step}")

        st.write("#### ❌ What NOT to do")
        for step in tips["donts"]:
            st.write(f"- {step}")

        st.write("#### 🚑 Emergency Steps")
        st.write(tips["emergency"])

    else:
        st.warning("No safety information available for this species.")
