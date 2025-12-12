import os
import streamlit as st
import onnxruntime as ort
import json
import numpy as np
from PIL import Image

# -------------------------
# FULL SCREEN MODE
# -------------------------
st.set_page_config(layout="wide")
font = "source-sans-pro, sans-serif"

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFE0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




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
    img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    arr = arr[np.newaxis, :, :, :]
    return arr.astype(np.float32)


def predict(img):
    x = preprocess(img)
    logits = sess.run(["logits"], {"input": x})[0]

    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    idx = int(np.argmax(probs))

    return idx, float(probs[0, idx])


# ----------------------------------------------------------
#   TWO-PAGE BOOK LAYOUT
# ----------------------------------------------------------

LEFT, RIGHT = st.columns([1.1, 1.4], gap="large")

# --------------------------
# LEFT PAGE (Title + upload + image)
# --------------------------
with LEFT:

    st.title("🐍 Snake Safety Advisor")
    st.write("Upload an image of a snake to get identification and Burmese safety tips.")

    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Show placeholder before upload
    if not img_file:
        st.write("### ")
        st.write("Upload an image to display it here.")
        img = None

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)


# --------------------------
# RIGHT PAGE (Prediction + Safety Info)
# --------------------------

with RIGHT:

    if img_file:

        idx, prob = predict(img)
        species = idx_to_class[idx]

        st.subheader(f"Prediction: **{species}**")
        st.write(f"Confidence: **{prob:.2f}**")

        # Normalize species
        species_normalized = species.replace("_", " ").replace("-", " ").title()

        safety_data_norm = {
            k.replace("_", " ").replace("-", " ").title(): v
            for k, v in safety_data.items()
        }

        tips = safety_data_norm.get(species_normalized)

        if tips:
            st.subheader("Burmese Safety Guidance")

            st.info(f"**Venom Status:** {tips['venom_status']}\n\n**Description:** {tips['description']}\n\n**Danger Level:** {tips['danger_level']}")

            st.subheader("What to do")
            for step in tips["identification_tips"]:
                st.write(f"- {step}")
            for step in tips["recommended_actions"]:
                st.write(f"- {step}")

            st.subheader("What NOT to do")
            for step in tips["donts"]:
                st.write(f"- {step}")
            for step in tips["first_aid"]:
                st.write(f"- {step}")

            st.subheader("Emergency Steps")
            st.write(tips["emergency"])
            st.write(tips["fallback_protocol"])

        else:
            st.warning("No safety information available for this species.")

    else:
        st.info("Prediction and safety tips will appear here after you upload an image.")
