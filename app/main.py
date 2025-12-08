"""
from fastapi import FastAPI, HTTPException
import onnxruntime as rt
import numpy as np
import json
import os

app = FastAPI()

# ---- Load ONNX model ----
sess = rt.InferenceSession("/home/ammk/snake_safety/artifacts/model_fp32.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAFETY_DIR = os.path.join(BASE_DIR, "safety_cards")

# ---- Prediction endpoint ----
@app.post("/predict/")
async def predict(data: list):
    input_data = np.array(data, dtype=np.float32)
    result = sess.run([output_name], {input_name: input_data})
    return {"prediction": result[0].tolist()}

# ---- Root endpoint ----
@app.get("/")
def read_root():
    file_path = os.path.join(SAFETY_DIR, "en.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Default safety card not found")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["note"] = "All bite events: seek medical care immediately."

    return data
"""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
#app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def dashboard(request: Request):
    # You can fetch model info, uncertainty, region data, etc. here.
    data = {
        "model_confidence": 0.64,
        "species_name": "Naja kaouthia",
        "region_safe": False,
        "fallback_message": "Confidence low — Treat as venomous.",
        "first_aid": [
            "Keep the patient calm and still.",
            "Immobilize the affected limb.",
            "Do not cut or suck the wound.",
            "Seek medical care immediately."
        ]
    }
    return templates.TemplateResponse("dashboard.html", {"request": request, "data": data})
