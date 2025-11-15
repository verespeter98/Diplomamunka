import os

MODEL_PATH = "forecast_model_training/models/nhits_weights_sentiment.pth"
OUTPUT_PY_FILE = "src/model_data.py"  

os.makedirs(os.path.dirname(OUTPUT_PY_FILE), exist_ok=True)

with open(MODEL_PATH, "rb") as f:
    model_bytes = f.read()


with open(OUTPUT_PY_FILE, "w") as f:
    f.write(f"model_weights = {model_bytes!r}\n")

print(f"Sikeresen átalakítva: {MODEL_PATH} -> {OUTPUT_PY_FILE}")