"""Flask web application serving the trained Intel scene classifier."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import yaml
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


PARAMS = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", f"artifacts/model/{PARAMS['model_name']}.keras"))
CLASS_NAMES = list(PARAMS["class_names"])
IMAGE_SIZE = tuple(int(value) for value in PARAMS["image_size"])

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
model = load_model(MODEL_PATH)

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Intel Scene Image Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1f2937; }
    h1 { margin-bottom: 4px; }
    .muted { color: #6b7280; }
    form, .result { border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin-top: 20px; }
    button { background: #2563eb; color: #fff; border: 0; border-radius: 6px; padding: 8px 16px; cursor: pointer; }
    img { max-width: 100%; max-height: 300px; border-radius: 6px; }
    .bar { background: #dbeafe; height: 18px; border-radius: 4px; }
    .bar span { display: block; background: #2563eb; height: 100%; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    td { padding: 4px 8px; }
    .error { color: #b91c1c; }
  </style>
</head>
<body>
  <h1>Intel Scene Image Classifier</h1>
  <p class="muted">MobileNetV2 transfer learning model trained with a DVC pipeline. Classes: {{ classes|join(", ") }}.</p>
  <form method="post" enctype="multipart/form-data">
    <label for="file">Upload an image:</label>
    <input id="file" type="file" name="file" accept="image/*" required>
    <button type="submit">Classify</button>
  </form>
  {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
  {% if result %}
  <div class="result">
    <img src="data:image/png;base64,{{ preview }}" alt="Uploaded image">
    <h2>Prediction: {{ result.label }} ({{ "%.1f"|format(result.confidence * 100) }}%)</h2>
    <table>
      {% for name, prob in result.probabilities.items() %}
      <tr><td>{{ name }}</td><td style="width:60%"><div class="bar"><span style="width: {{ prob * 100 }}%"></span></div></td><td>{{ "%.1f"|format(prob * 100) }}%</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
</body>
</html>"""


def classify(image: Image.Image) -> dict[str, object]:
    """Run the model on a PIL image and return the label with class probabilities."""

    batch = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)[None] / 255.0
    probabilities = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))
    return {
        "label": CLASS_NAMES[best],
        "confidence": float(probabilities[best]),
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)},
    }


def _read_upload() -> Image.Image:
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ValueError("No file uploaded. Send an image in the 'file' form field.")
    try:
        image = Image.open(upload.stream)
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH.name)


@app.post("/predict")
def predict_api():
    try:
        return jsonify(classify(_read_upload()))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, classes=CLASS_NAMES)
    try:
        image = _read_upload()
    except ValueError as exc:
        return render_template_string(PAGE, classes=CLASS_NAMES, error=str(exc)), 400
    preview = io.BytesIO()
    image.convert("RGB").save(preview, format="PNG")
    return render_template_string(
        PAGE,
        classes=CLASS_NAMES,
        result=classify(image),
        preview=base64.b64encode(preview.getvalue()).decode("ascii"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
