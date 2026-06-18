"""Interactive demo server for the Grounding DINO project frontend."""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LIVE_DIR = PROJECT_ROOT / "outputs" / "frontend_live"
UPLOAD_DIR = LIVE_DIR / "uploads"
ANNOTATED_DIR = LIVE_DIR / "annotated"

sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import Predictor  # noqa: E402
from src.inference.visualizer import save_annotated_image  # noqa: E402

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_predictor: Predictor | None = None
_predictor_lock = threading.Lock()


def _get_predictor() -> Predictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = Predictor(
                config_path=str(PROJECT_ROOT / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
                checkpoint_path=str(PROJECT_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        return _predictor


def _clean_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt.strip())
    if not prompt:
        raise ValueError("Text prompt is required.")
    return prompt


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(PROJECT_ROOT / "outputs", filename)


@app.get("/live_outputs/<path:filename>")
def live_outputs(filename: str):
    return send_from_directory(LIVE_DIR, filename)


@app.get("/result.txt")
def result_txt():
    return send_from_directory(PROJECT_ROOT, "result.txt")


@app.get("/api/status")
def status():
    checkpoint = PROJECT_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
    return jsonify(
        {
            "checkpoint": checkpoint.exists(),
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model_loaded": _predictor is not None,
        }
    )


@app.post("/api/infer")
def infer():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image file first."}), 400

    image_file = request.files["image"]
    text_prompt = _clean_prompt(request.form.get("text", ""))
    box_threshold = float(request.form.get("box_threshold", "0.35"))
    text_threshold = float(request.form.get("text_threshold", "0.25"))

    suffix = Path(image_file.filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return jsonify({"error": "Supported image types: jpg, jpeg, png, bmp, webp."}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    stem = secure_filename(Path(image_file.filename or "image").stem) or "image"
    run_id = f"{int(time.time() * 1000)}_{stem}"
    input_path = UPLOAD_DIR / f"{run_id}{suffix}"
    output_path = ANNOTATED_DIR / f"{run_id}_annotated.jpg"
    image_file.save(input_path)

    start = time.time()
    predictor = _get_predictor()
    result = predictor.predict(
        image_path=str(input_path),
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    image = cv2.imread(str(input_path))
    save_annotated_image(
        image=image,
        output_path=str(output_path),
        boxes=np.array(result["boxes"]) if result["boxes"] else None,
        scores=result["scores"],
        phrases=result["phrases"],
        score_threshold=0.0,
    )

    result["total_time"] = round(time.time() - start, 3)
    result["annotated_url"] = f"/live_outputs/annotated/{output_path.name}"
    result["input_url"] = f"/live_outputs/uploads/{input_path.name}"
    result["text_prompt"] = text_prompt
    result["box_threshold"] = box_threshold
    result["text_threshold"] = text_threshold
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
