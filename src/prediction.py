"""
Prediction module: loads the trained model once and classifies single images.
Used by the FastAPI /predict endpoint and by the notebook test section.
"""

from __future__ import annotations

from src.preprocessing import CLASS_NAMES, preprocess_image

_model = None


def get_model():
    """Load the trained model lazily and cache it in memory."""
    global _model
    if _model is None:
        from src.model import load_model

        _model = load_model()
    return _model


def reload_model():
    """Force a reload (called after retraining replaces the model file)."""
    global _model
    _model = None
    return get_model()


def predict_image(data: bytes) -> dict:
    """Classify one uploaded chest X-ray image.

    Returns the predicted label, the pneumonia probability and confidence.
    """
    model = get_model()
    x = preprocess_image(data)
    prob = float(model.predict(x, verbose=0)[0][0])  # P(PNEUMONIA)
    label = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return {
        "prediction": label,
        "pneumonia_probability": round(prob, 4),
        "confidence": round(confidence, 4),
    }
