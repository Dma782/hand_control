import numpy as np
from models.MLP import MLP
import math

model = MLP()

weights = np.load("model_weights.npz")

model.W1 = weights["W1"]
model.W2 = weights["W2"]
model.W3 = weights["W3"]

model.b1 = weights["b1"]
model.b2 = weights["b2"]
model.b3 = weights["b3"]


class_names = ["fist", "open_palm", "peace"]


def normalize_coords(coords):
    if len(coords) != 63:
        return None

    if not all(math.isfinite(v) for v in coords):
        return None

    x0, y0, z0 = coords[0:3]
    x9, y9, z9 = coords[27:30]

    s = math.sqrt(
        (x9 - x0) ** 2 +
        (y9 - y0) ** 2 +
        (z9 - z0) ** 2
    )

    if s < 1e-6:
        return None

    normalized = []

    for i in range(0, 63, 3):
        x = (coords[i] - x0) / s
        y = (coords[i + 1] - y0) / s
        z = (coords[i + 2] - z0) / s

        normalized.extend([x, y, z])

    return normalized


def determinate(coords):
    y_pred, _ = model.forward(coords)

    gesture_index = np.argmax(y_pred, axis=1)[0]
    confidence = y_pred[0, gesture_index]

    if confidence > 0.8:
        return class_names[gesture_index]

    return None
