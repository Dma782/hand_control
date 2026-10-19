import numpy as np
from dataset import X_test, y_test
from MLP import MLP

model = MLP()

weights = np.load("model_weights.npz")

model.W1 = weights["W1"]
model.W2 = weights["W2"]
model.W3 = weights["W3"]

model.b1 = weights["b1"]
model.b2 = weights["b2"]
model.b3 = weights["b3"]

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")