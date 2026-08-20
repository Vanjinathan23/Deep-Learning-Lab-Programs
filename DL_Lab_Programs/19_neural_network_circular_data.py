import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

X, y = make_circles(
    n_samples=300, noise=0.08, factor=0.5, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation="identity",
    learning_rate_init=0.03,
    max_iter=2000,
    random_state=42
)

model.fit(X, y)
y_pred = model.predict(X)

print("Neural Network Analysis - Circular Data")
print("Learning Rate: 0.03")
print("Activation: Linear")
print("Hidden Layers: 2")
print("Hidden Neurons: 3")
print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network - Circular Data")
plt.show()
