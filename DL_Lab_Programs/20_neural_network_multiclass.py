import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=300, n_features=2, n_informative=2,
    n_redundant=0, n_classes=3, n_clusters_per_class=1,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation="identity",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)

model.fit(X, y)
y_pred = model.predict(X)

print("Neural Network Analysis - Multi Class")
print("Learning Rate: 0.01")
print("Activation: Linear")
print("Hidden Layers: 2")
print("Hidden Neurons: 2")
print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network - Multi Class Data")
plt.show()
