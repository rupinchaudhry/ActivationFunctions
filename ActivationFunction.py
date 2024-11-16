import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Softmax Activation Function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=0)

# Plotting the activation functions
x = np.linspace(-10, 10, 100)  # Generate 100 points between -10 and 10

plt.figure(figsize=(10, 8))

# Sigmoid Plot
plt.subplot(3, 2, 1)
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.title("Sigmoid")
plt.grid(True)

# Tanh Plot
plt.subplot(3, 2, 2)
plt.plot(x, tanh(x), label="Tanh", color='orange')
plt.title("Tanh")
plt.grid(True)

# ReLU Plot
plt.subplot(3, 2, 3)
plt.plot(x, relu(x), label="ReLU", color='green')
plt.title("ReLU")
plt.grid(True)

# Leaky ReLU Plot
plt.subplot(3, 2, 4)
plt.plot(x, leaky_relu(x), label="Leaky ReLU", color='red')
plt.title("Leaky ReLU")
plt.grid(True)

# Softmax Plot
plt.subplot(3, 2, 5)
softmax_x = np.linspace(-2, 2, 5)  # Use a smaller range for softmax
plt.plot(softmax_x, softmax(softmax_x), label="Softmax", color='purple')
plt.title("Softmax (1D Example)")
plt.grid(True)

plt.tight_layout()
plt.show()
