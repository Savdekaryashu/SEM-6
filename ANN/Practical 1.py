import numpy as np
import matplotlib.pyplot as plt

# Define the input range
x = np.linspace(-10, 10, 400)

# Define activation functions
# 1. Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# 2. Tanh
tanh = np.tanh(x)

# 3. ReLU
relu = np.maximum(0, x)

# 4. Leaky ReLU (alpha = 0.01)
alpha = 0.01
leaky_relu = np.where(x > 0, x, alpha * x)

# Plotting the activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid, label='Sigmoid', color='blue')
plt.plot(x, tanh, label='Tanh', color='red')
plt.plot(x, relu, label='ReLU', color='green')
plt.plot(x, leaky_relu, label='Leaky ReLU', color='purple')

plt.title("Activation Functions in Neural Networks")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
