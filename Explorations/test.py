import math
import matplotlib.pyplot as plt
import random
import numpy as np

points = 5000
x = np.random.uniform(-1, 1, points)
y = np.random.uniform(-1, 1, points)
d = np.sqrt(x**2 + y**2)

inside = d < 1
outside = d >= 1

plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], color = "blue", label = "(d < 1)")
plt.scatter(x[outside], y[outside], color = "orange", label = "(d > 1)")
plt.title("Simulation of 5000 points")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
