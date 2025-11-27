import matplotlib.pyplot as plt
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

for i, color in zip(range(len(y)), ['red','blue','blue','red']):
    plt.scatter(X[i,0], X[i,1], color=color, s=100)

# Лінії для прихованого шару
x = np.linspace(-0.5, 1.5, 100)

# OR нейрон: x1 + x2 = 0.5 => x2 = 0.5 - x1
plt.plot(x, 0.5 - x, 'g--', label='OR neuron')

# AND нейрон: x1 + x2 = 1.5 => x2 = 1.5 - x1
plt.plot(x, 1.5 - x, 'b--', label='AND neuron')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Двошаровий персептрон XOR')
plt.grid(True)
plt.legend()
plt.show()
