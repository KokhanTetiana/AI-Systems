import sys
sys.path.insert(0, r"D:\PythonPackages")

import numpy as np
import tensorflow as tf

print("TensorFlow версія:", tf.__version__)

n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

k = tf.Variable(tf.random.normal((1, 1)), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

display_step = 1000
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = X_batch * k + b
        loss = tf.reduce_sum((y_batch - y_pred) ** 2)

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (i + 1) % display_step == 0:
        print(f'Епоха {i + 1}: Втрати={loss.numpy():.4f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}')

print("Навчання завершено!")
print("Кінцевий k:", k.numpy()[0][0])
print("Кінцевий b:", b.numpy()[0])
