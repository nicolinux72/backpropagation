import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Costruzione del modello SENZA bias
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(3, activation='tanh', use_bias=False),
    layers.Dense(2, activation='linear', use_bias=False)
])

#model.summary()

# set weights as in the article
w1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])

w2 = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]])

model.layers[0].set_weights([w1.T])
model.layers[1].set_weights([w2.T])

# show weights just set
print("--- Weights")
print("Layer 1:\n", model.layers[0].get_weights()[0].T, "\n")
print("Layer 2:\n", model.layers[1].get_weights()[0].T, "\n")

# set also a fake data point and label
x = tf.constant([[1.0, 2.0]])
y_true = tf.constant([[1.0, 1.15]])

# setup auto-differentiation
with tf.GradientTape() as tape:
    y_pred = model(x)                                     # Forward pass
    loss = tf.reduce_mean(tf.square(y_true - y_pred))     # MSE (L2)

# Compute gradient
grads = tape.gradient(loss, model.trainable_variables)

# show gradients for our datapoint (iteration)
#for i, grad in enumerate(grads):
#    print(f"\nGradient layer {i+1}:\n{grad.numpy().T}")

# # show gradients for our datapoint (direct)
print("--- Input:", x.numpy(), "\n")
print("--- Gradients")
print(f"Layer 1:\n{grads[0].numpy().T}")
print("Output:", model.layers[0](x).numpy(), "\n")
print(f"Layer 2:\n{grads[1].numpy().T}")
print("Output:", y_pred.numpy(), "\n")
print("--- Label:", y_true.numpy(), "\n")
