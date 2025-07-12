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

# Imposta pesi fissi per esempio (opzionale)
w1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])  # shape (2,3)

w2 = np.array([[0.1, 0.4],
               [0.2, 0.5],
               [0.3, 0.6]])       # shape (3,2)

model.layers[0].set_weights([w1])
model.layers[1].set_weights([w2])

# Controllo che siano impostati correttamente
print("Layer 1 weights:\n", model.layers[0].get_weights()[0].T)
print("Layer 2 weights:\n", model.layers[1].get_weights()[0].T)


# Singolo input e output atteso
x = tf.constant([[1.0, 2.0]])     # Input: (1, 2)
y_true = tf.constant([[1.0, 1.15]])     # Output atteso: (1, 1.15)

# Calcolo del gradiente
with tf.GradientTape() as tape:
    # Assicura che il tape tenga traccia dei pesi
    #tape.watch(model.trainable_variables)

    y_pred = model(x)                                     # Forward pass
    print("Output del primo livello:", model.layers[0](x).numpy())
    print("Output del secondo livello:", y_pred)                                         # [0.48693424 1.1463418 ]
    loss = tf.reduce_mean(tf.square(y_true - y_pred))     # MSE (L2)

# Calcolo del gradiente della loss rispetto ai pesi
grads = tape.gradient(loss, model.trainable_variables)

# Stampa i gradienti per ciascun layer
for i, grad in enumerate(grads):
    print(f"\nGradiente per i pesi del layer {i+1}:\n{grad.numpy().T}")
