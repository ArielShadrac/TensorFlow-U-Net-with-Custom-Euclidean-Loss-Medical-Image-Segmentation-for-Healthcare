import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

# Données simulées : 50 IRM 2D (64x64), masques vrais (biomarqueurs)
np.random.seed(42)
n_samples, img_size = 50, 64
X = np.random.rand(n_samples, img_size, img_size, 1)  # Images IRM-like
y_true = np.random.randint(0, 2, size=(n_samples, img_size, img_size, 1))  # Masques binaires

# Réimplémentation de la distance euclidienne (MSE pour loss stable)
@tf.keras.utils.register_keras_serializable()
def euclidean_loss(y_true, y_pred):
    # Différence pixel-wise
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = y_true - y_pred
    return tf.reduce_mean(tf.square(diff))  # MSE = mean of squares (L2^2 / n)

# Modèle U-Net pour la segmentation
def unet_model():
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    
    # Encodeur simple
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    
    # Decodeur
    up1 = UpSampling2D()(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(up1)
    outputs = Conv2D(1, 1, activation='sigmoid')(conv3)  # Masque prédit [0-1]
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=euclidean_loss)  # Compilation avec la perte euclidienne
    return model

model = unet_model()
X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
y_tf = tf.convert_to_tensor(y_true, dtype=tf.float32)       

# Entraînement rapide (ajoute validation_split=0.2 pour réalisme)
history = model.fit(X_tf, y_tf, epochs=25, batch_size=5, verbose=1)  # Verbose=1 pour voir progression
final_loss = model.evaluate(X_tf, y_tf, verbose=0)
print(f"Perte Euclidienne finale (Sur segmentation IRM) : {final_loss:.4f}")

# Viz : Masque prédit vs. vrai (échantillon 0)
pred_mask = model.predict(X_tf[0:1])[0].squeeze()  # [0] pour batch, squeeze pour 2D
y_true_sample = y_true[0].squeeze()
x_sample = X[0].squeeze()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(x_sample, cmap='gray'); axs[0].set_title('IRM Input')
axs[1].imshow(y_true_sample, cmap='gray'); axs[1].set_title('Masque Vrai')
axs[2].imshow(pred_mask, cmap='gray'); axs[2].set_title('Masque Prédit (Euclidienne)')
plt.tight_layout()
plt.show()  # Ou plt.savefig('unet_seg_viz.png') pour GitHub