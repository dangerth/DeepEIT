import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow import keras


print("loading data")
#might not be best to store as -.json, this is a leftover from earlier experiments
datafile="phantoms_3000_healthylungs.json"
with open(datafile,"r") as file:
	data=json.load(file)
	file.close()

imgtrain=np.array(data["img"])


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,1,1,latent_dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon


#build VAE model
latent_dim=16

encoder_inputs = keras.Input(shape=(128,128,1))
xx=layers.Conv2D(16,5,activation="relu", strides=2,padding='same')(encoder_inputs)
xx=layers.Conv2D(32,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(64,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(128,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(16,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
z_mean = layers.Conv2D(16,4,activation=None, strides=1,padding='valid')(xx)
z_mean = layers.Dense(16,activation=None)(z_mean)
z_log_var = layers.Conv2D(16,4,activation=None, strides=1,padding='valid')(xx)
z_log_var = layers.Dense(16,activation=None)(z_log_var)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(1,1,latent_dim))
xx=layers.Conv2DTranspose(16,4,strides=1,padding="valid")(latent_inputs)
xx = layers.Conv2DTranspose(128, 5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx = layers.Conv2DTranspose(64, 5, activation="relu", strides=2, padding="same")(xx)
xx=layers.BatchNormalization()(xx)
xx = layers.Conv2DTranspose(32, 5, activation="relu", strides=2, padding="same")(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2DTranspose(16,5, activation="relu", strides=2, padding="same")(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2DTranspose(1, 5, strides=2, activation='relu', padding="same")(xx)
decoder_outputs=xx
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.gen = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.gen(z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1, 2,3)),axis=0)
            reconstruction_loss = 0.05*tf.reduce_mean(tf.reduce_sum(tf.square(data-reconstruction), axis=(1, 2,3)))
            total_loss = kl_loss + reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

history=vae.fit(imgtrain[:-1], epochs=60, batch_size=25)

#save the model
vae.encoder.save("trained_models/vae_encoder")
vae.gen.save("trained_models/vae_gen")









nsamples=50
#Calculate samples from the decoder
noise=tf.random.normal(shape=(nsamples,1,1,16))
pred1=vae.gen(noise)

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(imgtrain[-1:]))
ax.set_aspect("equal")
ax.set_title("true, not in training data")

ax = axes[1]
im = ax.imshow(np.squeeze(pred1[-1,]))
ax.set_aspect("equal")
ax.set_title("sample 1")

ax = axes[2]
im = ax.imshow(np.squeeze(pred1[1]))
ax.set_aspect("equal")
ax.set_title("sample2")

ax = axes[3]
im = ax.imshow(np.squeeze(mean))
ax.set_aspect("equal")
ax.set_title("mean")

ax = axes[4]
im = ax.imshow(np.squeeze(var))
ax.set_aspect("equal")
ax.set_title("var")

fig.colorbar(im, cax=axes[5])
fig.suptitle("Learned Distribution with VAE")
plt.show()

fig2=plt.figure()
plt.plot(history.history['loss'], label='MSE (training data)')
plt.title('MSE for EIT reconstruction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
