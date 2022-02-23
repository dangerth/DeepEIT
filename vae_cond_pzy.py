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
datafile="imtrain.json"
with open(datafile,"r") as file:
	data=json.load(file)
	file.close()

imgtrain=np.array(data["img"])
vtrain=np.array(data["v"],dtype='f')
print(vtrain.size)

latent_dim = 16

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch,1,1,latent_dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon




encoder_inputs = keras.Input(shape=(128,128,1))
data_input=keras.Input(shape=(208,))
data=layers.Dense(256)(data_input)
data=layers.Reshape(target_shape=(4,4,16))(data)
xx=layers.Conv2D(16,5,activation="relu", strides=2,padding='same')(encoder_inputs)
xx=layers.Conv2D(32,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(64,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(128,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.BatchNormalization()(xx)
xx=layers.Conv2D(16,5, activation="relu", strides=2,padding='same')(xx)
xx=layers.Concatenate()([xx,data])
xx=layers.BatchNormalization()(xx)
z_mean = layers.Conv2D(16,4,activation=None, strides=1,padding='valid')(xx)
z_mean = layers.Dense(16,activation=None)(z_mean)
z_log_var = layers.Conv2D(16,4,activation=None, strides=1,padding='valid')(xx)
z_log_var = layers.Dense(16,activation=None)(z_log_var)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model([encoder_inputs,data_input], [z_mean, z_log_var, z], name="encoder")
encoder.summary()

prior_input=keras.Input(shape=(208,))
pp=layers.Dense(128,activation="relu")(prior_input)
pp=layers.Dense(32,activation="relu")(pp)
prior_log_var=layers.Dense(16,activation=None)(pp)
prior_log_var=layers.Reshape((1,1,16))(prior_log_var)
prior_mean=layers.Dense(16,activation=None)(pp)
prior_mean=layers.Reshape((1,1,16))(prior_mean)
p=Sampling()([prior_mean,prior_log_var])
prior=keras.Model(prior_input,[prior_mean, prior_log_var, p], name="prior")
prior.summary()



latent_inputs = keras.Input(shape=(1,1,latent_dim))
data_input=keras.Input(shape=(208,))
data=layers.Dense(256)(data_input)
data=layers.Reshape(target_shape=(4,4,16))(data)
xx=layers.Conv2DTranspose(16,4,strides=1,padding="valid")(latent_inputs)
xx=layers.Concatenate()([xx,data])
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
decoder = keras.Model([latent_inputs,data_input], decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder,prior, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.prior=prior
        self.decoder = decoder
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
        xin,yin=data
        maxin=tf.math.reduce_max(tf.math.abs(yin))
        yin=yin+tf.keras.backend.random_normal(shape=(tf.shape(yin)[0],208),stddev=0*maxin)  #this step cann add noise to the data simulating nosiey measurement. To add noise set stddev factor from 0 to 0.1 or so  
        with tf.GradientTape() as tape_dec, tf.GradientTape() as tape_enc, tf.GradientTape() as tape_p:
            tape_dec.watch(self.decoder.trainable_weights)
            tape_enc.watch(self.encoder.trainable_weights)
            tape_p.watch(self.prior.trainable_weights)
            p_mean, p_log_var, z_p =self.prior(yin)	    
            z_mean, z_log_var, z = self.encoder((xin,yin))
            reconstruction = self.decoder((z,yin))
            kl_loss = -0.5 * (1 + z_log_var-p_log_var - tf.math.divide(tf.square(p_mean-z_mean),tf.exp(p_log_var)) - tf.math.divide(tf.exp(z_log_var),tf.exp(p_log_var)))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1, 2,3)),axis=0)
            reconstruction_loss = 0.05*tf.reduce_mean(tf.reduce_sum(tf.square(xin-reconstruction), axis=(1, 2,3)))
            total_loss = kl_loss + reconstruction_loss
        grads_dec = tape_dec.gradient(total_loss, self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_dec, self.decoder.trainable_weights))
        grads_enc = tape_enc.gradient(total_loss, self.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_enc, self.encoder.trainable_weights))
        grads_p = tape_p.gradient(kl_loss, self.prior.trainable_weights) #Note: sign of KL term reversed above
        self.optimizer.apply_gradients(zip(grads_p, self.prior.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

''' Old step all-in one
    def train_step(self, data):
        xin,yin=data
        maxin=tf.math.reduce_max(tf.math.abs(yin))
        yin=yin+tf.keras.backend.random_normal(shape=(tf.shape(yin)[0],208),stddev=0*maxin)  #this step cann add noise to the data simulating nosiey measurement. To add noise set stddev factor from 0 to 0.1 or so  
        with tf.GradientTape() as tape:
            p_mean, p_log_var, z_p =self.prior(yin)	    
            z_mean, z_log_var, z = self.encoder((xin,yin))
            reconstruction = self.decoder((z,yin))
            kl_loss = -0.5 * (1 + z_log_var-p_log_var - tf.math.divide(tf.square(p_mean-z_mean),tf.exp(p_log_var)) - tf.math.divide(tf.exp(z_log_var),tf.exp(p_log_var)))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1, 2,3)),axis=0)
            reconstruction_loss = 0.05*tf.reduce_mean(tf.reduce_sum(tf.square(xin-reconstruction), axis=(1, 2,3)))
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
'''

data=(imgtrain[:-1],vtrain[:-1])
data1=data[0]
data2=data[1]

vae = VAE(encoder,prior, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
batchsize=25
history=vae.fit(imgtrain[:-1],vtrain[:-1], epochs=20, batch_size=batchsize)

vae.encoder.save("smarter_vae_enc")
vae.decoder.save("smarter_vae_dec")
vae.prior.save("smarter_vae_prior")









nsamples=50
#Calculate two samples from the generator
yrep=tf.repeat(vtrain[-1:],nsamples,axis=0)
dummy1,dummy2,noise=vae.prior(yrep)
pred1=vae.decoder((noise,yrep))

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(imgtrain[-1:]))
ax.set_aspect("equal")
ax.set_title("true")

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
fig.suptitle("Learned Distribution with conditional VAE, no noise in reconstructed data")

fig.colorbar(im, cax=axes[5])
plt.savefig("trained.png")
plt.show()

nsamples=50
#Calculate two samples from the generator
data_for_rec=vtrain[-1:]
datanoise=tf.keras.backend.random_normal(shape=data_for_rec.shape,stddev=0.01*tf.math.reduce_max(tf.math.abs(data_for_rec)))
data_for_rec=data_for_rec+datanoise
yrep=tf.repeat(data_for_rec,nsamples,axis=0)
dummy1,dummy2,noise=vae.prior(yrep)
pred1=vae.decoder((noise,yrep))

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(imgtrain[-1:]))
ax.set_aspect("equal")
ax.set_title("true")

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
fig.suptitle("Learned Distribution with conditional VAE, some noise in data")

fig.colorbar(im, cax=axes[5])
plt.savefig("trained.png")
plt.show()

fig2=plt.figure()
plt.plot(history.history['loss'], label='MSE (training data)')
#plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
plt.title('MSE for EIT reconstruction')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
plt.savefig("loss.png")
