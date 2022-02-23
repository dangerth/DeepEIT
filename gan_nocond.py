import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import time

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras import regularizers
from tensorflow import keras


print("loading data")
#might not be best to store as -.json, this is a leftover from earlier experiments
datafile="imtrain.json"
with open(datafile,"r") as file:
	data=json.load(file)
	file.close()

#load training data
xtrain=np.array(data["img"],dtype='f') #conductivity images
vtrain=np.array(data["v"],dtype='f') #eit measurements


##Define the ResNet-type block of layers
def Resblock(y0,channels,regpar):
    y=layers.BatchNormalization(epsilon=1e-5)(y0)
    y=layers.LeakyReLU(alpha=0.3)(y)
    y=layers.Conv2D(channels,3,padding='same',kernel_regularizer=regularizers.l2(regpar),bias_regularizer=regularizers.l2(regpar))(y)
    y=layers.BatchNormalization(epsilon=1e-5)(y)
    y=layers.LeakyReLU(alpha=0.3)(y)
    y=layers.Conv2D(channels,3,padding='same',kernel_regularizer=regularizers.l2(regpar),bias_regularizer=regularizers.l2(regpar))(y)
    z=layers.Conv1D(channels,1,padding='same',kernel_regularizer=regularizers.l2(regpar),bias_regularizer=regularizers.l2(regpar))(y0)
    return z+y



#Our Wasserstein-GAN
class CGAN(keras.Model):

		
    def __init__(self):
        # Iinitialize parameters
        super(CGAN, self).__init__()
        print("Dimensions by hand,not yet automatically")
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.datapoints = 208 #automatisieren später
        self.latent_dim = 16 #dimension of the random sampling vector
        
        self.epochs=20
		
        self.reg_disc=0.001 #regularization parameter for drift in discriminator
        self.reg_dgrad=10 #regularization parameter for gradients in discriminator
        self.reg_dpars=1e-4#regularization parameter for discriminator weights
        self.reg_gpars=1e-4 #regularization parameter for generator weights
        self.lr=2e-4 #learning rate for optimizer
        

        self.g_optimizer = keras.optimizers.RMSprop(learning_rate=self.lr)
        self.d_optimizer = keras.optimizers.RMSprop(learning_rate=self.lr)

        # Build the generator and discriminator
        print("init gen")
        self.generator = self.build_generator()
        print("gen done")
        self.discriminator = self.build_discriminator()
        print("disc done")

        # set up loss trackers
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")


    @property
    def metrics(self):
        return [
            self.gen_loss_tracker,
            self.disc_loss_tracker,
        ]

    def build_generator(self):
    	#there is much room to make this more intelligent
        noisein=keras.Input(shape=(self.latent_dim))
        
        xx=layers.Dense(16)(noisein)
        xx=layers.Reshape(target_shape=(1,1,16))(xx)
        xx=layers.Conv2DTranspose(16,4,strides=1,padding="valid")(xx)
        xx = layers.Conv2DTranspose(128, 5, activation="relu", strides=2,padding='same')(xx)
        xx=layers.Conv2D(128,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2DTranspose(64, 5, activation="relu", strides=2, padding="same")(xx)
        xx=layers.Conv2D(64,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2DTranspose(32, 5, activation="relu", strides=2, padding="same")(xx)
        xx=layers.Conv2D(32,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2DTranspose(16,5, activation="relu", strides=2, padding="same")(xx)
        xx=layers.Conv2D(16,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2DTranspose(1, 5, strides=2, activation='relu', padding="same")(xx)
        xx=layers.Conv2D(1,3,padding="same")(xx)
        xx=Resblock(xx,1,0)
        xx=layers.ReLU()(xx)





        generator = keras.Model(noisein, xx, name="generator")
        generator.summary()

        return generator

    def build_discriminator(self):
	#there is much room to make this more intelligent
        xin=keras.Input(shape=(self.img_rows,self.img_cols,1))
        
        xx=layers.Conv2D(16,5,activation="relu", strides=2,padding='same')(xin)
        xx=layers.Conv2D(32,5, activation="relu", strides=2,padding='same')(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(64,5, activation="relu", strides=2,padding='same')(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(128,5, activation="relu", strides=2,padding='same')(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(16,5, activation="relu", strides=2,padding='same')(xx)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2D(16,4,activation="relu", strides=1,padding='valid')(xx)
        xx = layers.Dense(16,activation="relu")(xx)
        xx=layers.Flatten()(xx)
        xx=layers.Dense(128)(xx)
        xx=layers.LeakyReLU(alpha=0.2)(xx)
        xx=layers.Dense(1)(xx)
        discriminator=keras.Model([xin], xx, name="discriminator")

        discriminator.summary()


        return discriminator

 

print("initializing")
cgan = CGAN()
cgan.compile()
print("initialization done")

batchsize=25
epochs=150
it_gen=3
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain[:-1])) #load training data for learning task
train_dataset = train_dataset.batch(batchsize,drop_remainder=True) #prepare batches

print("start loop")

for epoch in range(epochs):
	print("Start epoche %i"%(epoch+1))
	start_time=time.time()
	for step, x_batch_train in enumerate(train_dataset):
	
		if step % it_gen !=0:
			latent_samples=keras.backend.random_normal(shape=(batchsize,cgan.latent_dim))
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(cgan.generator.trainable_variables)				
				image_fake=cgan.generator(latent_samples)
				D_fake=cgan.discriminator(image_fake)
				D_real=cgan.discriminator(x_batch_train)
				lossGAN=tf.reduce_mean(D_fake-D_real)
				gen_losses=sum(cgan.generator.losses)
				total_loss_G=lossGAN+gen_losses
			gradsG=tape.gradient(total_loss_G,cgan.generator.trainable_variables)
			cgan.g_optimizer.apply_gradients(zip(gradsG,cgan.generator.trainable_variables))
			
		else:
			latent_samples=keras.backend.random_normal(shape=(batchsize,cgan.latent_dim))
			eps=tf.random.uniform(shape=[],minval=0,maxval=1)

			x_interpol=tf.add(tf.math.scalar_mul(eps,x_batch_train),tf.math.scalar_mul((1.0-eps),cgan.generator(latent_samples)))
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(cgan.discriminator.trainable_variables)
				image_fake=cgan.generator(latent_samples)
				D_fake=cgan.discriminator(image_fake)
				D_real=cgan.discriminator(x_batch_train)
				lossGAN=tf.reduce_mean(D_real-D_fake)
				disc_losses=sum(cgan.discriminator.losses)
				loss_drift=tf.square(D_real)
				with tf.GradientTape(watch_accessed_variables=False) as innertape:
					innertape.watch(x_interpol)
					D_interpol=cgan.discriminator(x_interpol)
				D_grads_x=innertape.gradient(D_interpol,x_interpol)
				loss_grads=tf.square(tf.sqrt(tf.reduce_sum(tf.square(D_grads_x)))-1)
				total_loss_D=lossGAN+cgan.reg_disc*loss_drift+disc_losses+cgan.reg_dgrad*loss_grads
			gradsD=tape.gradient(total_loss_D,cgan.discriminator.trainable_variables)
			cgan.d_optimizer.apply_gradients(zip(gradsD,cgan.discriminator.trainable_variables))
				

	print("Time for epoche: %.2fs" % (time.time() - start_time))
	

#Plot learned distribution
nsamples=50
#Calculate two samples from the generator
noise=tf.random.normal(shape=(nsamples,cgan.latent_dim)) #create random noise
pred1=cgan.generator(noise) #forward pass

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(xtrain[-1:]))
ax.set_aspect("equal")
ax.set_title("conductivity not in training data")

ax = axes[1]
im = ax.imshow(np.squeeze(pred1[-1,]))
ax.set_aspect("equal")
ax.set_title("learned sample 1")

ax = axes[2]
im = ax.imshow(np.squeeze(pred1[1]))
ax.set_aspect("equal")
ax.set_title("learned sample 2")

ax = axes[3]
im = ax.imshow(np.squeeze(mean))
ax.set_aspect("equal")
ax.set_title("mean of learned distribution")

ax = axes[4]
im = ax.imshow(np.squeeze(var))
ax.set_aspect("equal")
ax.set_title("variance of learned distribution")
fig.suptitle("Learned distribution with WGAN")
fig.colorbar(im, cax=axes[5])
plt.show()


nsamples=50
#Calculate two samples from the generator
noise=tf.random.normal(shape=(nsamples,cgan.latent_dim))
print(xtrain[:1].shape)
#xrep=tf.repeat(vtrain[:1],nsamples,axis=0)
#print(xrep.shape)
pred1=cgan.generator(noise)

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(xtrain[1,]))
ax.set_aspect("equal")
ax.set_title("image from training set")

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
fig.suptitle("Learned Distribution with WGAN")
plt.savefig("trained.png")
plt.show()

