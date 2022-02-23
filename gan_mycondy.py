import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import time

from keras_visualizer import visualizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow.keras.backend as K


print("loading data")
#might not be best to store as -.json, this is a leftover from earlier experiments
datafile="imtrain.json"
with open(datafile,"r") as file:
	data=json.load(file)
	file.close()


xtrain=np.array(data["img"],dtype='f')
vtrain=np.array(data["v"],dtype='f')


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
        print("dsc done")

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
    #genrerator, can certainly be imporved
        noisein=keras.Input(shape=(self.latent_dim))
        datain=keras.Input(shape=(self.datapoints))
        data=layers.Dense(256)(datain)
        data=layers.Reshape(target_shape=(4,4,16))(data)
        
        xx=layers.Dense(16)(noisein)
        xx=layers.Reshape(target_shape=(1,1,16))(xx)
        xx=layers.Conv2DTranspose(16,4,strides=1,padding="valid")(xx) #(4,4,16)
        xx=layers.Concatenate()([xx,data])
        xx = layers.Conv2DTranspose(128, 5, activation="relu", strides=2,padding='same')(xx) #(8,8,128)
        xx=layers.Conv2D(128,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2DTranspose(64, 5, activation="relu", strides=2, padding="same")(xx) #(16,16,64)
        xx=layers.Conv2D(64,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2DTranspose(32, 5, activation="relu", strides=2, padding="same")(xx) #(32,32,32)
        xx=layers.Conv2D(32,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2DTranspose(16,5, activation="relu", strides=2, padding="same")(xx) #(64,64,16)
        xx=layers.Conv2D(16,3,padding="same")(xx)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2DTranspose(1, 5, strides=2, activation='relu', padding="same")(xx) #(128,128,1)
        xx=layers.Conv2D(1,3,padding="same")(xx)
        xx=Resblock(xx,1,0)
        xx=layers.ReLU()(xx)





        generator = keras.Model([noisein,datain], xx, name="generator")
        generator.summary()

        return generator

    def build_discriminator(self):
        #discriminator, to be discussed. can certainly be improved

        xin=keras.Input(shape=(self.img_rows,self.img_cols,1))
        datain=keras.Input(shape=(self.datapoints))
        data=layers.Dense(256)(datain)
        data=layers.Reshape(target_shape=(4,4,16))(data)
        
        xx=layers.Conv2D(16,5,activation="relu", strides=2,padding='same')(xin) #(64,64,16)
        xx=layers.Conv2D(32,5, activation="relu", strides=2,padding='same')(xx) #(32,32,32)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(64,5, activation="relu", strides=2,padding='same')(xx) #(16,16,64)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(128,5, activation="relu", strides=2,padding='same')(xx) #(8,8,128)
        xx=layers.BatchNormalization()(xx)
        xx=layers.Conv2D(16,5, activation="relu", strides=2,padding='same')(xx) #(4,4,16)
        xx=layers.Concatenate()([xx,data])
        xx=Resblock(xx,16,0)
        xx=layers.BatchNormalization()(xx)
        xx = layers.Conv2D(16,4,activation="relu", strides=1,padding='valid')(xx) #(1,1,16)
        xx = layers.Dense(16,activation="relu")(xx)
        xx=layers.Flatten()(xx)
        xx=layers.Dense(128)(xx)
        xx=layers.LeakyReLU(alpha=0.2)(xx)
        xx=layers.Dense(1)(xx)
        #end of discriminator definition

        discriminator=keras.Model([xin,datain], xx, name="discriminator")

        discriminator.summary()


        return discriminator

 

print("initializing")
cgan = CGAN()
cgan.compile()
print("initialization done")

batchsize=25
epochs=100
it_gen=3
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain[:-1],vtrain[:-1]))
train_dataset = train_dataset.batch(batchsize,drop_remainder=True)

print("start loop")

for epoch in range(epochs):
	print("Start epoche")
	start_time=time.time()
	for step, (x_batch_train,v_batch_train) in enumerate(train_dataset):
	
		if step % it_gen !=0:
			latent_samples=keras.backend.random_normal(shape=(batchsize,cgan.latent_dim))
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(cgan.generator.trainable_variables)				
				image_fake=cgan.generator((latent_samples,v_batch_train))
				D_fake=cgan.discriminator((image_fake,v_batch_train))
				D_real=cgan.discriminator((x_batch_train,v_batch_train))
				lossGAN=tf.reduce_mean(D_fake-D_real)
				gen_losses=sum(cgan.generator.losses)
				total_loss_G=lossGAN+gen_losses
			gradsG=tape.gradient(total_loss_G,cgan.generator.trainable_variables)
			cgan.g_optimizer.apply_gradients(zip(gradsG,cgan.generator.trainable_variables))
			
		else:
			latent_samples=keras.backend.random_normal(shape=(batchsize,cgan.latent_dim))
			eps=tf.random.uniform(shape=[],minval=0,maxval=1)
			fakes=cgan.generator((latent_samples,v_batch_train))
			x_interpol=tf.add(tf.math.scalar_mul(eps,x_batch_train),tf.math.scalar_mul((1.0-eps),fakes))
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(cgan.discriminator.trainable_variables)
				image_fake=cgan.generator((latent_samples,v_batch_train))
				D_fake=cgan.discriminator((image_fake,v_batch_train))
				D_real=cgan.discriminator((x_batch_train,v_batch_train))
				lossGAN=tf.reduce_mean(D_real-D_fake)
				disc_losses=sum(cgan.discriminator.losses)
				loss_drift=tf.square(D_real)
				with tf.GradientTape(watch_accessed_variables=False) as innertape:
					innertape.watch(x_interpol)
					D_interpol=cgan.discriminator((x_interpol,v_batch_train))
				D_grads_x=innertape.gradient(D_interpol,x_interpol)
				loss_grads=tf.square(tf.sqrt(tf.reduce_sum(tf.square(D_grads_x)))-1)
				total_loss_D=lossGAN+cgan.reg_disc*loss_drift+disc_losses+cgan.reg_dgrad*loss_grads
			gradsD=tape.gradient(total_loss_D,cgan.discriminator.trainable_variables)
			cgan.d_optimizer.apply_gradients(zip(gradsD,cgan.discriminator.trainable_variables))
				
			
				
  
	print("Time for epoche: %.2fs" % (time.time() - start_time))
	


nsamples=50
#Calculate two samples from the generator
noise=tf.random.normal(shape=(nsamples,cgan.latent_dim))
print(xtrain[-1:].shape)
yrep=tf.repeat(vtrain[-1:],nsamples,axis=0)
#print(xrep.shape)
pred1=cgan.generator((noise,yrep))

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(xtrain[-1:]))
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
fig.suptitle("Learned Distribution with conditionsl GAN")

fig.colorbar(im, cax=axes[5])
plt.savefig("trained.png")
plt.show()


nsamples=50
#Calculate two samples from the generator
noise=tf.random.normal(shape=(nsamples,cgan.latent_dim))
print(xtrain[:1].shape)
yrep=tf.repeat(vtrain[:1],nsamples,axis=0)
#print(xrep.shape)
pred1=cgan.generator((noise,yrep))

mean=tf.math.reduce_mean(pred1,axis=0)
var=tf.math.reduce_mean(tf.square(tf.subtract(pred1,mean)),axis=0)



#plot stuff
#fig, axes = plt.subplots(1, 6, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1,0.1]})
fig, axes = plt.subplots(1, 5, figsize=(6,6),gridspec_kw={"width_ratios":[1,1,1,1,1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
im = ax.imshow(np.squeeze(xtrain[1,]))
ax.set_aspect("equal")

ax = axes[1]
im = ax.imshow(np.squeeze(pred1[1,]))
ax.set_aspect("equal")

ax = axes[2]
im = ax.imshow(np.squeeze(pred1[-1,]))
ax.set_aspect("equal")

ax = axes[3]
im = ax.imshow(np.squeeze(mean))
ax.set_aspect("equal")

ax = axes[4]
im = ax.imshow(np.squeeze(var))
ax.set_aspect("equal")

#fig.colorbar(im, cax=axes[5])
fig.colorbar(im,ax=axes.ravel().tolist())
plt.savefig("trained.png")
plt.show()

