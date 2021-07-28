#!/usr/bin/env python
# coding: utf-8

# In[3]:

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from denoising.unets.unet import Unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from denoising.evaluate import keras_psnr, keras_ssim, center_keras_psnr
from denoising.preprocessing import eigenPSF_data_gen
from astropy.io import fits

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.test.gpu_device_name()) 

run_id = f'unet_{int(time.time())}'
checkpoint_path = f'/home/ayed/github/denoising/trained_models/saved_unets/unets_32_{run_id}' + '.hdf5'
summary_path = '/home/ayed/github/denoising/trained_models/saved_unets/modelsummary_32.txt'
loss_path = "/home/ayed/github/denoising/trained_models/saved_unets/Loss_32.png"
model_path = '/home/ayed/github/denoising/trained_models/saved_unets/saving_unets_32'
history_path = '/home/ayed/github/denoising/trained_models/saved_unets/history_32.npy'

img = fits.open('/n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits')

img = img[1].data['VIGNETS_NOISELESS']
img = np.reshape(img, (len(img), 51, 51, 1))

for i in range (len(img)):
    if np.sum(img[i, :, :, :]) < 0:
        img[i, :, :, :] = -img[i, :, :, :]
        
np.random.shuffle(img)

size_train = np.floor(len(img)*0.95)
training, test = img[:int(size_train),:,:], img[int(size_train):,:,:]

batch_size = 64

training = eigenPSF_data_gen(path=training,
                    snr_range=[0,100],
                    img_shape=(51, 51),
                    batch_size=batch_size,
                    n_shuffle=20,
                    noise_estimator=False)

test = eigenPSF_data_gen(path=test,
                 snr_range=[0,100],
                 img_shape=(51, 51),
                 batch_size=1,
                 noise_estimator=False)

n_epochs = 1000
steps = int(size_train/batch_size)

model=Unet(n_output_channels=1, kernel_size=3, layers_n_channels=[32, 64, 128, 256, 512])
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=adam, loss='mse')

#def l_rate_schedule(epoch):
#        return max(1e-3 / 2**(epoch//25), 1e-5
                   
#lrate_cback = LearningRateScheduler(l_rate_schedule)                   
       
                   
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=False,
    save_freq=int(steps*50))

history = model.fit(training, 
                    validation_data=test, 
                    epochs=n_epochs, 
                    steps_per_epoch=steps,
                    validation_steps=1,
                    callbacks=[cp_callback],
                    shuffle=False)

plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss of the Unets 64 on the EigenPSF Dataset')
plt.ylabel('Loss value')
plt.yscale('log')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig(loss_path)

with open(summary_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
model.save(model_path)
np.save(history_path,history.history)
             


# In[ ]:




