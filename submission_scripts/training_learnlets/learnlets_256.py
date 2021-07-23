#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from denoising.learnlets.learnlet_model import Learnlet
from tensorflow.keras.optimizers import Adam
from denoising.evaluate import keras_psnr, keras_ssim, center_keras_psnr
from denoising.preprocessing import eigenPSF_data_gen
from astropy.io import fits

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.test.gpu_device_name()) 

img = fits.open('/n05data/ayed/outputs/eigenpsfs/dataset_eigenpsfs.fits')
img = img[1].data['VIGNETS_NOISELESS']
img = np.reshape(img, (len(img), 51, 51, 1))

for i in range (len(img)):
    if np.sum(img[i, :, :, :]) < 0:
        img[i, :, :, :] = -img[i, :, :, :]
        
np.random.shuffle(img)

size_train = np.floor(len(img)*0.8)
training, test = img[:int(size_train),:,:], img[int(size_train):,:,:]

training = eigenPSF_data_gen(path=training,
                    snr_range=[0,100],
                    img_shape=(51, 51),
                    batch_size=1)

test = eigenPSF_data_gen(path=test,
                 snr_range=[0,100],
                 img_shape=(51, 51),
                 batch_size=1)

run_params = {
    'denoising_activation': 'dynamic_soft_thresholding',
    'learnlet_analysis_kwargs':{
        'n_tiling': 256, 
        'mixing_details': False,    
        'skip_connection': True,
    },
    'learnlet_synthesis_kwargs': {
        'res': True,
    },
    'threshold_kwargs':{
        'noise_std_norm': True,
    },
#     'wav_type': 'bior',
    'n_scales': 5,
    'n_reweights_learn': 1,
    'clip': False,
}

checkpoint_path = "trained_models/saved_learnlets/cp_256.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=steps*50)

model=Learnlet(**run_params)
n_epochs = 1000
batch_size = 64
steps = np.ceil(len(training)/batch_size)
model.compile(optimizer=Adam(learning_rate=1e-4),
    loss='mse',
    metrics=[keras_psnr, center_keras_psnr],
)

history = model.fit(
    training,
    validation_data=test,
    steps_per_epoch=steps, 
    epochs=n_epochs,
    callbacks=[cp_callback],
    batch_size=batch_size,
)

plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss of the Learnlets 256 on the EigenPSF Dataset')
plt.ylabel('Loss value')
plt.yscale('log')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig("trained_models/saved_learnlets/Loss_256.png")

with open('trained_models/saved_learnlets/modelsummary_256.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

