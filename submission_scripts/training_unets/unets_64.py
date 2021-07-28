#!/usr/bin/env python
# coding: utf-8

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from denoising.unets.unet import Unet
from denoising.preprocessing import eigenPSF_data_gen
from astropy.io import fits

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print(tf.test.gpu_device_name()) 


# Define paths
base_path = '/gpfswork/rech/xdy/ulx23va/deep_denoising/'
chkp_folder = 'chkp/'
model_folder = 'model/'
log_hist_folder = 'log-hist/'
dataset_path = '/gpfswork/rech/xdy/ulx23va/data/eigenPSF_datasets/'

run_id = f'_{int(time.time())}'
checkpoint_path = base_path + chkp_folder + f'unets_64_{run_id}' + '.hdf5'
summary_path = base_path + log_hist_folder + 'modelsummary_64.txt'
loss_path = base_path + log_hist_folder + 'Loss_64.png'
model_path = base_path + model_folder + 'saving_unets_64'
history_path = base_path + log_hist_folder + 'history_64.npy'


img = fits.open(dataset_path + 'dataset_eigenpsfs.fits')

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

model=Unet(n_output_channels=1, kernel_size=3, layers_n_channels=[64, 128, 256, 512, 1024])
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=adam, loss='mse')

#def l_rate_schedule(epoch):
#        return max(1e-3 / 2**(epoch//25), 1e-5)
                   
#lrate_cback = LearningRateScheduler(l_rate_schedule)                   
       
                   
cp_callback = tf.keras.callbacks.ModelCheckpoint(
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
np.save(history_path, history.history)

