#!/usr/bin/env python
# coding: utf-8

r"""PREPROCESSING.
Defines the preprocessing functions that will be used to train the model.
"""


import numpy as np
import tensorflow as tf
from astropy.io import fits

def mad(x):
    r"""Compute an estimation of the standard deviation 
    of a Gaussian distribution using the robust 
    MAD (Median Absolute Deviation) estimator."""
    return 1.4826*np.median(np.abs(x - np.median(x)))

def STD_estimator(image, window):    
    # Calculate noise std dev
    return mad(image[window])


def calculate_window(im_shape=(51, 51), win_rad=14):
    # Calculate window function for estimating the noise
    # We take the center of the image and a large radius to cut all the flux from the star
    window = np.ones(im_shape, dtype=bool)
    
    for coord_x in range(im_shape[0]):
        for coord_y in range(im_shape[1]):
            if np.sqrt((coord_x - im_shape[0]/2)**2 + (coord_y - im_shape[1]/2)**2) <= win_rad :
                window[coord_x, coord_y] = False

    return window

def add_noise_function(image, snr_range, tf_window, noise_estimator=True):
    # Draw random SNR
    snr = tf.random.uniform(
            (1,),
            minval=snr_range[0],
            maxval=snr_range[1],
            dtype=tf.float64
        )
    
    # Apply the noise
    im_shape = tf.cast(tf.shape(image), dtype=tf.float64)
    sigma_noise = tf.math.sqrt(tf.norm(image)/(snr * im_shape[0] * im_shape[1]))
    noisy_image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=sigma_noise, dtype=tf.float64)
    norm_noisy_img = noisy_image / tf.norm(noisy_image)

    # Apply window to the normalised noisy image
    windowed_img = tf.boolean_mask(norm_noisy_img, tf_window)
    if noise_estimator:       
        return norm_noisy_img, tf.reshape(tf.numpy_function(mad,[windowed_img], Tout=tf.float64), [1,])
    else:
        return norm_noisy_img


def eigenPSF_data_gen(path, 
              snr_range,
              img_shape=(51, 51),
              batch_size=1,
              win_rad=14,
              n_shuffle=20,
              noise_estimator=True):
    """ Dataset generator of eigen-PSFs.

    On-the-fly addition of noise following a SNR distribution.
    We also calculate the noise std.

    """
    # Init dataset from file
    ds = tf.data.Dataset.from_tensor_slices(path)
    # Cast SNR range 
    tf_snr_range = tf.cast(snr_range, dtype=tf.float64)
    # Create window for noise estimation
    tf_window = tf.cast(calculate_window(
        im_shape=(img_shape[0], img_shape[1]), win_rad=win_rad),
        dtype=tf.bool)
    tf_window = tf.reshape(tf_window, (img_shape[0], img_shape[1], 1))

    # Apply noise and estimate noise std
    image_noisy_ds = ds.map(
        lambda x: (add_noise_function(x, tf_snr_range, tf_window, noise_estimator=noise_estimator), x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    image_noise_ds = image_noisy_ds.shuffle(buffer_size=n_shuffle*batch_size)
    image_noisy_ds = image_noisy_ds.batch(batch_size)
    image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds




