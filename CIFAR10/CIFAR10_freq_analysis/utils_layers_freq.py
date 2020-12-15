from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_spectral_ops import fft2d, ifft2d
from tensorflow.python import roll as _roll


def fftshift(x, axes=None):
    #x = ops.convert_to_tensor_v2(x)
    if axes is None:
        axes = tuple(range(x.get_shape().ndims))[1:]
        shift = [dim // 2 for dim in x.shape.as_list()[1:]]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return _roll(x, shift, axes)

def ifftshift(x, axes=None):
    #x = ops.convert_to_tensor_v2(x)
    if axes is None:
        axes = tuple(range(x.get_shape().ndims))[1:]
        shift = [-(dim // 2) for dim in x.shape.as_list()[1:]]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return _roll(x, shift, axes)



def fft_low_pass(input_tensor, lim_freq, nb_channels=3):
  if (nb_channels==1):
      input_tensor_c = tf.cast(input_tensor, dtype=tf.complex64)
      mask = np.zeros((input_tensor.shape.as_list()[2],input_tensor.shape.as_list()[2]))
      crow = int(input_tensor.shape.as_list()[2] /2)
      ccol = int(input_tensor.shape.as_list()[2] /2)
      mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 1
      
      fourier_transform = fft2d(tf.squeeze(input_tensor_c, axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = tf.expand_dims(ifft2d(inverse_fft), axis=3)
      
      res = tf.real(inverse_fft) + tf.imag(inverse_fft)
      
  if (nb_channels==3):  
      input_tensor_c = tf.cast(input_tensor, dtype=tf.complex64)
      mask = np.zeros((input_tensor.shape.as_list()[2],input_tensor.shape.as_list()[2]))
      crow = int(input_tensor.shape.as_list()[2] /2)
      ccol = int(input_tensor.shape.as_list()[2] /2)
      mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 1

      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[0]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c1 = tf.real(inverse_fft) + tf.imag(inverse_fft)    
      
      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[1]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c2 = tf.real(inverse_fft) + tf.imag(inverse_fft)
      
      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[2]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c3 = tf.real(inverse_fft) + tf.imag(inverse_fft)        
      
      res = tf.stack([res_c1, res_c2, res_c3], axis=3)
      
      return(res)
      
      
      
      
def fft_high_pass(input_tensor, lim_freq, nb_channels=3):
  if (nb_channels==1):
      input_tensor_c = tf.cast(input_tensor, dtype=tf.complex64)
      mask = np.ones((input_tensor.shape.as_list()[2],input_tensor.shape.as_list()[2]))
      crow = int(input_tensor.shape.as_list()[2] /2)
      ccol = int(input_tensor.shape.as_list()[2] /2)
      mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 0
      
      fourier_transform = fft2d(tf.squeeze(input_tensor_c, axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = tf.expand_dims(ifft2d(inverse_fft), axis=3)
      
      res = tf.real(inverse_fft) + tf.imag(inverse_fft)
      
  if (nb_channels==3):  
      input_tensor_c = tf.cast(input_tensor, dtype=tf.complex64)
      mask = np.ones((input_tensor.shape.as_list()[2],input_tensor.shape.as_list()[2]))
      crow = int(input_tensor.shape.as_list()[2] /2)
      ccol = int(input_tensor.shape.as_list()[2] /2)
      mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 0

      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[0]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c1 = tf.real(inverse_fft) + tf.imag(inverse_fft)    
      
      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[1]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c2 = tf.real(inverse_fft) + tf.imag(inverse_fft)
      
      fourier_transform = fft2d(tf.squeeze(tf.gather(input_tensor_c, axis=3, indices=[2]), axis=3))
      fourier_transform_shift = fftshift(fourier_transform)
      masked = fourier_transform_shift * mask
      inverse_fft = ifftshift(masked)
      inverse_fft = ifft2d(inverse_fft)
      res_c3 = tf.real(inverse_fft) + tf.imag(inverse_fft)        
      
      res = tf.stack([res_c1, res_c2, res_c3], axis=3)
      
      return(res)      
      
