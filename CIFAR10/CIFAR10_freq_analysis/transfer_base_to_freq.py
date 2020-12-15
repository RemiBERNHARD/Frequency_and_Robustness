import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]

from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import cifar10
import keras.backend as K
import tensorflow as tf
import numpy as np
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from MIM_InputDiverse import MomentumIterativeMethod_Diverse
from utils_func import metrics, get_indices, filt

sess = tf.Session()
K.set_session(sess)
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)

#######################################
#Load data set
#######################################
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
 
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
y_train = y_train[:,0]
y_test = y_test[:,0]


#######################################
#Load model
#######################################
model_source = sys.argv[1]
print("Crafting adversarial examples on model: " + model_source)

target = sys.argv[3]
print("Transferring to model: " + target)

model = load_model("models/CIFAR10_" + model_source + ".h5")

model_target = load_model("models/CIFAR10_" + target + ".h5")


type_freq_t = sys.argv[4]
print("Type of filtering for target model: " + type_freq_t)
lim_freq_t1 = int(sys.argv[5])
print("Limite freq for target model: " + str(lim_freq_t1))

    
X_freq_s = X_test
X_freq_t = filt(X_test, type_freq_t, lim_freq_t1)

print("Source model: Accuracy on test set: " + str(model.evaluate(X_freq_s, Y_test, verbose=0)[1]))
print("Target model: Accuracy on freq test set: " + str(model_target.evaluate(X_freq_t, Y_test, verbose=0)[1]))    


#######################################
#Attack model
#######################################
wrap = KerasModelWrapper(model)

##Select 1000 well-classifier test set examples to craft adversarial examples from them
model_dict = {}
model_dict[0] = model
model_dict[1] = model_target

Xdata_dict = {}    
Xdata_dict[0] = X_freq_s
Xdata_dict[1] = X_freq_t   
    
indices = get_indices(model_dict, Xdata_dict, y_test, len(model_dict))
indices = np.random.choice(indices, 1000, replace=False)


#FGSM
print("FGSM")
fgsm_params = {'eps': 0.03,
               'clip_min': 0.,
               'clip_max': 1.
               }

fgsm_attack = FastGradientMethod(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0, len(indices),200):
    X_adv[i:(i+200)] = fgsm_attack.generate_np(X_test[indices[i:(i+200)]], **fgsm_params)
print("metrics")    
print(metrics(model, X_adv, X_test, y_test, indices))    
print(metrics(model_target, X_adv, X_test, y_test, indices))
X_adv_freq = filt(X_adv, type_freq_t, lim_freq_t1)
print(metrics(model_target, X_adv_freq, X_test, y_test, indices))


#MIM Diverse
print("MIM-DIVERSE")

mim_params = {'eps': 0.03,
              'eps_iter': 0.01,
              'nb_iter': 40,
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.,
               'prob': 0.8
               }

mim_attack = MomentumIterativeMethod_Diverse(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in np.arange(0,len(indices),200):
    X_adv[i:(i+200)] = mim_attack.generate_np(X_test[indices[i:(i+200)]], **mim_params)
print("metrics")    
print(metrics(model, X_adv, X_test, y_test, indices))    
print(metrics(model_target, X_adv, X_test, y_test, indices))
X_adv_freq = filt(X_adv, type_freq_t, lim_freq_t1)
print(metrics(model_target, X_adv_freq, X_test, y_test, indices))

