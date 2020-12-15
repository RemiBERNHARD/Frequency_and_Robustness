import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]

from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.layers import Input
from wide_resnet import create_wide_residual_network
import keras.backend as K
import tensorflow as tf
import numpy as np
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, CarliniWagnerL2 
from utils_func import metrics

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
model_type = sys.argv[1]
print("Attacking model:" + model_type)

model_for_weights = load_model("models/CIFAR10_" + model_type + ".h5")

type_freq = sys.argv[3]
print("Type of filtering for model: " + type_freq)
lim_freq_1 = int(sys.argv[4])
print("Limite freq 1 for model: " + str(lim_freq_1))
lim_freq_2 = int(sys.argv[5])
print("Limite freq 2 for model: " + str(lim_freq_2))


if (type_freq == "none"):
    model = model_for_weights
else :    
    model_input = Input((32,32,3))
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1,
                      input_tensor=model_input, type_freq=type_freq, low_freq=lim_freq_1, high_freq=lim_freq_2, soft_act="yes")
    for j in range(2, len(model.layers)):        
                    model.layers[j].set_weights(model_for_weights.layers[j-1].get_weights())
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

print("Accuracy on (possibly freq filtered) test set: " + str(model.evaluate(X_test, Y_test, verbose=0)[1]))


#######################################
#Attack model
####################################### 
wrap = KerasModelWrapper(model)

#Select 1000 well-classifier test set examples to craft adversarial examples from them
pred_clean = np.argmax(model.predict(X_test), axis = 1)
well_pred = np.arange(0, len(X_test))[pred_clean == y_test]
indices = np.random.choice(well_pred, 1000, replace=False)


#############################
#FGSM
fgsm_params = {'eps': 0.03,
               }

fgsm_attack = FastGradientMethod(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0, len(indices),200):
    X_adv[i:(i+200)] = fgsm_attack.generate_np(X_test[indices[i:(i+200)]], **fgsm_params)
print(metrics(model, X_adv, X_test, y_test, indices))
X_adv_noise = X_adv + np.random.normal(0,0.05,size=X_adv.shape)
print(metrics(model, X_adv_noise, X_test, y_test, indices))

#############################
#PGD
pgd_params = {'eps': 0.03,
              'eps_iter': 0.01,
              'nb_iter': 100,
              'ord': np.inf,
               'rand_init': True
               }

pgd_attack = ProjectedGradientDescent(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in range(0, len(indices),200):
    X_adv[i:(i+200)] = pgd_attack.generate_np(X_test[indices[i:(i+200)]], **pgd_params)
print(metrics(model, X_adv, X_test, y_test, indices))

#############################
#CWL2
cwl2_params = {'binary_search_steps': 10,
               'max_iterations': 100,
               'learning_rate': 0.1,
               'batch_size': 200,
               'initial_const': 0.5,
               'clip_min': 0.,
               'clip_max': 1.,
               'confidence': 0
               }

cwl2_attack = CarliniWagnerL2(wrap, sess=sess)
X_adv = np.zeros((len(indices),32,32,3))
for i in np.arange(0,len(indices),200):
    X_adv[i:(i+200)] = cwl2_attack.generate_np(X_test[indices[i:(i+200)]], **cwl2_params)
print(metrics(model, X_adv, X_test, y_test, indices))    





















