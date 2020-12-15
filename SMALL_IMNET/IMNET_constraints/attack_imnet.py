import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf
import numpy as np
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, CarliniWagnerL2, SPSA
from utils_func import metrics, imnet_load_data

sess = tf.Session()
K.set_session(sess)
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)


#######################################
#Load data set
#######################################
(X_train, y_train), (X_test, y_test) = imnet_load_data()
 
X_train = X_train.reshape(X_train.shape[0], 224, 224, 3)
X_test = X_test.reshape(X_test.shape[0], 224, 224, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)



#######################################
#Load model
#######################################
model_type = sys.argv[1]
print("Attacking model: " + model_type)
model = load_model("models/IMNET_" + model_type + ".h5")

print("Accuracy on test set: " + str(model.evaluate(X_test, Y_test, verbose=0)[1]))


#######################################
#Attack model
#######################################
wrap = KerasModelWrapper(model)


#Select 1000 well-classifier test set examples to craft adversarial examples from them
pred_clean = np.argmax(model.predict(X_test), axis = 1)
well_pred = np.arange(0, len(X_test))[pred_clean == y_test]
indices = np.random.choice(well_pred, 1000, replace=False)


#FGSM
fgsm_params = {'eps': 0.03,
               #'clip_min': 0.,
               #'clip_max': 1.
               }

fgsm_attack = FastGradientMethod(wrap, sess=sess)
X_adv = np.zeros((len(indices),224,224,3))
for i in range(0, len(indices),200):
    X_adv[i:(i+200)] = fgsm_attack.generate_np(X_test[indices[i:(i+200)]], **fgsm_params)
print("Accuracy on adv examples")
print(metrics(model, X_adv, X_test, y_test, indices))


#PGD
pgd_params = {'eps': 0.03,
              'eps_iter': 0.01,
              'nb_iter': int(sys.argv[3]),
              'ord': np.inf,
               'clip_min': 0.,
               'clip_max': 1.,
               'rand_init': True
               }

pgd_attack = ProjectedGradientDescent(wrap, sess=sess)
X_adv = np.zeros((len(indices),224,224,3))
for i in range(0, len(indices),100):
    print(i)
    X_adv[i:(i+100)] = pgd_attack.generate_np(X_test[indices[i:(i+100)]], **pgd_params)
print("Accuracy on adv examples")
print(metrics(model, X_adv, X_test, y_test, indices))
