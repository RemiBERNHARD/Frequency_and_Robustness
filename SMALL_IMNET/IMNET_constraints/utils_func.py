from __future__ import division
import numpy as np


#Project adv example on l_ inf adv_bound ball centered on clean example
def clip_adv(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        X_clip_adv[j] = X_test[i] + np.clip(diff, -adv_bound, adv_bound)
        j = j +1
    return(X_clip_adv)
    

#Project adv example on l_2 adv_bound ball centered on clean example
def clip_adv_l2(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        norm = np.sqrt(np.maximum(1e-12, np.sum(np.square(diff))))
        # We must *clip* to within the norm ball, not *normalize* onto the# surface of the ball
        factor = np.minimum(1., np.divide(adv_bound, norm))
        diff = diff * factor
        X_clip_adv[j] = X_test[i] + diff
        j = j +1
    return(X_clip_adv)
    

#Function which returns the adversarial accuracy, the number of successful adversarial examples, l_2,l_inf, l_1 and l_0 distances between successful adversarial examples and clean observations
def metrics(model, X_adv, X_test, y_pred, indices_test):    
    adv_pred = np.argmax(model.predict(X_adv), axis = 1)
    adv_acc =  np.mean(np.equal(adv_pred, y_pred[indices_test]))
    l2_distort_success = 0
    linf_distort_success = 0
    l1_distort_success = 0
    l0_distort_success = 0
    l2_distort_fail = 0
    linf_distort_fail = 0
    l1_distort_fail = 0
    l0_distort_fail = 0
    nb_success = 0
    j = 0
    for i in indices_test:
        if (adv_pred[j] != y_pred[i]):
            l2_distort_success = l2_distort_success + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_success = linf_distort_success + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_success = l1_distort_success + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_success = l0_distort_success +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
            nb_success = nb_success + 1     
        if (adv_pred[j] == y_pred[i]):
            l2_distort_fail = l2_distort_fail + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_fail = linf_distort_fail + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_fail = l1_distort_fail + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_fail = l0_distort_fail +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
        j = j+1        
    nb_fail = len(indices_test) - nb_success
    if ((nb_fail != 0) & (nb_success != 0)):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
    elif (nb_fail == 0):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, "non", "non", "non", "non")
    elif (nb_success == 0):
        return(adv_acc, nb_success, "non", "non", "non", "non", l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
        
       
from data_augmentation import aug_freq_low, aug_freq_high, aug_freq_cut
def filt(X_data, type_freq, lim_freq_1, lim_freq_2):
    if type_freq == "low":
        X_data_f = aug_freq_low(X_data, lim_freq_1, 3, "float")
    if type_freq == "high":
        X_data_f = aug_freq_high(X_data, lim_freq_1, 3, "float")
    if type_freq == "cut":
        X_data_f = aug_freq_cut(X_data, lim_freq_1, lim_freq_2, 3, "float")
    if type_freq == "none":
        X_data_f = X_data
    return(X_data_f)


def get_indices(model_dict, Xdata_dict, y_data, nb_agree):
    pred_tab = np.zeros((len(model_dict), len(Xdata_dict[0])))    
    for i in range(len(model_dict)):
        pred_tab[i] = np.argmax(model_dict[i].predict(Xdata_dict[i]), axis = 1)
    indices = list()        
    for j in range(len(Xdata_dict[0])):
        count = np.sum(pred_tab[:,j] == y_data[j])
        if (count == nb_agree):
            indices.append(j)
    return(indices)        

    
def adv_indices(model_dict, Xdata_dict, y_data, indices):
    pred_tab = np.zeros((len(model_dict), len(Xdata_dict[0])))    
    for i in range(len(model_dict)):
        pred_tab[i] = np.argmax(model_dict[i].predict(Xdata_dict[i]), axis = 1)
    agree_max_list = list()    
    agree_label_list = list()
    for j in range(len(Xdata_dict[0])):
        unique, counts = np.unique(pred_tab[:,j], return_counts=True)
        agree_max_list.append(np.max(counts))
        agree_label_list.append(int(unique[np.argmax(counts)]))
    agree_max_list = np.array(agree_max_list)
    ########
    agree_max = np.zeros(len(model_dict), dtype="int")
    for nb_m in range(len(model_dict)): 
        agree_max[nb_m] = np.sum(agree_max_list == nb_m + 1)
        print("Number of examples for which " + str(nb_m+1) + " models agree: " + str(agree_max[nb_m]))        
    ########
    well_p = np.zeros(len(model_dict), dtype="int")
    for nb_m in range(len(model_dict)):
        ind_c = list()    
        for i in range(0, len(agree_label_list)):
            if (agree_max_list[i] == nb_m+1) & (agree_label_list[i] == y_data[indices][i]):
                ind_c.append(i)
        well_p[nb_m] = len(ind_c)
        print("Number of well-classified examples when " + str(nb_m +1) + " models agree:" + str(well_p[nb_m]))
    print(well_p / agree_max)
    print(1 - (agree_max - well_p)/len(indices))


def metrics_freq(model_dict, Xadv_dict, X_test, y_test, indices):
    print("metrics without filtering:")
    for i in range(len(model_dict)):
        print(metrics(model_dict[i], Xadv_dict[0], X_test, y_test, indices))    
    print("metrics with filtering: ")
    for i in range(1, len(model_dict)):
        print(metrics(model_dict[i], Xadv_dict[i], X_test, y_test, indices))

#m_1 = load_model("models/CIFAR10_base.h5")        
#m_2 = load_model("models/CIFAR10_low_freq_8.h5")        
       
#model_dict = {}
#model_dict[0] = m_1
#model_dict[1] = m_2
#
#y_data = y_test[:5]

#Xdata_dict = {}
#Xdata_dict[0] = X_test[:5]
#Xdata_dict[1] = X_test[:5]

#pred_tab = np.zeros((3,5))
#pred_tab[0] = np.arange(5)
#pred_tab[1] = np.array([0,1,2,5,6])
#pred_tab[2] = np.array([0,1,6,5,5])
#
#indices = np.arange(5)

def majority_vote(model_dict, Xdata_dict, y_data, indices):
    pred_tab = np.zeros((len(model_dict), len(Xdata_dict[0])))    
    for i in range(len(model_dict)):
        pred_tab[i] = np.argmax(model_dict[i].predict(Xdata_dict[i]), axis = 1)
    agree_max_list = list()    
    agree_label_list = list()
    for j in range(len(Xdata_dict[0])):
        unique, counts = np.unique(pred_tab[:,j], return_counts=True)
        agree_max_list.append(np.max(counts))
        agree_label_list.append(int(unique[np.argmax(counts)]))
    agree_label_list = np.array(agree_label_list)
    ########
    print("Accuracy using majority vote: " + str(np.mean(np.equal(agree_label_list, y_data[indices]))))    

def average_vote(model_dict, Xdata_dict, y_data, indices):
    pred_tab = np.zeros((len(model_dict), len(Xdata_dict[0]), 10))
    for i in range(len(model_dict)):
        pred_tab[i] = model_dict[i].predict(Xdata_dict[i])
    pred_mean = np.mean(pred_tab, axis=0)
    pred_mean_label = np.argmax(pred_mean, axis=1)
    ########
    print("Accuracy using average prediction: " + str(np.mean(np.equal(pred_mean_label, y_data[indices]))))    




def adv_indices_2(model_dict, Xdata_dict, y_data, indices):
    pred_tab = np.zeros((len(model_dict), len(Xdata_dict[0])))    
    for i in range(len(model_dict)):
        pred_tab[i] = np.argmax(model_dict[i].predict(Xdata_dict[i]), axis = 1)
    agree_max_list = list()    
    agree_label_list = list()
    for j in range(len(Xdata_dict[0])):
        unique, counts = np.unique(pred_tab[:,j], return_counts=True)
        agree_max_list.append(np.max(counts))
        agree_label_list.append(int(unique[np.argmax(counts)]))
    agree_max_list = np.array(agree_max_list)
    ########
    agree_max = np.zeros(len(model_dict), dtype="int")
    for nb_m in range(len(model_dict)): 
        agree_max[nb_m] = np.sum(agree_max_list >= nb_m + 1)
        print("Number of examples for which more than " + str(nb_m+1) + " models agree: " + str(agree_max[nb_m]))        
    ########
    well_p = np.zeros(len(model_dict), dtype="int")
    for nb_m in range(len(model_dict)):
        ind_c = list()    
        for i in range(0, len(agree_label_list)):
            if (agree_max_list[i] >= nb_m+1) & (agree_label_list[i] == y_data[indices][i]):
                ind_c.append(i)
        well_p[nb_m] = len(ind_c)
        print("Number of well-classified examples when more than " + str(nb_m +1) + " models agree:" + str(well_p[nb_m]))
    print(well_p / agree_max)
    print(1 - (agree_max - well_p)/len(indices))




def imnet_load_data():
    X_train = np.load("IMNET_data/X_train.npy")
    X_test = np.load("IMNET_data/X_test.npy")
    y_train = np.load("IMNET_data/y_train.npy")
    y_test = np.load("IMNET_data/y_test.npy")
    return((X_train, y_train), (X_test, y_test))














