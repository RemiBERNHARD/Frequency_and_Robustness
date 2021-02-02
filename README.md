# Towards better understanding the impact of frequency characteristics on adversarial robustness

This repository contains python scripts to reproduce results presented in the article "Towards better understanding the impact of frequency characteristics on adversarial robustness".

## Environment and libraries

The python scripts were executed in the following environment:

* OS: CentOS Linux 7
* GPU: NVIDIA GeForce GTX 1080 
* Cuda version: 9.0.176
* Python version: 2.7.5

The following version of some Python packages are necessary: 

* Tensorflow: 1.12.0
* Cleverhans: 3.0.1
* Keras: 2.2.4
* Numpy: 1.16.12


## File structure

This repository is divided at a high-level based on the data set considered (CIFAR10, SVHN or Small Imagenet). Inside a repository corresponding to a data set X are two repositories:
- X_freq_analysis: code files to train models on filtered data sets and to evaluate transferability of adversarial perturbations between them.

As an example, for the CIFAR10 data set, go to the CIFAR10/CIFAR10_freq_analysis folder.

1. To train a model with images low-pass filtered at intensity 10, run:

        python train_cifar10.py low_freq 0 10 
    
(The 0 enables to use the first available GPU only.)

2. To evaluate transferability from a natural model to a model trained on image high-pass filtered at intensity 8, run:
 
        python transfer_base_to_freq.py base 0 low_freq_8 low 8

- X_constraints : code files to train models (classical training or Adversarial Training) with the loss functions ```L^{low}```, ```L^{high}``` and ```L^{all}```

As an example, for the SVHN data set, go to the SVHN/SVHN_freq_analysis folder.

1. To train model with loss ```L^{low}_6 ```,  run:

         python train_svhn.py freq_const_lowf 0 6

2. To train a model with Adversarial Training with loss ```L_{AT,all,2,10} ```, run:

         python train_svhn_adv.py madry_cons_all 0 2 10
   
3. To attack this model with the ```l_{\infty}``` PGD attack <sup>[1](#madry_pgd)</sup> with a perturbation budget ``` \epsilon = 0.03```, and 5000 iterations,
run:

         python attack_svhn_adv.py madry_cons_all_2_10 0 5000


## Data files

In order to get the SVHN (or Small Imagenet) data set (to have the same training, validation and testing tests, as well as to run attacks), download the "SVHN_data" (or "Small_IMNET") folders from https://drive.google.com/drive/folders/18FJE0lPe0QPzLyAj0ATZWFO1DU31z4sH and place them in the same directory as all other files.


<a name="madry_pgd">1</a>: Aleksander Madry,  Aleksandar Makelov,  Ludwig Schmidt,Dimitris Tsipras,  and Adrian Vladu. Towards deep learn-ing models resistant to adversarial attacks. In *InternationalConference on Learning Representations, 2018*

### Details about Small Imagenet
The Small Imagenet data set is built by extracting 10 meta classes from the ImageNet ILSVRC2012 benchmark. For each meta class, we extracted 3000 images from the original training set and 300 images from the non-blacklisted validation set.
Each of the 10 meta classes consists in 3 classes from the original ImageNet data set. The 10 meta classes with the corresponding label indices of the 3 real classes are as follows:
* shark: 2 3 4
* bird: 85 86 87
* frog: 30 31 32
* dog: 153 154 155
* cat: 281 282 283
* bear: 294,295,296
* monkey: 367 368 369
* ball: 768, 805, 852
* chair: 423, 559, 765
*truck: 569, 717, 867




