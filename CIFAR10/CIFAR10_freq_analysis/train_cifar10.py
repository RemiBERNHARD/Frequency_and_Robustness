import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from wide_resnet import create_wide_residual_network
from sklearn.model_selection import train_test_split


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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=123)

Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)


#######################################
#Train model
#######################################
model_type = sys.argv[1]
print("Training model:" + model_type)

if (model_type == "base"):

    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    
    print(model.summary())
    print(model.layers[-1].activation)
    
    filepath="cifar10_weights_best_" + model_type + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    def lr_schedule(epoch):
        lr = 1e-1
        if epoch > 90:
            lr = 0.00008
        elif epoch > 75:
            lr = 0.004
        elif epoch > 40:
            lr = 0.02    
        print('Learning rate: ', lr)
        return lr    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, lr_scheduler]
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0), momentum=0.9), metrics=['accuracy'])
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=100), 
                        epochs=100, steps_per_epoch=len(X_train)//100, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("cifar10_weights_best_" + model_type + ".hdf5")
    
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_base.h5")
    


if (model_type == "low_freq"):

    lim_freq = int(sys.argv[3])
    print("limit frequency: " + str(lim_freq))
    
    from data_augmentation import aug_freq_low
    
    X_train = aug_freq_low(X_train, lim_freq, 3, type_r="float")
    X_test = aug_freq_low(X_test, lim_freq, 3, type_r="float")

    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    filepath="cifar10_weights_best_" + model_type + "_"+ str(lim_freq) +".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    def lr_schedule(epoch):
        lr = 1e-1
        if epoch > 90:
            lr = 0.00008
        elif epoch > 75:
            lr = 0.004
        elif epoch > 40:
            lr = 0.02    
        print('Learning rate: ', lr)
        return lr    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, lr_scheduler]
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0), momentum=0.9), metrics=['accuracy'])
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=100), 
                        epochs=100, steps_per_epoch=len(X_train)//100, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("cifar10_weights_best_" + model_type + "_"+ str(lim_freq) +".hdf5")
    
    print("Accuracy on low freq train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on low freq test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
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
    
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_low_freq_" + str(lim_freq) + ".h5")
       
    
    
if (model_type == "high_freq"):

    lim_freq = int(sys.argv[3])
    print("limit frequency: " + str(lim_freq))
    
    from data_augmentation import aug_freq_high

    X_train = aug_freq_high(X_train, lim_freq, 3, type_r="float")
    X_test = aug_freq_high(X_test, lim_freq, 3, type_r="float")
            
    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    filepath="cifar10_weights_best_" + model_type + "_"+ str(lim_freq) +".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    def lr_schedule(epoch):
        lr = 1e-1
        if epoch > 90:
            lr = 0.00008
        elif epoch > 75:
            lr = 0.004
        elif epoch > 40:
            lr = 0.02    
        print('Learning rate: ', lr)
        return lr    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, lr_scheduler]
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0), momentum=0.9), metrics=['accuracy'])
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=100), 
                        epochs=100, steps_per_epoch=len(X_train)//100, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("cifar10_weights_best_" + model_type + "_"+ str(lim_freq) +".hdf5")
    
    print("Accuracy on low freq train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on low freq test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
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
    
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_high_freq_" + str(lim_freq) + ".h5")



if (model_type == "cut_freq"):

    low_cut_freq = int(sys.argv[3])
    high_cut_freq = int(sys.argv[4])
    print("limit frequency low : " + str(low_cut_freq))
    print("limit frequency high : " + str(high_cut_freq))
    
    from data_augmentation import aug_freq_cut
    
    X_train = aug_freq_cut(X_train, low_cut_freq, high_cut_freq, 3, type_r="float")
    X_test = aug_freq_cut(X_test, low_cut_freq, high_cut_freq, 3, type_r="float")

    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    filepath="cifar10_weights_best_" + model_type + "_"+ str(low_cut_freq) + "_" + str(high_cut_freq) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    def lr_schedule(epoch):
        lr = 1e-1
        if epoch > 90:
            lr = 0.00008
        elif epoch > 75:
            lr = 0.004
        elif epoch > 40:
            lr = 0.02    
        print('Learning rate: ', lr)
        return lr    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, lr_scheduler]
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0), momentum=0.9), metrics=['accuracy'])
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=100), 
                        epochs=100, steps_per_epoch=len(X_train)//100, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("cifar10_weights_best_" + model_type + "_"+ str(low_cut_freq) + "_" + str(high_cut_freq) + ".hdf5")
    
    print("Accuracy on low freq train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on low freq test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
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
    
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_cut_freq_" + str(low_cut_freq) + "_" + str(high_cut_freq) +".h5")



            
            
 
 
