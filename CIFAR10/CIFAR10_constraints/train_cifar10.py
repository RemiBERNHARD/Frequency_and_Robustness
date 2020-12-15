import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]


from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import tensorflow as tf
import numpy as np
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
    model.fit_generator(generator.flow(X_train, Y_train, batch_size=128), 
                        epochs=100, steps_per_epoch=len(X_train)//100, 
                        verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)
    model.load_weights("cifar10_weights_best_" + model_type + ".hdf5")
    
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_base.h5")
    
    
    
if (model_type == "freq_const_lowf"):    
    
    from utils_layers_freq import fft_low_pass
    lim_freq = int(sys.argv[3])
    print("Freq limit: " + str(lim_freq))    

    generator=ImageDataGenerator(rotation_range=10,
                             width_shift_range=5./32,
                             height_shift_range=5./32,
                             horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    batch_size = 128
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    
    coeff = tf.placeholder(tf.float32, shape=())
    y = tf.placeholder(tf.float32, shape=(None, 10))
    x_lowf = fft_low_pass(model.inputs[0], lim_freq, nb_channels=3)
    
    sess = tf.Session()
    K.set_session(sess)
    
    #Loss    
    model_logits = model(model.inputs[0])._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_low = model(x_lowf)._op.inputs[0]
    cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
    tot_loss =  ce_loss + coeff*cons_loss    
    
    step_size_schedule = [[0, 0.1], [35000, 0.01], [56000, 0.001]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
    opt_op = optimizer.minimize(tot_loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
    acc_tamp = model.evaluate(X_val, Y_val, verbose=0)[1]
    
    for step in np.arange(0, 70000):  
        x_batch, y_batch = next(generator)
        if (step <= 7000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1,
                     K.learning_phase(): 1 })
        if (step > 7000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0, 
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.75,
                     K.learning_phase(): 1 })    
        if (step > 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 2.5,
                     K.learning_phase(): 1 })
        if (step % 100 == 0):
            print(step)
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
            print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
            print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
        if (step % 1000 == 0) & (step > 45000):
            acc_val = model.evaluate(X_val, Y_val, verbose=0)[1]
            print("Accuracy on validation set: ", acc_val)
            if (acc_val > acc_tamp):
                acc_tamp = acc_val
                print("Best accuracy on validation set so far: ", acc_tamp)
                model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq) + ".hdf5")                      
                
    model.load_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq) + ".hdf5")    
     
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_" + model_type + "_" + str(lim_freq) + ".h5")    
    
    
    
if (model_type == "freq_const_highf"):    
   
    from utils_layers_freq import fft_high_pass
    lim_freq = int(sys.argv[3])
    print("Freq limit: " + str(lim_freq))    

    generator=ImageDataGenerator(rotation_range=10,
                             width_shift_range=5./32,
                             height_shift_range=5./32,
                             horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    batch_size = 100
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    
    coeff = tf.placeholder(tf.float32, shape=())
    y = tf.placeholder(tf.float32, shape=(None, 10))
    x_highf = fft_high_pass(model.inputs[0], lim_freq, nb_channels=3)
    
    sess = tf.Session()
    K.set_session(sess)
    
    #Loss    
    model_logits = model(model.inputs[0])._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_high = model(x_highf)._op.inputs[0]
    cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))
    
    tot_loss =  ce_loss + coeff*cons_loss    
    
    step_size_schedule = [[0, 0.1], [35000, 0.01], [56000, 0.001]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
    opt_op = optimizer.minimize(tot_loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
    acc_tamp = model.evaluate(X_val, Y_val, verbose=0)[1]
    
    for step in np.arange(0, 70000):  
        x_batch, y_batch = next(generator)
        if (step <= 7000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1,
                     K.learning_phase(): 1 })
        if (step > 7000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0, 
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.75,
                     K.learning_phase(): 1 })    
        if (step > 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 2.5,
                     K.learning_phase(): 1 })
        if (step % 100 == 0):
            print(step)
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
            print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
            print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
        if (step % 1000 == 0) & (step > 45000):
            acc_val = model.evaluate(X_val, Y_val, verbose=0)[1]
            print("Accuracy on validation set: ", acc_val)
            if (acc_val > acc_tamp):
                acc_tamp = acc_val
                print("Best accuracy on validation set so far: ", acc_tamp)
                model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq) + ".hdf5")          

    model.load_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq) + ".hdf5")    
     
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_" + model_type + "_" + str(lim_freq) + ".h5")        


    
if (model_type == "freq_const_all"):    
        
    from utils_layers_freq import fft_low_pass
    from utils_layers_freq import fft_high_pass
    lim_freq_low = int(sys.argv[3])
    lim_freq_high = int(sys.argv[4])
    print("Freq limit low: " + str(lim_freq_low))    
    print("Freq limit high: " + str(lim_freq_high))    

    generator=ImageDataGenerator(rotation_range=10,
                             width_shift_range=5./32,
                             height_shift_range=5./32,
                             horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    sess = tf.Session()
    K.set_session(sess)
  
    batch_size = 100
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    
    coeff_low = tf.placeholder(tf.float32, shape=())
    coeff_high = tf.placeholder(tf.float32, shape=())
    
    y = tf.placeholder(tf.float32, shape=(None, 10))
    x_lowf = fft_low_pass(model.inputs[0], lim_freq_low, nb_channels=3)
    x_highf = fft_high_pass(model.inputs[0], lim_freq_high, nb_channels=3)
    
    #Loss    
    model_logits = model(model.inputs[0])._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_low = model(x_lowf)._op.inputs[0]
    cons_loss_1 = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
    model_logits_high = model(x_highf)._op.inputs[0]
    cons_loss_2 = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))
    
    tot_loss =  ce_loss + coeff_low*cons_loss_1 + coeff_high*cons_loss_2   

    step_size_schedule = [[0, 0.1], [35000, 0.01], [56000, 0.001]]
    global_step = tf.train.get_or_create_global_step()
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values) 
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
    opt_op = optimizer.minimize(tot_loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
        
    acc_tamp = model.evaluate(X_test, Y_test, verbose=0)[1]
    for step in np.arange(0, 70000):  
        x_batch, y_batch = next(generator)
        if (step <= 7000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.1, coeff_high: 0.1,
                     K.learning_phase(): 1 })
        if (step > 7000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.5, coeff_high: 0.5,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 1.0, coeff_high: 1.0,
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 1.75, coeff_high: 1.75,
                     K.learning_phase(): 1 })    
        if (step > 60000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 2.5, coeff_high: 2.5,
                     K.learning_phase(): 1 })
        if (step % 100 == 0):
            print(step)
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(cons_loss_1, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(cons_loss_2, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
        if (step % 1000 == 0) & (step > 45000):
            acc_val = model.evaluate(X_val, Y_val, verbose=0)[1]
            print("Accuracy on validation set: ", acc_val)
            if (acc_val > acc_tamp):
                acc_tamp = acc_val
                print("Best accuracy on validation set so far: ", acc_tamp)
                model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq_low) + "_" + str(lim_freq_high) + ".hdf5")
                                        
    model.load_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq_low) + "_" + str(lim_freq_high) + ".hdf5")    
      
    print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
    print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
    
    model.save("models/CIFAR10_" + model_type + "_" + str(lim_freq_low) + "_" + str(lim_freq_high) + ".h5")        
    
 
    
