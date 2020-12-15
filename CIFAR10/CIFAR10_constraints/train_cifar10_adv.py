import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[2]

from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
import numpy as np
from wide_resnet import create_wide_residual_network
from pgd_attack import pgd_generate


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

X_ho = X_test[:1000]
Y_ho = Y_test[:1000]
X_test = X_test[1000:]
Y_test = Y_test[1000:]


#######################################
#Train model
#######################################

model_type = sys.argv[1]
print("Training model:" + model_type)


if (model_type == "madry"):
    
    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)
    
    model = create_wide_residual_network((32,32,3), nb_classes=10, N=4, k=8, dropout=0.0, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    print(model.summary())
    
    batch_size = 128
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    sess = tf.Session()
    K.set_session(sess)

    x_adv = pgd_generate(model.inputs[0], model, eps=0.03, eps_iter=0.008, nb_iter= 10, y=None, ord=np.inf, clip_min=0.0, clip_max=1.0, 
                             y_target=None, rand_init= True, rand_init_eps= 0.03, clip_grad=False, sanity_checks=True)

    #Loss    
    model_logits = model(model.inputs[0])._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    tot_loss =  ce_loss     
    
    step_size_schedule = [[0, 0.1], [40000, 0.01], [60000, 0.001]]
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

    for step in np.arange(0, 80000):  
        x_batch, y_batch = next(generator)
        x_batch_adv = sess.run(x_adv, feed_dict={model.inputs[0]: x_batch})
        sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch_adv, K.learning_phase(): 1 })
        if (step % 100 ==0):
            print("step: " + str(step))
            print("ce loss: " + str(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100]})))
        if (step % 1000 ==0):
            acc_val = model.evaluate(X_ho, Y_ho, verbose=0)[1]
            print("Accuracy on hold-out set: " + str(acc_val))
            x_adv_eval = np.zeros((1000,32,32,3))
            for i in np.arange(0,1000,100):
                x_adv_eval[i:(i+100)] = sess.run(x_adv, feed_dict={model.inputs[0]: X_ho[i:(i+100)]})
            acc_adv_eval = model.evaluate(x_adv_eval, Y_ho[0:1000], verbose=0)[1]
            print("Accuracy on hold-out adv examples: " + str(acc_adv_eval))
            print("Accuracy on training batch of adv examples: " + str(model.evaluate(x_batch_adv, y_batch, verbose=0)[1]))   
            model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(step) + "_" + str(acc_val) +
                               "_" + str(acc_adv_eval) + ".hdf5")     
             
    
    
if (model_type == "madry_cons_all"):
    
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
    
    print(model.summary())
    
    batch_size = 128
    generator = generator.flow(X_train, Y_train, batch_size=batch_size)
    
    coeff_low = tf.placeholder(tf.float32, shape=())
    coeff_high = tf.placeholder(tf.float32, shape=())
    
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    x_adv = pgd_generate(model.inputs[0], model, eps=0.03, eps_iter=0.008, nb_iter= 10, y=None, ord=np.inf, clip_min=0.0, clip_max=1.0, 
                         y_target=None, rand_init= True, rand_init_eps= 0.03, clip_grad=False, sanity_checks=True)
    
    
    x_lowf = fft_low_pass(x_adv, lim_freq_low, nb_channels=3)
    x_highf = fft_high_pass(x_adv, lim_freq_high, nb_channels=3)
    
    sess = tf.Session()
    K.set_session(sess)
    
    #Loss    
    model_logits = model(x_adv)._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_low = model(x_lowf)._op.inputs[0]
    cons_loss_1 = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
    model_logits_high = model(x_highf)._op.inputs[0]
    cons_loss_2 = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))

    tot_loss =  ce_loss + coeff_low*cons_loss_1 + coeff_high*cons_loss_2    
   
    step_size_schedule = [[0, 0.1], [40000, 0.01], [60000, 0.001]]
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
    
    print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], 
                                                                coeff_low: 0.1, coeff_high: 0.1,}))
    
    for step in np.arange(0, 80000):  
        x_batch, y_batch = next(generator)
        x_batch_adv = sess.run(x_adv, feed_dict={y: y_batch, model.inputs[0]: x_batch})
        if (step <= 10000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.1, coeff_high: 0.1,
                     K.learning_phase(): 1 })
        if (step > 10000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.3, coeff_high: 0.3,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.5, coeff_high: 0.5,
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.7, coeff_high: 0.7,
                     K.learning_phase(): 1 })    
        if (step > 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 1.0, coeff_high: 1.0,
                     K.learning_phase(): 1 })
    
        if (step % 100 == 0):
            print("step: " + str(step))
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(cons_loss_1, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(cons_loss_2, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
        if (step % 1000 == 0):
            acc_val = model.evaluate(X_ho, Y_ho, verbose=0)[1]
            print("Accuracy on hold-out set: " + str(acc_val))
            x_adv_eval = np.zeros((1000,32,32,3))
            for i in np.arange(0,1000,100):
                x_adv_eval[i:(i+100)] = sess.run(x_adv, feed_dict={model.inputs[0]: X_ho[i:(i+100)]})
            acc_adv_eval = model.evaluate(x_adv_eval, Y_ho[0:1000], verbose=0)[1]
            print("Accuracy on hold-out adv examples: " + str(acc_adv_eval))
            print("Accuracy on training batch of adv examples: " + str(model.evaluate(x_batch_adv, y_batch, verbose=0)[1]))       
            model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq_low) + "_" + str(lim_freq_high) +
                               "_" + str(step) + "_" + str(acc_val) + "_" + str(acc_adv_eval) + ".hdf5")     

    
    
if (model_type == "madry_low"):
    
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
    
    x_adv = pgd_generate(model.inputs[0], model, eps=0.03, eps_iter=0.008, nb_iter= 10, y=None, ord=np.inf, clip_min=0.0, clip_max=1.0, 
                         y_target=None, rand_init= True, rand_init_eps= 0.03, clip_grad=False, sanity_checks=True)
    
    
    x_lowf = fft_low_pass(x_adv, lim_freq, nb_channels=3)
    
    sess = tf.Session()
    K.set_session(sess)
    
    #loss    
    model_logits =  model(x_adv)._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_low = model(x_lowf)._op.inputs[0]
    cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
    tot_loss =  ce_loss + coeff*cons_loss
    
    step_size_schedule = [[0, 0.1], [40000, 0.01], [60000, 0.001]]
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
    
    for step in np.arange(0, 80000):  
        x_batch, y_batch = next(generator)
        x_batch_adv = sess.run(x_adv, feed_dict={y: y_batch, model.inputs[0]: x_batch})
        if (step <= 10000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1,
                     K.learning_phase(): 1 })
        if (step > 10000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.3,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5,
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.7,
                     K.learning_phase(): 1 })    
        if (step > 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0,
                     K.learning_phase(): 1 })
        if (step % 100 == 0):
            print("step: " + str(step))
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100]}))
            print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100]}))
        if (step % 1000 == 0):
            acc_val = model.evaluate(X_ho, Y_ho, verbose=0)[1]
            print("Accuracy on hold-out set: " + str(acc_val))
            x_adv_eval = np.zeros((1000,32,32,3))
            for i in np.arange(0,1000,100):
                x_adv_eval[i:(i+100)] = sess.run(x_adv, feed_dict={model.inputs[0]: X_ho[i:(i+100)]})
            acc_adv_eval = model.evaluate(x_adv_eval, Y_ho[0:1000], verbose=0)[1]
            print("Accuracy on hold-out adv examples: " + str(acc_adv_eval))
            print("Accuracy on training batch of adv examples: " + str(model.evaluate(x_batch_adv, y_batch, verbose=0)[1]))       
            model.save_weights("weights/cifar10_weights_best_" + model_type +  "_" + str(lim_freq) + 
                               "_" + str(step) + "_" + str(acc_val) + "_" + str(acc_adv_eval) + ".hdf5")     

    
    
if (model_type == "madry_high"):
    
    from utils_layers_freq import fft_high_pass
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
    
    x_adv = pgd_generate(model.inputs[0], model, eps=0.03, eps_iter=0.008, nb_iter= 10, y=None, ord=np.inf, clip_min=0.0, clip_max=1.0, 
                         y_target=None, rand_init= True, rand_init_eps= 0.03, clip_grad=False, sanity_checks=True)
    
    
    x_highf = fft_high_pass(x_adv, lim_freq, nb_channels=3)
    
    sess = tf.Session()
    K.set_session(sess)
    
    #loss    
    model_logits =  model(x_adv)._op.inputs[0]
    ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    model_logits_high = model(x_highf)._op.inputs[0]
    cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))
    
    tot_loss =  ce_loss + coeff*cons_loss
    
    step_size_schedule = [[0, 0.1], [40000, 0.01], [60000, 0.001]]
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
    
    import time
    a = time.time()
    
    print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))
    for step in np.arange(0, 80000):  
        x_batch, y_batch = next(generator)
        x_batch_adv = sess.run(x_adv, feed_dict={y: y_batch, model.inputs[0]: x_batch})
        if (step <= 10000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1,
                     K.learning_phase(): 1 })
        if (step > 10000) & (step <= 20000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.3,
                     K.learning_phase(): 1 })
        if (step > 20000) & (step <= 40000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5,
                     K.learning_phase(): 1 })
        if (step > 40000) & (step <= 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.7,
                     K.learning_phase(): 1 })    
        if (step > 70000):
            sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0,
                     K.learning_phase(): 1 })
        if (step % 100 == 0):
            print("step: " + str(step))
            print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100]}))
            print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100]}))
        if (step % 1000 == 0):
            acc_val = model.evaluate(X_ho, Y_ho, verbose=0)[1]
            print("Accuracy on hold-out set: " + str(acc_val))
            x_adv_eval = np.zeros((1000,32,32,3))
            for i in np.arange(0,1000,100):
                x_adv_eval[i:(i+100)] = sess.run(x_adv, feed_dict={model.inputs[0]: X_ho[i:(i+100)]})
            acc_adv_eval = model.evaluate(x_adv_eval, Y_ho[0:1000], verbose=0)[1]
            print("Accuracy on hold-out adv examples: " + str(acc_adv_eval))
            print("Accuracy on training batch of adv examples: " + str(model.evaluate(x_batch_adv, y_batch, verbose=0)[1]))       
            model.save_weights("weights/cifar10_weights_best_" + model_type + "_" + str(lim_freq) + 
                               "_" + str(step) + "_" + str(acc_val) + "_" + str(acc_adv_eval) + ".hdf5")   

            
