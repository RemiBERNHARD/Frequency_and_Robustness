import sys
from keras.utils import np_utils
from utils_func import imnet_load_data
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
import numpy as np
from model_imnet import MobileNetV2
from sklearn.model_selection import train_test_split


config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth
sess = tf.Session(config=config) 
K.set_session(sess)


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

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=123)

Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)


#######################################
#Perform training
#######################################
model_type = sys.argv[1]
print("Training model: " + model_type)


if (model_type == "freq_const_lowf"):

    from utils_layers_freq_old import fft_low_pass
    lim_freq = int(sys.argv[2])
    print("Freq limit: " + str(lim_freq))    
    
    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)

    with tf.device('/gpu:0'):
    
        model = MobileNetV2((224,224,3))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
                
        batch_size = int(sys.argv[3])
        generator = generator.flow(X_train, Y_train, batch_size=batch_size)
        
        coeff = tf.placeholder(tf.float32, shape=()) 
        y = tf.placeholder(tf.float32, shape=(None, 10))
        
        x_lowf = fft_low_pass(model.inputs[0], lim_freq, nb_channels=3)
        
        #Loss    
        model_logits = model(model.inputs[0])._op.inputs[0]
        ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    with tf.device('/gpu:1'):    
    
        model_logits_low = model(x_lowf)._op.inputs[0]
        cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
        tot_loss =  ce_loss + coeff*cons_loss

        step_size_schedule = [[0, 0.1], [25000, 0.02], [38000, 0.004]]
        global_step = tf.train.get_or_create_global_step()
        boundaries = [int(sss[0]) for sss in step_size_schedule]
        boundaries = boundaries[1:]
        values = [sss[1] for sss in step_size_schedule]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values) 
        
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
        opt_op = optimizer.minimize(tot_loss, global_step=global_step, colocate_gradients_with_ops=True)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))            
        acc_tamp = model.evaluate(X_val, Y_val, verbose=0)[1]
        
        for step in np.arange(0, 45000):  
            x_batch, y_batch = next(generator)
            if (step <= 5000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.05, 
                         K.learning_phase(): 1 })
            if (step > 5000) & (step <= 15000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1, 
                         K.learning_phase(): 1 })
            if (step > 15000) & (step <= 30000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5, 
                         K.learning_phase(): 1 })
            if (step > 30000) & (step <= 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0, 
                         K.learning_phase(): 1 })    
            if (step > 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 2.0, 
                         K.learning_phase(): 1 })
            if (step % 100 == 0):
                print(step)
                print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
                print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
                print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
            if (step % 1000 == 0):
                acc_val = model.evaluate(X_val, Y_val, verbose=0)[1]
                print("Accuracy on validation set: ", acc_val)
                loss_val = sess.run(cons_loss, feed_dict={model.inputs[0]: X_val, y: Y_val, coeff:1.})
                model.save_weights("weights/imnet_weights_best_" + model_type + "_" + str(lim_freq) + "_" + str(step) +
                                   "_" + str(loss_val) + "_" + str(acc_val) +".hdf5")

        model.load_weights("weights/imnet_weights_best_" + model_type  + "_" + str(lim_freq) + "_" + str(step) +
                           "_" + str(loss_val) + "_" + str(acc_val) +".hdf5")
        
        print("Accuracy on train set: " + str(model.evaluate(X_train, Y_train, verbose=0)[1]))
        print("Accuracy on test set: " +  str(model.evaluate(X_test, Y_test, verbose=0)[1]))
        
        model.save("models/IMNET_" + str(model_type) + "_" + str(lim_freq) + ".h5")
        


if (model_type == "freq_const_highf"):

    from utils_layers_freq_old import fft_high_pass
    lim_freq = int(sys.argv[2])
    print("Freq limit: " + str(lim_freq))    
    
    generator=ImageDataGenerator(rotation_range=10,
                                 width_shift_range=5./32,
                                 height_shift_range=5./32,
                                 horizontal_flip=True)

    with tf.device('/gpu:0'):
    
        model = MobileNetV2((224,224,3))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
                
        batch_size = int(sys.argv[3])
        generator = generator.flow(X_train, Y_train, batch_size=batch_size)
        
        coeff = tf.placeholder(tf.float32, shape=()) 
        y = tf.placeholder(tf.float32, shape=(None, 10))
        
        x_highf = fft_high_pass(model.inputs[0], lim_freq, nb_channels=3)
        
        #Loss    
        model_logits = model(model.inputs[0])._op.inputs[0]
        ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    with tf.device('/gpu:1'):    
    
        model_logits_high = model(x_highf)._op.inputs[0]
        cons_loss = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))
    
        tot_loss =  ce_loss + coeff*cons_loss

        step_size_schedule = [[0, 0.1], [25000, 0.02], [38000, 0.004]]
        global_step = tf.train.get_or_create_global_step()
        boundaries = [int(sss[0]) for sss in step_size_schedule]
        boundaries = boundaries[1:]
        values = [sss[1] for sss in step_size_schedule]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values) 
        
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
        opt_op = optimizer.minimize(tot_loss, global_step=global_step, colocate_gradients_with_ops=True)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff: 1.}))            
        acc_tamp = model.evaluate(X_val, Y_val, verbose=0)[1]
        
        for step in np.arange(0, 45000):  
            x_batch, y_batch = next(generator)
            if (step <= 5000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.05, 
                         K.learning_phase(): 1 })
            if (step > 5000) & (step <= 15000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.1, 
                         K.learning_phase(): 1 })
            if (step > 15000) & (step <= 30000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 0.5, 
                         K.learning_phase(): 1 })
            if (step > 30000) & (step <= 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 1.0, 
                         K.learning_phase(): 1 })    
            if (step > 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff: 2.0, 
                         K.learning_phase(): 1 })
            if (step % 100 == 0):
                print(step)
                print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
                print(sess.run(cons_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
                print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff:1.}))
            if (step % 1000 == 0):
                acc_val = model.evaluate(X_val, Y_val, verbose=0)[1]
                print("Accuracy on validation set: ", acc_val)
                loss_val = sess.run(cons_loss, feed_dict={model.inputs[0]: X_val, y: Y_val, coeff:1.})
                model.save_weights("weights/imnet_weights_best_" + model_type + "_" + str(lim_freq) + "_" + str(step) +
                                   "_" + str(loss_val) + "_" + str(acc_val) +".hdf5")



if (model_type == "freq_const_all"):
    
    generator=ImageDataGenerator(rotation_range=10,
                             width_shift_range=5./32,
                             height_shift_range=5./32,
                             horizontal_flip=True)
    
    from utils_layers_freq_old import fft_low_pass, fft_high_pass
    lim_freq_low = int(sys.argv[2])
    lim_freq_high = int(sys.argv[3])
    print("Freq limit low: " + str(lim_freq_low))    
    print("Freq limit high: " + str(lim_freq_high))    
    
    
    with tf.device('/gpu:0'):
    
        model = MobileNetV2((224,224,3))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
                
        batch_size = int(sys.argv[4])
        generator = generator.flow(X_train, Y_train, batch_size=batch_size)
        
        coeff_low = tf.placeholder(tf.float32, shape=())
        coeff_high = tf.placeholder(tf.float32, shape=())
        
        y = tf.placeholder(tf.float32, shape=(None, 10))
        
        x_lowf = fft_low_pass(model.inputs[0], lim_freq_low, nb_channels=3)
        x_highf = fft_high_pass(model.inputs[0], lim_freq_high, nb_channels=3)
        
        #Loss    
        model_logits = model(model.inputs[0])._op.inputs[0]
        ce_loss = tf.losses.softmax_cross_entropy(y, model_logits)
    
    with tf.device('/gpu:1'):    
    
        model_logits_low = model(x_lowf)._op.inputs[0]
        cons_loss_1 = tf.reduce_mean(tf.norm(model_logits - model_logits_low, ord=2, axis=1))
    
        model_logits_high = model(x_highf)._op.inputs[0]
        cons_loss_2 = tf.reduce_mean(tf.norm(model_logits - model_logits_high, ord=2, axis=1))
        
        tot_loss =  ce_loss + coeff_low*cons_loss_1 + coeff_high*cons_loss_2   

        step_size_schedule = [[0, 0.1], [25000, 0.02], [38000, 0.004]]
        global_step = tf.train.get_or_create_global_step()
        boundaries = [int(sss[0]) for sss in step_size_schedule]
        boundaries = boundaries[1:]
        values = [sss[1] for sss in step_size_schedule]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32),
            boundaries,
            values) 
        
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)  
        opt_op = optimizer.minimize(tot_loss, global_step=global_step, colocate_gradients_with_ops=True)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
            
        print("Initial loss value: ", sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high: 1}))            
        acc_tamp = model.evaluate(X_val, Y_val, verbose=0)[1]
        
        for step in np.arange(0, 45000):  
            x_batch, y_batch = next(generator)
            if (step <= 5000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.1, coeff_high: 0.01,
                         K.learning_phase(): 1 })
            if (step > 5000) & (step <= 15000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 0.5, coeff_high: 0.05,
                         K.learning_phase(): 1 })
            if (step > 15000) & (step <= 30000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 1.0, coeff_high: 0.5,
                         K.learning_phase(): 1 })
            if (step > 30000) & (step <= 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 1.75, coeff_high: 1.0,
                         K.learning_phase(): 1 })    
            if (step > 40000):
                sess.run([opt_op, model.updates], feed_dict={y: y_batch, model.inputs[0]: x_batch, coeff_low: 2.5, coeff_high: 2.0,
                         K.learning_phase(): 1 })
            if (step % 100 == 0):
                print(step)
                print(sess.run(ce_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
                print(sess.run(cons_loss_1, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
                print(sess.run(cons_loss_2, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
                print(sess.run(tot_loss, feed_dict={model.inputs[0]: X_train[0:100], y: Y_train[0:100], coeff_low: 1., coeff_high:1.}))
            if (step % 1000 == 0):
                acc_val = model.evaluate(X_test, Y_test, verbose=0)[1]
                print("Accuracy on validation set: ", acc_val)
                loss_val_1 = sess.run(cons_loss_1, feed_dict={model.inputs[0]: X_val, y: Y_val, coeff_low: 1., coeff_high:1.})
                loss_val_2 = sess.run(cons_loss_2, feed_dict={model.inputs[0]: X_val, y: Y_val, coeff_low: 1., coeff_high:1.})
                model.save_weights("weights/imnet_weights_best_" + model_type + "_" + str(lim_freq_low) + "_" + str(lim_freq_high) + 
                                   "_" + str(step) + "_" + str(loss_val_1) + "_" + str(loss_val_2) + "_" + str(acc_val) +".hdf5")

                
