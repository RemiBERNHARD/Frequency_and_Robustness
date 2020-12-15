from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from utils_layers_freq import fft_low_pass, fft_high_pass


def svhn_model(input_tensor = "no", type_freq = None, low_freq = 0, high_freq = 0, softmax="yes"):
    if (input_tensor == "no"):
        inputs = Input(shape=(32,32,3))
        l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(inputs)
    if (input_tensor != "no") & (type_freq == None):        
        inputs = input_tensor
        l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(inputs)
    if (input_tensor != "no") & (type_freq != None):
        inputs = input_tensor
        if (type_freq == "low"):
            j = Lambda(fft_low_pass, output_shape=(32,32,3), arguments={'lim_freq': low_freq})(inputs)
        if (type_freq == "high"):
            j = Lambda(fft_high_pass, output_shape=(32,32,3), arguments={'lim_freq': low_freq})(inputs)      
        l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(j)

    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False, kernel_initializer="random_uniform")(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False, kernel_initializer="random_uniform")(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l) 
    l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False, kernel_initializer="random_uniform")(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(512, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False, kernel_initializer="random_uniform")(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Conv2D(512, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False, kernel_initializer="random_uniform")(l)
    l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Flatten()(l)
    l = Dense(1024, use_bias=False, kernel_initializer="random_uniform")(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    l = Dropout(0.25)(l)
    l = Dense(1024, use_bias=False, kernel_initializer="random_uniform")(l)
    l = BatchNormalization()(l)
    l = Activation("relu")(l)
    
    if (softmax=="yes"):
        l = Dense(10, activation="softmax", use_bias=False, kernel_initializer="random_uniform")(l)
    else :
        l = Dense(10, use_bias=False, kernel_initializer="random_uniform")(l)
        
    predictions = l
    
    model = Model(inputs=inputs, outputs=predictions)   
    print("Model for SVHN created")
    
    return(model)
    



