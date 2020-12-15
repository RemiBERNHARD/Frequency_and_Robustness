from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout


def svhn_model(input_tensor = "no", type_freq = None, low_freq = 0, high_freq = 0, softmax="yes"):

    inputs = input_tensor
    l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(inputs)    
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
    



