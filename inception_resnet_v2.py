from tensorflow.nn import lrn
from tensorflow.keras import backend
from tensorflow.keras import losses, Model 
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Concatenate, Lambda, AveragePooling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation, Add
from tensorflow.keras.initializers import HeNormal, HeUniform

def inception_resnet_v2(initializer_A, initializer_B, nbr_classes):

    def Conv2DBN(input, filters, kernel_size, strides, initializer, padding, activation):
        
        x = Conv2D(filters, kernel_size, strides=strides, kernel_initializer=initializer_A, padding=padding)(input)
        x = BatchNormalization(axis=3, scale=False)(x)

        if activation:
            x = Activation('relu')(x)
        
        return x

    def stem(x):
        
        x = Conv2DBN(x, 32, (3, 3), strides=2, initializer=initializer_A, padding='valid', activation=True)
        x = Conv2DBN(x, 32, (3, 3), strides=1, initializer=initializer_A, padding='valid', activation=True)
        x = Conv2DBN(x, 64, (3, 3), strides=1, initializer=initializer_A, padding='valid', activation=True)

        x_11 = MaxPool2D((3, 3), strides=2)(x)
        x_12 = Conv2DBN(x, 96, (3, 3), strides=2, initializer=initializer_A, padding='valid', activation=True)
        
        x = Concatenate()([x_11, x_12])

        x_21 = Conv2DBN(x, 64, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x_21 = Conv2DBN(x_21, 96, (3, 3), strides=1, padding='valid', initializer=initializer_A, activation=True)
        
        x_22 = Conv2DBN(x, 64, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x_22 = Conv2DBN(x_22, 64, (7, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x_22 = Conv2DBN(x_22, 64, (1, 7), strides=1, padding='same', initializer=initializer_A, activation=True)
        x_22 = Conv2DBN(x_22, 96, (3, 3), strides=1, padding='valid', initializer=initializer_A, activation=True)

        x = Concatenate()([x_21, x_22])

        x_31 = Conv2DBN(x, 192, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)
        x_32 = MaxPool2D((3, 3), strides=2)(x)

        output = Concatenate()([x_31, x_32])
        
        return output
    
    def inception_resnet_A(x):

        x1 = Conv2DBN(x, 32, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
  
        x2 = Conv2DBN(x, 32, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 32, (3, 3), strides=1, padding='same', initializer=initializer_A, activation=True)
        
        x3 = Conv2DBN(x, 32, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x3 = Conv2DBN(x3, 48, (3, 3), strides=1, padding='same', initializer=initializer_A, activation=True)
        x3 = Conv2DBN(x3, 64, (3, 3), strides=1, padding='same', initializer=initializer_A, activation=True)

        x4 = Concatenate()([x1, x2, x3])
        x4 = Conv2DBN(x4, backend.int_shape(x)[-1], (1, 1), strides=1, padding='same', initializer=initializer_A, activation=False)

        x5 = Add()([x, x4 * 0.1])

        output = Activation('relu')(x5)

        return output

    def inception_resnet_B(x):

        x1 = Conv2DBN(x, 192, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
  
        x2 = Conv2DBN(x, 128, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 160, (1, 7), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 192, (7, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        
        x3 = Concatenate()([x1, x2])
        x3 = Conv2DBN(x3, backend.int_shape(x)[-1], (1, 1), strides=1, padding='same', initializer=initializer_A, activation=False)

        x4 = Add()([x, x3 * 0.1])

        output = Activation('relu')(x4)

        return output

    def inception_resnet_C(x):

        x1 = Conv2DBN(x, 192, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
  
        x2 = Conv2DBN(x, 192, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 224, (1, 3), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 256, (3, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        
        x3 = Concatenate()([x1, x2])
        x3 = Conv2DBN(x3, 2144, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=False)

        x4 = Lambda(lambda inputs: inputs[0] + inputs[1] * 0.1,
            output_shape=backend.int_shape(x)[1:])([x, x3])

        output = Activation('relu')(x4)

        return output
    
    def reduction_A(x):

        x1 = MaxPool2D((3, 3), strides=2)(x)
  
        x2 = Conv2DBN(x, 384, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)

        x3 = Conv2DBN(x, 256, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x3 = Conv2DBN(x3, 256, (3, 3), strides=1, padding='same', initializer=initializer_A, activation=True)
        x3 = Conv2DBN(x3, 384, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)

        output = Concatenate()([x1, x2, x3])

        return output

    def reduction_B(x):

        x1 = MaxPool2D((3, 3), strides=2, padding='valid')(x)
  
        x2 = Conv2DBN(x, 256, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x2 = Conv2DBN(x2, 384, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)

        x3 = Conv2DBN(x, 256, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x3 = Conv2DBN(x3, 288, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)
        
        x4 = Conv2DBN(x, 256, (1, 1), strides=1, padding='same', initializer=initializer_A, activation=True)
        x4 = Conv2DBN(x4, 288, (3, 3), strides=1, padding='same', initializer=initializer_A, activation=True)
        x4 = Conv2DBN(x4, 320, (3, 3), strides=2, padding='valid', initializer=initializer_A, activation=True)

        output = Concatenate(axis=3)([x1, x2, x3, x4])

        return output
    
    input = Input(shape=(299, 299, 3))

    x = stem(input)
    
    for i in range(5):
        x = inception_resnet_A(x)

    x = reduction_A(x)

    for i in range(10):
        x = inception_resnet_B(x)
    
    x = reduction_B(x)

    for i in range(5):
        x = inception_resnet_C(x)
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2, seed=11)(x)

    output = Dense(units=nbr_classes, kernel_initializer=initializer_B, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)

    return model

if __name__ == '__main__':

    model = inception_resnet_v2('he_uniform', 'glorot_uniform', 8)

    summary = list()
    model.summary(print_fn=lambda x: summary.append(x))
    summary = "\n".join(summary)

    out = open('summary.txt', 'w')
    out.write(summary)
    out.close()
