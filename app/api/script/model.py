from tensorflow.nn import lrn
from tensorflow.keras import losses, Model 
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Concatenate, Lambda, AveragePooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.initializers import HeNormal

def googlenet(nbr_classes, target_size, initializer):

    def inception_layer(x, filters):
        
        a = Conv2D(filters[0], (1, 1), kernel_initializer=initializer, activation='relu')(x)

        b = Conv2D(filters[1], (1, 1), kernel_initializer=initializer, activation='relu')(x)
        b = Conv2D(filters[2], (3, 3), kernel_initializer=initializer, padding='same', activation='relu')(b)

        c = Conv2D(filters[3], (1, 1), kernel_initializer=initializer, activation='relu')(x)
        c = Conv2D(filters[4], (5, 5), kernel_initializer=initializer, padding='same', activation='relu')(c)

        d = MaxPool2D((3, 3), strides=1, padding='same')(x)
        d = Conv2D(filters[5], (1, 1), kernel_initializer=initializer, activation='relu')(d)

        output = Concatenate()([a, b, c, d])

        return output

    input = Input(shape=(target_size[0], target_size[1], 3))

    x = Conv2D(64, (7, 7), strides=2, padding='same', kernel_initializer=initializer, activation='relu')(input)
    x = MaxPool2D((3, 3), strides=2)(x)
    x = Lambda(lrn)(x)

    x = Conv2D(64, (1, 1), padding='same', kernel_initializer=initializer, activation='relu')(input)
    x = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializer, activation='relu')(input)
    # x = Lambda(lrn)(x)
    x = MaxPool2D((3, 3), strides=2)(x)

    x = inception_layer(x, [64, 96, 128, 16, 32, 32])
    x = inception_layer(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D((3, 3), strides=2)(x)

    x = inception_layer(x, [192, 96, 208, 16, 48, 64])
    
    y1 = AveragePooling2D((5, 5), strides=3)(x)
    y1 = Conv2D(128, (1,1), kernel_initializer=initializer, activation='relu')(y1)
    y1 = Flatten()(y1)
    y1 = Dense(1024, kernel_initializer=initializer, activation='relu')(y1)
    y1 = Dropout(0.7, seed=11)(y1)
    y1 = Dense(8, activation='softmax', name='sf-1')(y1) 
    
    x = inception_layer(x, [192, 96, 208, 16, 48, 64])
    x = inception_layer(x, [112, 144, 288, 32, 64, 64])
    x = inception_layer(x, [256, 160, 320, 32, 128, 128])

    y2 = AveragePooling2D((5, 5), strides=3)(x)
    y2 = Conv2D(128, (1,1), kernel_initializer=initializer)(y2)
    y2 = Flatten()(y2)
    y2 = Dense(1024, kernel_initializer=initializer, activation='relu')(y2)
    y2 = Dropout(0.7, seed=11)(y2)
    y2 = Dense(8, activation='softmax', name='sf-2')(y2)

    x = inception_layer(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D((3, 3), strides=2)(x)

    x = inception_layer(x, [256, 160, 320, 32, 128, 128]) 
    x = inception_layer(x, [384, 192, 384, 48, 128, 128])
    x = GlobalAveragePooling2D()(x)

    y3 = Dropout(0.4, seed=11)(x)
    output = Dense(nbr_classes, activation='softmax', name='sf-3')(y3)

    model = Model(inputs=input, outputs=[output, y1, y2])

    return model