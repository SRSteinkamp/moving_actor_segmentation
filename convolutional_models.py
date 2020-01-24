import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model

def forward_block(inp, no_filter, kernel_size=(3,3), activation='relu', drp=0.25):
    skip = Conv2D(no_filter, kernel_size, padding='same')(inp)
    x = Activation(activation)(skip)
    x = Dropout(drp)(x)
    x = MaxPooling2D(strides=2)(x)

    return x, skip

def backward_block(inp, skip, no_filter, kernel_size=(3,3), activation='relu', drp=0.25):
    x = Conv2DTranspose(no_filter, kernel_size, strides=(2,2), padding='same')(inp)
    x = concatenate([x, skip])
    x = Conv2D(no_filter, kernel_size, padding='same')(x)
    x = Activation(activation)(x)
    x = Dropout(drp)(x)

    return x

class BasicCNN:

    def __init__(self, input_shape, img_channels):
        self.channels = img_channels + 1
        self.image_shape = image_shape

    def make_model(self):

        inp = Input(shape=[*self.image_shape, img_channels])

        x, c1 = forward_block(inp, 8) # 112
        x, c2 = forward_block(x, 16) # 56
        x, c3 = forward_block(x, 32) # 28
        x, c4 = forward_block(x, 64) # 14
        x, c5 = forward_block(x, 128) # 7

        x = Conv2D(128, (3,3), padding='same')(x)
        x = Activation('relu')(x)

        x = backward_block(x, c5, 128) # 7
        x = backward_block(x, c4, 64) # 14
        x = backward_block(x, c3, 32) # 28
        x = backward_block(x, c2, 16) # 56
        x = backward_block(x, c1, 8) # 112

        x = Conv2D(1, (1,1), padding='same')(x)
        out = Activation('sigmoid')(x)

        model = Model(inputs=inp, outputs=out)

        return model
