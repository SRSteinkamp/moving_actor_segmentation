import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def weighted_BCE(y_true, y_pred):
    '''
    Loss function to weight the mask against background.
    Getting mask correct is twice as important.
    '''
    not_background_pred = y_pred[y_true == 1]
    background_pred = y_pred[y_true == 0]
    not_background_true = y_true[y_true == 1]
    background_true = y_true[y_true == 0]

    BCE_mask = tf.losses.binary_crossentropy(not_background_true, not_background_pred)
    BCE_back = tf.losses.binary_crossentropy(background_true, background_pred)

    return (2 * BCE_mask + BCE_back) / 3


def balanced_accuracy(y_true, y_pred):
    '''
    Calculating balanced accuracy.
    '''
    not_background_pred = y_pred[y_true == 1]
    background_pred = y_pred[y_true == 0]
    not_background_true = y_true[y_true == 1]
    background_true = y_true[y_true == 0]

    BCE_mask = tf.metrics.binary_accuracy(not_background_true, not_background_pred)
    BCE_back = tf.metrics.binary_accuracy(background_true, background_pred)

    return (BCE_mask + BCE_back) / 2


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

    def __init__(self, image_shape, img_channels):
        self.channels = img_channels + 1
        self.image_shape = image_shape

    def make_model(self):

        inp = Input(shape=[*self.image_shape, self.channels])

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
