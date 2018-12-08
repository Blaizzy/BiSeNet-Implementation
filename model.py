import keras
import tensorflow as tf
from keras.applications.xception import Xception,preprocess_input
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Dropout, multiply, Dot, Concatenate,Add, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, AveragePooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard


from keras.layers.core import Lambda
from keras.backend import tf as ktf

#-----------------

def conv_bn_act(inputs, n_filters=64, kernel=(2, 2), strides=1, activation='relu'):

    conv = Conv2D(n_filters, kernel_size= kernel, strides = strides, data_format='channels_last')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)

    return conv


def conv_act(inputs, n_filters, kernel = (1,1), activation = 'relu', pooling = False):
    if pooling:
        conv = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
        conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(conv)
        conv = Activation(activation)(conv)
    else:
        conv = Conv2D(n_filters, kernel_size= kernel, strides=1)(inputs)
        conv = Activation(activation)(conv)


    return conv


def CP(layer_13, layer_14):


    # Combine the up-sampled output feature of Global avg pooling and Xception features
    tail_avg = GlobalAveragePooling2D()(layer_14)
    tail_upS = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')(layer_14)
    tail = Add()([tail_avg, tail_upS])

    layer_13 = ARM(layer_13, 1024)
    layer_14 = ARM(layer_14, 2048)

    layer_13 = UpSampling2D(size=2, data_format='channels_last', interpolation='nearest')(layer_13)
    layer_14 = UpSampling2D(size=2, data_format='channels_last', interpolation='nearest')(layer_14)

    context_features = Concatenate(axis=-1)([layer_14, layer_13])
    context_features = Concatenate(axis=-1)([context_features, tail])

    context_features = UpSampling2D(size=2, data_format='channels_last', interpolation='nearest')(context_features)

    return context_features
# ARM (Attention Refinement Module)
# Refines features at each stage of the Context path

def ARM(inputs, n_filters):
    arm = AveragePooling2D(pool_size=(1, 1), padding='same', data_format='channels_last')(inputs)
    arm = conv_bn_act(arm, n_filters, (1, 1), activation='sigmoid')
    arm = multiply([inputs, arm])

    return arm


# FFM (Feature Fusion Module)
# Fuses the low level features of Spatial Path and high level
# features of Context Path

def FFM(input_sp, input_cp, n_classes):
    ffm = Concatenate(axis=-1)([input_sp, input_cp])
    conv = conv_bn_act(ffm, n_classes, (3, 3), strides= 2)

    conv_1 = conv_act(conv, n_classes, (1,1), pooling= True)
    conv_1 = conv_act(conv_1, n_classes, (1,1), activation='sigmoid')

    ffm = multiply([conv, conv_1])
    ffm = Add()([conv, ffm])

    return ffm



#Model
inputs = Input(shape=(224,224,3))
x = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(inputs)
x = Lambda(lambda image: preprocess_input(image))(x)

# Spatial Path
SP = conv_bn_act(inputs, 32, strides=2)
SP = conv_bn_act(SP, 64, strides=2)
SP = conv_bn_act(SP, 156, strides=2)


Xception_model = Xception(weights='imagenet',input_shape= (224,224,3), include_top=False)

# Context_path
layer_13 = Xception_model.get_layer('block13_pool').output
layer_14 = Xception_model.output

CP_features = CP(layer_13, layer_14)

FFM = FFM(SP, CP_features, 32)

output = UpSampling2D(size=(16,16), data_format='channels_last', interpolation='nearest')(FFM)


bisnet = Model(inputs = [inputs, Xception_model.input], output = [output, layer_13, layer_14])

print(bisnet.summary())

from keras.utils import plot_model
plot_model(bisnet, to_file='model.png')