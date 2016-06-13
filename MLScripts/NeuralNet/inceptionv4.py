from keras.layers import merge
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D

def inception_stem(input): # Input (299,299,3)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', )(c)

    c1 = MaxPooling2D((3,3), strides=(2,2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=1)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=1)

    p1 = MaxPooling2D((3,3), strides=(2,2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu')(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=1)
    return m3


def inception_A(input):
    a1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    a1 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(a1)

    a2 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(input)

    a3 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a3 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a3)

    a4 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=1)
    return m

def inception_B(input):
    b1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    b1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(b1)

    b2 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)

    b3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b3 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b3)
    b3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(b3)

    b4 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b4 = Convolution2D(192, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 7, 1, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(256, 7, 1, activation='relu', border_mode='same')(b4)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=1)
    return m

def inception_C(input):
    c1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(c1)

    c2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)

    c3 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c3)
    c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c3)

    c4 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c4 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(c4)
    c4 = Convolution2D(224, 3, 1, activation='relu', border_mode='same')(c4)
    c4_1 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c4)
    c4_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c4)

    m = merge([c1, c2, c3_1, c3_2, c4_1, c4_2], mode='concat', concat_axis=1)
    return m

def reduction_1(input):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(224, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(256, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    return m

def reduction_2(input):
    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Convolution2D(192, 1, 1, activation='relu')(input)
    r2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 3, 3, activation='relu', border_mode='valid', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    return m

