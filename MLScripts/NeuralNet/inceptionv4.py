from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
import numpy as np

def inception_stem(input): # Input (299,299,3)
    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

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
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2,2))(m2)

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

def reduction_A(input, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    return m

def reduction_B(input):
    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Convolution2D(192, 1, 1, activation='relu')(input)
    r2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 3, 3, activation='relu', border_mode='valid', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=1)
    return m


def create_inception_v4(input, nb_output=1000):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(input)

    # 4 x Inception A
    x = inception_A(x)
    x = inception_A(x)
    x = inception_A(x)
    x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    x = inception_C(x)
    x = inception_C(x)
    x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((7,7))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    x = Dense(output_dim=nb_output, activation='softmax')(x)
    return x

def inception_resnet_stem(input, final=256):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', )(c)
    c = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c = Convolution2D(80, 1, 1, activation='relu', border_mode='same')(c)
    c = Convolution2D(192, 3, 3, activation='relu')(c)
    c = Convolution2D(final, 3, 3, activation='relu', subsample=(2,2), border_mode='same')(c)
    return c

def inception_resnet_A(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=1, mode='concat')

    ir_conv = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_B(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(128, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(128, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(896, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_C(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(192, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(1792, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def reduction_resnet_B(input):
    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Convolution2D(256, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=1, mode='concat')
    return m

def create_inception_resnet_v1(input, nb_output=1000, scale=False):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(input)

    # 5 x Inception Resnet A
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)
    x = inception_resnet_A(x, scale_residual=scale)

    # Reduction A
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)
    x = inception_resnet_B(x, scale_residual=scale)

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)
    x = inception_resnet_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((7,7))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    x = Dense(output_dim=nb_output, activation='softmax')(x)
    return x

def inception_resnet_v2_A(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=1, mode='concat')

    ir_conv = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_v2_B(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(160, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(192, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(1152, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def inception_resnet_v2_C(input, scale_residual=False):
    # Input is relu activation
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(224, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=1)

    ir_conv = Convolution2D(2144, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * np.random.uniform(0.1, 0.3))(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = Activation("relu")(out)
    return out

def reduction_resnet_v2_B(input):
    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(288, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Convolution2D(288, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Convolution2D(320, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=1, mode='concat')
    return m

def create_inception_resnet_v2(input, nb_output=1000, scale=False):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(input, final=384)

    # 5 x Inception Resnet A
    x = inception_resnet_v2_A(x, scale_residual=scale)
    x = inception_resnet_v2_A(x, scale_residual=scale)
    x = inception_resnet_v2_A(x, scale_residual=scale)
    x = inception_resnet_v2_A(x, scale_residual=scale)
    x = inception_resnet_v2_A(x, scale_residual=scale)

    # Reduction A
    x = reduction_A(x, k=256, l=256, m=384, n=384)

    # 10 x Inception Resnet B
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)
    x = inception_resnet_v2_B(x, scale_residual=scale)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x)

    # 5 x Inception Resnet C
    x = inception_resnet_v2_C(x, scale_residual=scale)
    x = inception_resnet_v2_C(x, scale_residual=scale)
    x = inception_resnet_v2_C(x, scale_residual=scale)
    x = inception_resnet_v2_C(x, scale_residual=scale)
    x = inception_resnet_v2_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((7,7))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    x = Dense(output_dim=nb_output, activation='softmax')(x)
    return x

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    from keras.utils.visualize_util import plot

    ip = Input(shape=(3,299,299))

    #inception_v4 = create_inception_v4(ip)
    #model = Model(input=ip, output=inception_v4)

    #plot(model, to_file="Inception-v4.png", show_shapes=True)

    inception_resnet_v1 = create_inception_resnet_v1(ip, scale=True)
    model = Model(ip, inception_resnet_v1)

    plot(model, to_file="Inception ResNet-v1.png", show_shapes=True)

    inception_resnet_v2 = create_inception_resnet_v2(ip, scale=True)
    model = Model(ip, inception_resnet_v2)

    plot(model, to_file="Inception ResNet-v2.png", show_shapes=True)


