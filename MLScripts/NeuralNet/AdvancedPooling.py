from keras.layers.convolutional import _Pooling1D, _Pooling2D, _Pooling3D
import keras.backend as K

class MaxAveragePool1D(_Pooling1D):

    def __init__(self, pool_length=2, stride=None,
                 border_mode='valid', **kwargs):
        super(MaxAveragePool1D, self).__init__(pool_length, stride,
                                           border_mode, **kwargs)

    def _pooling_function(self, back_end, inputs, pool_size, strides,
                          border_mode, dim_ordering):

        output1 = K.pool2d(inputs, pool_size, strides,
                  border_mode, dim_ordering, pool_mode='max')
        output2 = K.pool2d(inputs, pool_size, strides,
                           border_mode, dim_ordering, pool_mode='ave')
        return output1 + output2


class MaxAveragePool2D(_Pooling2D):

    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering=K.image_dim_ordering(), **kwargs):
        super(MaxAveragePool2D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                              border_mode, dim_ordering):
        output1 = K.pool2d(inputs, pool_size, strides,
                  border_mode, dim_ordering, pool_mode='max')
        output2 = K.pool2d(inputs, pool_size, strides,
                     border_mode, dim_ordering, pool_mode='avg')
        return output1 + output2
