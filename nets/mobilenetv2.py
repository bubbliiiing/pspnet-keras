from keras import layers
from keras.activations import relu
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, Dropout,
                          GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D)
from keras.models import Model


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def relu6(x):
    return relu(x, max_value=6)

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1].value
    pointwise_filters = _make_divisible(int(filters * alpha), 8)
    prefix = 'expanded_conv_{}_'.format(block_id)

    x = inputs

    #----------------------------------------------------#
    #   利用1x1卷积根据输入进来的通道数进行通道数上升
    #----------------------------------------------------#
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    #----------------------------------------------------#
    #   利用深度可分离卷积进行特征提取
    #----------------------------------------------------#
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name=prefix + 'depthwise_BN')(x)
    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    #----------------------------------------------------#
    #   利用1x1的卷积进行通道数的下降
    #----------------------------------------------------#
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    #----------------------------------------------------#
    #   添加残差边
    #----------------------------------------------------#
    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def get_mobilenet_encoder(inputs_size, downsample_factor=8):
    if downsample_factor == 16:
        block4_dilation = 1
        block5_dilation = 2
        block4_stride = 2
    elif downsample_factor == 8:
        block4_dilation = 2
        block5_dilation = 4
        block4_stride = 1
    else:
        raise ValueError('Unsupported factor - `{}`, Use 8 or 16.'.format(downsample_factor))
    
    # 473,473,3
    inputs = Input(shape=inputs_size)

    alpha=1.0
    first_block_filters = _make_divisible(32 * alpha, 8)

    # 473,473,3 -> 237,237,32
    x = Conv2D(first_block_filters,
                kernel_size=3,
                strides=(2, 2), padding='same',
                use_bias=False, name='Conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    # 237,237,32 -> 237,237,16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    #---------------------------------------------------------------#
    # 237,237,16 -> 119,119,24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)
                            
    #---------------------------------------------------------------#
    # 119,119,24 -> 60,60.32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    #---------------------------------------------------------------#
    # 60,60,32 -> 30,30.64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=block4_stride,
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=9, skip_connection=True)

    # 30,30.64 -> 30,30.96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=12, skip_connection=True)
    # 辅助分支训练
    f4 = x

    #---------------------------------------------------------------#
    # 30,30.96 -> 30,30,160 -> 30,30,320
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=block4_dilation,
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=block5_dilation,
                            expansion=6, block_id=16, skip_connection=False)
    f5 = x
    return inputs, f4, f5
