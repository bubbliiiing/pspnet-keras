from keras.models import *
from keras.layers import *
from nets.mobilenetv2 import get_mobilenet_encoder
from nets.resnet50 import get_resnet50_encoder
import tensorflow as tf

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def resize_image(inp, s, data_format):
	return Lambda(lambda x: tf.image.resize_images(x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1])))(inp)

def pool_block(feats, pool_factor, out_channel):
	h = K.int_shape(feats)[1]
	w = K.int_shape(feats)[2]
	# strides = [30,30],[15,15],[10,10],[5,5]
	# poolsize 30/6=5 30/3=10 30/2=15 30/1=30
	pool_size = strides = [int(np.round(float(h)/pool_factor)),int(np.round(float(w)/pool_factor))]
	# 进行不同程度的平均
	x = AveragePooling2D(pool_size , data_format=IMAGE_ORDERING , strides=strides, padding='same')(feats)
	# 进行卷积
	x = Conv2D(out_channel//4, (1 ,1), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu' )(x)
	x = Lambda(lambda x: tf.image.resize_images(x, (K.int_shape(feats)[1], K.int_shape(feats)[2]), align_corners=True))(x)
	return x

def pspnet(n_classes, inputs_size, downsample_factor=8, backbone='mobilenet', aux_branch=True):
	if backbone == "mobilenet":
		img_input, f4, o = get_mobilenet_encoder(inputs_size, downsample_factor=downsample_factor)
		out_channel = 320
	elif backbone == "resnet50":
		img_input, f4, o = get_resnet50_encoder(inputs_size, downsample_factor=downsample_factor)
		out_channel = 2048
	else:
		raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))
	#-------------------------------------#
	#	PSP模块
	#	分区域进行池化
	#-------------------------------------#
	pool_factors = [1,2,3,6]
	pool_outs = [o]

	for p in pool_factors:
		pooled = pool_block(o, p, out_channel)
		pool_outs.append(pooled)
	
	# 连接
	# 60x60xout_channel*2
	o = Concatenate(axis=MERGE_AXIS)(pool_outs)

	#-------------------------------------#
	#	利用特征获得预测结果
	#-------------------------------------#
	# 卷积
	# 60x60xout_channel//4
	o = Conv2D(out_channel//4, (3,3), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)
	# 正则化，防止过拟合
	o = Dropout(0.1)(o)

	# 60x60x21
	o = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(o)
	# [473,473,nclasses]
	o = Lambda(lambda x: tf.image.resize_images(x, (inputs_size[1], inputs_size[0]), align_corners=True))(o)
	# 获得每一个像素点属于每一个类的概率了
	o = Activation("softmax", name="main")(o)
	
	if aux_branch:
		f4 = Conv2D(out_channel//8, (3,3), data_format=IMAGE_ORDERING, padding='same', use_bias=False)(f4)
		f4 = BatchNormalization()(f4)
		f4 = Activation('relu')(f4)
		# 防止过拟合
		f4 = Dropout(0.1)(f4)

		# 60x60x21
		f4 = Conv2D(n_classes,(1,1),data_format=IMAGE_ORDERING, padding='same')(f4)
		# [473,473,nclasses]
		f4 = Lambda(lambda x: tf.image.resize_images(x, (inputs_size[1], inputs_size[0]), align_corners=True))(f4)
		# 获得每一个像素点属于每一个类的概率了
		f4 = Activation("softmax", name="aux")(f4)
		model = Model(img_input,[f4,o])
		return model
	else:
		model = Model(img_input,[o])
		return model

	
