#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.pspnet import pspnet

if __name__ == "__main__":
    model = pspnet([473, 473, 3], 21, backbone='mobilenet', downsample_factor=16, aux_branch=False)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
