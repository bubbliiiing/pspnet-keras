#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.pspnet import pspnet

if __name__ == "__main__":
    model = pspnet(21,[473,473,3],downsample_factor=16,backbone='mobilenet',aux_branch=False)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
