from nets.pspnet import pspnet

model = pspnet(21,[473,473,3],downsample_factor=16,backbone='resnet50',aux_branch=False)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)
