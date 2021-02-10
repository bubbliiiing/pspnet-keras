import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from pspnet import Pspnet


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

class miou_Pspnet(Pspnet):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            img, nw, nh = letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        else:
            img = image.convert('RGB')
            img = img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        img = np.asarray([np.array(img)/255])
        
        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        if self.letterbox_image:
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)
        return image

pspnet = miou_Pspnet()

image_ids = open("VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt",'r').read().splitlines()

if not os.path.exists("miou_pr_dir"):
    os.makedirs("miou_pr_dir")

for image_id in tqdm(image_ids):
    image_path = "VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    image = pspnet.detect_image(image)
    image.save("miou_pr_dir/" + image_id + ".png")
