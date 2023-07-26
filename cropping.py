from ultralytics import YOLO
import cv2
from yolomasking import get_masks
from ultralytics.utils.ops import scale_image
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

model = YOLO('yolov8n-seg.pt')  # load an official model
img_url = 'https://ultralytics.com/images/bus.jpg'
img = cv2.imread("C:/Users/adipi/Documents/Code/Internship-RAKRIC-0723/trash-classification/bus.jpg")
imgshape = img.shape
print("imgshape: ", imgshape)

masks = get_masks(model, img_url)
masks = masks.data
print("size: ", len(masks))

# what object is mask?
print("type: ", type(masks))
print("type: ", type(masks[0]))

for i in range(len(masks.data)): # type: ignore
        print("mask shape: ", masks.data[i].shape)

testmask = masks[3]
transform = T.ToPILImage()
testmask = transform(testmask)
testmask = testmask.resize((1080, 810))
#testmask = (masks[3].numpy() * 255).astype("uint8")

testmask.save('image_resize.jpg')

testmask = np.array(testmask)

#cv2.imshow("mask", testmask)
#cv2.waitKey(0)
#
#cv2.imshow("image", img)
#cv2.waitKey(0)
#

img = np.array(img * 255).astype("uint8")
testmask = np.array(testmask * 255).astype("uint8")

result = cv2.bitwise_and(img, img, mask= testmask)
cv2.imshow('Masked Image',result)
cv2.waitKey(0)