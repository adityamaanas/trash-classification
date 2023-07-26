from ultralytics import YOLO
import cv2
from ROI import get_box
from urllib.request import urlopen
from fastai.vision.all import * # type: ignore

model = YOLO('yolov8n-seg.pt')

def crop(img_url, img):
    cropped = []
    boxes = get_box(model, img_url)
    boxes = boxes.xyxy
    #img = PILImage.create(urlopen(img_url))
    #img = np.array(img)

    # create separate images for each box
    for box in boxes:
        # get the coordinates of the box
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # crop the image
        crop_img = img[y1:y2, x1:x2]
        cropped.append(crop_img)

        # show the cropped image
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)

    return cropped # type: ignore