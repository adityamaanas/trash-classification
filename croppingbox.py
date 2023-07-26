from ultralytics import YOLO
import cv2
from ROI import get_box

model = YOLO('yolov8n-seg.pt')  # load an official model
img_url = 'https://ultralytics.com/images/bus.jpg'
img = cv2.imread("C:/Users/adipi/Documents/Code/Internship-RAKRIC-0723/trash-classification/bus.jpg")

boxes = get_box(model, img_url)
boxes = boxes.xyxy

# create separate images for each box
for box in boxes:
    # get the coordinates of the box
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    # crop the image
    crop_img = img[y1:y2, x1:x2]

    # show the cropped image
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)