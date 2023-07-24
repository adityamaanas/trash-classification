from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

img_url = 'https://ultralytics.com/images/bus.jpg'

# Define a function to return the coordinates of the segments
def get_masks(model, img_url):
    results = model.predict(img_url, save=True)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        
    print(type(masks))

    return masks

# region test
#img_url = 'https://ultralytics.com/images/bus.jpg'
#img = cv2.imread(img_url)

# Predict with the model
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
#
#for result in results:                                         # iterate results
#    boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
#    for box in boxes:                                          # iterate boxes
#        r = box.xyxy[0].astype(int)                            # get corner points as int
#        print(r)                                               # print boxes
#        cv2.rectangle(img, r[:2], r[2:], (255, 255, 255), 2)   # draw boxes on img
#
#res = model(img)
#res_plotted = res[0].plot()
#cv2.imshow("result", res_plotted)
#cv2.waitKey(0)

# endregion test