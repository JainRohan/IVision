import cv2
from imageai.Detection import ObjectDetection
import os

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()
#
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        image_path = detector.detectObjectsFromImage(os.path.join(execution_path, img_name), os.path.join(execution_path, "annotated"))
        print("{} written!".format(img_name))
        img_counter += 1
        print(image_path)
#
cam.release()

cv2.destroyAllWindows()
