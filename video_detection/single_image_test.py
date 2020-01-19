##  Written by Connor Glossop

import cv2
from imageai.Detection import ObjectDetection
import os


def cleanup(path):
    for file in os.listdir(path):
        if file.endswith(".png"):
            os.remove(file)
    print("Directory Cleared")


cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break

    img_name = "frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))
    image_path = detector.detectObjectsFromImage(input_image=img_name, output_image_path=img_name)
    print(image_path)  # gives probabilities

    img_counter += 1

    if cv2.waitKey(1) % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()

cleanup(os.getcwd())
