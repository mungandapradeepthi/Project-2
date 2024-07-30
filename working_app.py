import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
import cv2
import numpy as np

cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
body_cascade = cv2.CascadeClassifier('fullbody.xml')

class VideoSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Video Selector')

        self.btnSelect = QPushButton('Select Video', self)
        self.btnSelect.setGeometry(10, 10, 150, 30)
        self.btnSelect.clicked.connect(self.showDialog)

        self.videoPath = ""

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './', "Video files (*.mp4 *.avi)")
        if fname[0]:
            self.videoPath = fname[0]
            self.close()
            self.runSimulator()

    def runSimulator(self):
        if self.videoPath:
            Simulator(self.videoPath)

def detect_cars_and_pedestrian(frame):
    # YOLO Configuration
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f]

    layer_names = net.getUnconnectedOutLayersNames()

    # YOLO Detection
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # 2 corresponds to the 'car' class in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for cars
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)

    # Haarcascade car detection
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), color=(255, 0, 0), thickness=2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)

    # Haarcascade pedestrian detection
    pedestrians = body_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2)

    # Count the number of bounding boxes
    total_boxes = len(indices) + len(cars) + len(pedestrians)

    return frame, total_boxes

def Simulator(video_path):
    CarVideo = cv2.VideoCapture(video_path)
    box_count = 0
    while CarVideo.isOpened():
        ret, frame = CarVideo.read()
        controlkey = cv2.waitKey(1)
        if ret:
            cars_frame, current_boxes = detect_cars_and_pedestrian(frame)
            box_count += current_boxes
            cv2.putText(cars_frame, f'vehicle count: {box_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('VIEW', cars_frame)
        else:
            break
        if controlkey == ord('q'):
            break

    CarVideo.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    selector = VideoSelector()
    selector.show()

    sys.exit(app.exec_())
