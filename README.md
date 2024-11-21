# Robotic-Vision-Pick-an-dPlace-Type-Vision
There two aspects I'd like to solve.  
1.  Identifying objects amongst others in a basket of different goods.  there is no robot involved in picking, it just needs to see the items.

2.  To identify items that a person know picks off a shelf, to track their movements to tell which item is picked.
==============
For the problem you’ve described, we can approach it with computer vision techniques and object detection models. These tasks typically involve image classification, object detection, and motion tracking. Below is an outline of Python code that leverages libraries such as OpenCV, TensorFlow, and YOLO (You Only Look Once) for these tasks.
Python Code for Object Detection and Tracking

We will break the solution into two main sections:

    Object Detection in a Basket of Goods – Detecting and identifying multiple objects within a basket.
    Tracking Items Picked from a Shelf – Identifying and tracking objects picked by a person from a shelf.

Prerequisites:

    OpenCV for image processing and video capture.
    TensorFlow or PyTorch for running object detection models (YOLO, SSD, Faster R-CNN).
    dlib for facial or general tracking if needed for movement tracking.

You can install required packages using:

pip install opencv-python tensorflow numpy dlib

1. Object Detection in a Basket of Goods

Here, we will use a pre-trained object detection model (such as YOLO or MobileNet). This model will detect and classify the objects in the basket.

import cv2
import numpy as np

# Load the pre-trained YOLO model
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO class labels (which YOLO can detect)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function for detecting objects
def detect_objects_in_basket(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to eliminate redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indexes

# Function to draw bounding boxes
def draw_bounding_boxes(frame, boxes, confidences, class_ids, indexes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Example of processing a video or an image
cap = cv2.VideoCapture("basket_goods_video.mp4")  # Replace with video feed or image path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    boxes, confidences, class_ids, indexes = detect_objects_in_basket(frame)
    frame_with_boxes = draw_bounding_boxes(frame, boxes, confidences, class_ids, indexes)

    cv2.imshow("Detected Objects", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Explanation:

    YOLO Model: This script uses YOLOv3 pre-trained weights and config file to detect objects in a basket. You can replace the YOLOv3 weights, config, and coco.names with the correct paths if you download them from official repositories.
    Object Detection: The script processes each frame of a video, detects objects using YOLO, and draws bounding boxes around the detected objects.
    Classes: The object classes detected by YOLO are from the COCO dataset, and each object detected is labeled accordingly.

2. Tracking Items Picked from a Shelf

For this, we can track the movement of a person picking up an item from a shelf using OpenCV's tracking algorithms. We will use a DeepSORT or KCF tracking algorithm to track the picked items across frames.

import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture("shelf_video.mp4")  # Replace with video feed

# Initialize a tracker (e.g., KCF, MOSSE, or CSRT)
tracker = cv2.TrackerCSRT_create()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Manually select the region of interest (ROI) (the item the person picks)
roi = cv2.selectROI("Select Item to Track", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, roi = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the tracked object
    cv2.imshow("Item Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Explanation:

    Tracking: The code uses the CSRT tracker to track the selected object in a video frame. You can manually select the item when prompted with selectROI and the tracker will follow the item’s movement.
    Update: The tracker updates in each frame to follow the object.
    Visualization: The tracked object is displayed in a bounding box on the video feed.

Combining the Two

You can combine the object detection and tracking in a single workflow, where:

    You first detect the object in the basket (using YOLO).
    Once an object is picked from the shelf, the tracker (like CSRT) can track that object.

# Detect and track logic can be combined based on scenario.

Summary of Approach:

    Object Detection: Use YOLO or another pre-trained model to detect objects in a basket.
    Tracking: Use OpenCV tracking methods (CSRT, MOSSE, KCF) to track a specific object picked by a person.
    Video/Camera Feed: Process each frame of a video or real-time camera feed to detect and track the movement.

This approach should be sufficient to identify objects in a basket and track the items picked by a person from a shelf. The models can be customized for specific objects based on your dataset.


