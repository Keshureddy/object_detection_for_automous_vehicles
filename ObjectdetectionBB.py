import cv2
from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolov8n.pt")

camera_height = 1.0
camera_focal_length = 700


object_width = 0.5


def estimate_distance(box, frame_height):
    """Estimate distance to object based on object size in the image and camera parameters."""
    box_height = box[3] - box[1]
    distance = (object_width * camera_focal_length) / box_height

    return distance

def log_detections(class_id, confidence, box, distance):
    """Log detection details to a file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"{model.names[class_id]}"
    log_entry = f"{timestamp}, {label}, {confidence:.2f}, {box}, {distance:.2f}m\n"
    with open("detections_log.txt", "a") as log_file:
        log_file.write(log_entry)

#  video capture
cap = cv2.VideoCapture(0)  # 0 for the first webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width, _ = frame.shape

    # Perform inference
    results = model(frame, stream=True)

    # Loop through the results
    for result in results:
        # Extract bounding boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Loop through the detections
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box[:4])  # Extract coordinates
            label = f"{model.names[class_id]} {confidence:.2f}"  # Create label
            color = (0, 255, 0)  # Default color for bounding box

            # Estimate distance to the object
            distance = estimate_distance(box, frame_height)
            label += f" {distance:.2f}m"

            # Set color based on class (example: car=blue, person=red)
            if model.names[class_id] == "car":
                color = (255, 0, 0)
            elif model.names[class_id] == "person":
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Log detections
            log_detections(class_id, confidence, box, distance)

    # Display the frame
    cv2.imshow("Video", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
