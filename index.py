import cv2
import random
import numpy as np
from ultralytics import YOLO

# Define virtual sensor classes
class VirtualSensor:
    def __init__(self, sensor_type, location):
        self.type = sensor_type
        self.location = location

    def simulate_data(self):
        # Simulate sensor readings (placeholder)
        if self.type == "laser":
            return random.uniform(0, 10)  # Simulate laser distance
        elif self.type == "acoustic":
            return random.uniform(20, 80)  # Simulate sound level in dB
        else:
            return None

# Load YOLO model
model = YOLO('yolov8n.pt')  # Or any other suitable YOLO model

# Initialize virtual sensors (excluding camera which will be real)
sensors = [
    VirtualSensor("laser", (10, 10)),
    VirtualSensor("acoustic", (20, 20))
]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Main loop
while True:
    sensor_data = {}
    for sensor in sensors:
        data = sensor.simulate_data()
        sensor_data[sensor.type] = data
    
    # Get real camera frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from webcam")
        break
    
    sensor_data["camera"] = frame
    
    # Process camera data with YOLO
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()
    cv2.imshow("Camera Feed", annotated_frame)
    
    # Extract detections
    detections = results[0].boxes.data.tolist()

    # Anomaly detection logic based on YOLO results
    for detection in detections:
        class_id = int(detection[5])
        confidence = detection[4]
        # Add your custom anomaly detection logic here
        # For example, alert if a person is detected with low confidence
        if class_id == 0 and confidence < 0.6:  # 0 is the class ID for 'person'
            print("Anomaly detected: Low confidence person detected")

    # Combine sensor data and generate alerts (placeholder)
    # Example: alert if laser distance is below threshold or sound level is high
    if "laser" in sensor_data and sensor_data["laser"] < 2:
        print("Anomaly detected: Laser distance threshold exceeded.")
    if "acoustic" in sensor_data and sensor_data["acoustic"] > 75:
        print("Anomaly detected: High sound level detected.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
