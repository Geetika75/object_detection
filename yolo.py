from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv5s model and custom model
coco_model = YOLO("yolov5s.pt")
custom_model = YOLO("runs/detect/train2/weights/best.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run both models
    coco_results = coco_model(frame, conf=0.5)[0]
    custom_results = custom_model(frame, conf=0.5)[0]

    combined_frame = frame.copy()

    # Inference
    combined_frame = coco_results.plot(line_width=2, img=combined_frame)
    combined_frame = custom_results.plot(line_width=2, img=combined_frame) # Draw boxes and labels

    # Display result
    cv2.imshow("YOLO Real-Time Detection", combined_frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
