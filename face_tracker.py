import cv2
import mediapipe as mp

# 1. Set up MediaPipe's Face Detection AI
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 2. Turn on the webcam (0 is usually your default laptop camera)
cap = cv2.VideoCapture(0)

print("Starting VeriShield Face Tracker... Press 'q' to quit.")

# 3. Start the AI model
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
q
        # Convert the image color for the AI to understand
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find faces
        results = face_detection.process(image)

        # Draw the face bounding box
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Display the video feed on your screen
        cv2.imshow('VeriShield - Face Tracking Test', image)

        # Press 'q' to quit the window
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up and close the camera
cap.release()
cv2.destroyAllWindows()