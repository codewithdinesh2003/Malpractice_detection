import cv2

# Initialize variables to track head positions
previous_head_position = None
consecutive_suspicious_frames = 0
SUSPICIOUS_THRESHOLD = 20  # Number of consecutive frames to consider as suspicious

# Load the template image
template = cv2.imread('cell_phone_template.jpg', 0)
if template is None:
    raise ValueError("Template image not found or cannot be loaded.")

# Resize the template to a reasonable size
frame_width = 640
frame_height = 480
template_width = int(frame_width / 4)
template_height = int(frame_height / 4)
template = cv2.resize(template, (template_width, template_height))

def detect_eyes_and_head_position(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    global previous_head_position
    global consecutive_suspicious_frames
    
    suspicious = False
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mark the center of the face for head positioning
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Check if head position has moved significantly
        if previous_head_position is not None:
            movement_x = abs(center_x - previous_head_position[0])
            movement_y = abs(center_y - previous_head_position[1])
            if movement_x > 30 or movement_y > 30:
                suspicious = True
                consecutive_suspicious_frames += 1
            else:
                consecutive_suspicious_frames = 0
        previous_head_position = (center_x, center_y)
        
        # Region of Interest (ROI) for detecting eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
    
    return frame, suspicious

def detect_suspicious_activity(frame):
    # Perform eye tracking and head positioning detection
    frame, suspicious = detect_eyes_and_head_position(frame)
    
    # Convert the frame to grayscale for template matching
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Debugging: Print max_val to understand the detection results
    print(f"Max Val: {max_val}")
    
    # Set a threshold for template matching
    threshold = 0.8
    if max_val >= threshold:
        suspicious = True
        # Draw a rectangle around the detected object
        top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
    
    # Always show a message if suspicious activity is detected
    if suspicious:
        cv2.putText(frame, "Suspicious Activity Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

def capture_video_with_suspicious_activity_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_suspicious_activity(frame)
        cv2.imshow('Webcam', frame)
        
        # Terminate with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video_with_suspicious_activity_detection()
