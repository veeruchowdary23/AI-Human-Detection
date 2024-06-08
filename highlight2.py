import cv2
import mediapipe as mp
import numpy as np
import concurrent.futures
import warnings
import pytesseract

# Suppress the deprecated warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Path to Tesseract executable (change this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Open the video file or capture device
video_path = 'Video1.mov'  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
output_path = 'volleyball_3_video.mp4'  # Change this to your desired output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set for keeping track of detected numbers
detected_numbers = set()

# Function to detect the color red in a given frame
def detect_red_jersey(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    return mask

# Function to extract jersey number using Tesseract OCR
def extract_jersey_number(roi):
    # Preprocess the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Further preprocessing for better OCR results
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Perform OCR
    number = pytesseract.image_to_string(binary, config='--psm 6 -c tessedit_char_whitelist=0123456789').strip()
    return number

# Function to validate the extracted number is within the range of 0-100
def is_valid_number(number):
    try:
        num = int(number)
        if 0 <= num <= 100:
            return True
        else:
            return False
    except ValueError:
        return False

# Function to process each frame
def process_frame(frame):
    global detected_numbers

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame for better performance
    image_resized = cv2.resize(image_rgb, (640, 360))

    # Process the frame with MediaPipe Holistic
    results = holistic.process(image_resized)

    # Detect red jerseys
    red_mask = detect_red_jersey(frame)
    red_output = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Draw landmarks only if a red jersey is detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2))
    
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1))

    # Highlight the person with the red jersey and display the jersey number
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]
            number = extract_jersey_number(roi)
            if is_valid_number(number) and number not in detected_numbers:
                detected_numbers.add(number)
                print(f"Detected jersey number: {number}")
                cv2.putText(frame, f"Jersey: {number}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle around detected red jersey

    return frame

# Function to read and process frames
def read_and_process_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        frame = process_frame(frame)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Human Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Use threading for video reading and processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(read_and_process_frames)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
