import cv2
import numpy as np
import pyautogui
import imutils
import mediapipe as mp

def Press(key):
    pyautogui.press(key)
    return key  # Return the key pressed for visual feedback

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# Drum kit positions and labels
drum_components = {
    'RIDE': {'pos': (100, 100), 'key': '7', 'color': (255, 0, 0), 'radius': 50},
    'RIDE BELL': {'pos': (320, 100), 'key': '8', 'color': (0, 0, 255), 'radius': 50},
    'HITHAT close': {'pos': (550, 100), 'key': '6', 'color': (255, 0, 0), 'radius': 50},
    'CRASH': {'pos': (780, 100), 'key': '9', 'color': (0, 0, 255), 'radius': 50},
    'SNARE': {'pos': (100, 320), 'key': '2', 'color': (255, 0, 0), 'radius': 50},
    'SNARE RIM': {'pos': (100, 540), 'key': '3', 'color': (0, 0, 255), 'radius': 50},
    'HIT HAT': {'pos': (820, 320), 'key': '4', 'color': (255, 0, 0), 'radius': 50},
    'HIT HAT OPEN': {'pos': (820, 540), 'key': '5', 'color': (0, 0, 255), 'radius': 50},
    'TOM HI': {'pos': (100, 660), 'key': 'q', 'color': (255, 0, 0), 'radius': 50},
    'TOM MID': {'pos': (320, 660), 'key': 'w', 'color': (0, 0, 255), 'radius': 50},
    'TOM LOW': {'pos': (550, 660), 'key': 'e', 'color': (255, 0, 0), 'radius': 50},
    'KICK': {'pos': (780, 660), 'key': '1', 'color': (0, 0, 255), 'radius': 50}
}

# For visual feedback and debouncing
active_drums = {}  # Track active drums per finger
last_trigger_time = {}  # Prevent rapid retriggering
cooldown_frames = 10  # Minimum frames between triggers

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, height=700, width=900)
    frame_count += 1
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Draw drum components
    for name, data in drum_components.items():
        pos = data['pos']
        color = data['color']
        radius = data['radius']
        
        # Highlight if active
        if name in active_drums.values():
            cv2.circle(frame, pos, radius + 10, (0, 255, 0), 3)
        cv2.circle(frame, pos, radius, color, -1)
        cv2.putText(frame, name, (pos[0] - 30, pos[1] + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Reset active drums for this frame
    active_drums = {}
    
    if results.multi_hand_landmarks:
     for hand_landmarks in results.multi_hand_landmarks:
        # Only draw index finger tip (landmark 8)
        index_tip = hand_landmarks.landmark[8]
        height, width, _ = frame.shape
        cx, cy = int(index_tip.x * width), int(index_tip.y * height)

        # Draw just the fingertip
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

        # Check drum hit
        for name, data in drum_components.items():
            pos = data['pos']
            radius = data['radius']
            distance = np.sqrt((cx - pos[0])**2 + (cy - pos[1])**2)

            if distance < radius:
                if name not in last_trigger_time or (frame_count - last_trigger_time.get(name, 0)) > cooldown_frames:
                    pressed_key = Press(data['key'])
                    active_drums[(cx, cy)] = name
                    last_trigger_time[name] = frame_count
                    break

    
    # Display the frame
    cv2.imshow("Air Drum Kit (Index Finger Control)", frame)
    
    # Exit on ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()