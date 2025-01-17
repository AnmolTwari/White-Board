import cv2
import numpy as np
import mediapipe as mp

# Initializing Mediapipe hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Create a blank canvas
canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Variables for drawing
drawing_color = (0, 0, 255)  # Red
drawing = False
eraser_mode = False
brush_size = 5
last_x, last_y = None, None  # Track the last position for smoother lines
space_distance_threshold = 30  # Threshold for smooth drawing

# Palette setup
palette_positions = [(50, 50), (150, 50), (250, 50), (350, 50)]
palette_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]  # Red, Green, Blue, Black

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw color palette
    for i, (x, y) in enumerate(palette_positions):
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), palette_colors[i], -1)
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip position
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if the fingertip is in the palette area to change the drawing color
            for i, (px, py) in enumerate(palette_positions):
                if px - 20 < x < px + 20 and py - 20 < y < py + 20:
                    drawing_color = palette_colors[i]
                    break

            # Draw on the canvas if drawing mode is enabled
            if drawing:
                if last_x is not None and last_y is not None:
                    distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                    if distance < space_distance_threshold:  # Only draw if close enough
                        if eraser_mode:
                            cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), brush_size * 2)
                        else:
                            cv2.line(canvas, (last_x, last_y), (x, y), drawing_color, brush_size)
                last_x, last_y = x, y
            else:
                last_x, last_y = None, None  # Reset positions if not drawing

            # Change cursor color based on mode
            cursor_color = (0, 0, 255) if eraser_mode else (0, 255, 0)
            cv2.circle(frame, (x, y), 10, cursor_color, -1)

    # Resize the canvas to match the frame height
    resized_canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

    # Combine the camera frame and the canvas
    combined = np.hstack((frame, resized_canvas))
    cv2.imshow('Whiteboard', combined)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):  # Toggle drawing mode
        drawing = not drawing
        print(f"Drawing mode: {'ON' if drawing else 'OFF'}")
    elif key == ord('e'):  # Toggle eraser mode
        eraser_mode = not eraser_mode
        print(f"Eraser mode: {'ON' if eraser_mode else 'OFF'}")
    elif key == ord('q'):  # Quit the application
        break

cap.release()
cv2.destroyAllWindows()
