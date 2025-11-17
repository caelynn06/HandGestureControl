# main.py
"""
Main application file for hand gesture control
This is the complete, production-ready version with smooth mouse control
"""

# Import required libraries
import cv2  # OpenCV - for camera and image processing
import mediapipe as mp  # MediaPipe - for hand detection
import time  # For timing and cooldowns
import pyautogui  # For controlling mouse and keyboard

# Import our custom modules
from gestures import (
    fingers_up,      # Function to detect which fingers are up
    is_fist,         # Check if hand is in fist position
    is_open_palm,    # Check if hand is in open palm position
    is_picks_up,     # Check if hand is in hang loose gesture
    is_rock_roll,    # Check if hand is in rock on gesture
    detect_pinch     # Calculate pinch distance between thumb and index
)
from actions import (
    open_myutep,     # Function to open my.utep.edu
    open_spotify,    # Function to open Spotify
    change_volume    # Function to change system volume
)
from mouse_smoother import MouseController  # Our smooth mouse control class
import config  # Configuration file with all settings


# Disable PyAutoGUI failsafe feature
# Normally, moving mouse to corner triggers failsafe - we don't want that
pyautogui.FAILSAFE = False

def main():
    """Main application loop - this runs the entire program"""    

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    mpHands = mp.solutions.hands 
    
    # Create Hands object with settings from config file
    hands = mpHands.Hands(
        max_num_hands=config.MAX_HANDS,  # How many hands to detect (usually 2)
        model_complexity=config.MODEL_COMPLEXITY,  # 0=lite, 1=full (accuracy vs speed)
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,  # Confidence to detect hand
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE  # Confidence to keep tracking
    )
    
    mpDraw = mp.solutions.drawing_utils

    # Initialize mouse controller with smooth movement
    screen_w, screen_h = pyautogui.size()  # Get screen dimensions
    # Create MouseController object with screen size and edge dampening
    mouse_controller = MouseController(screen_w, screen_h, config.EDGE_DAMPENING_MARGIN)

    # For FPS calculation
    pTime = 0  # Previous time
    
    # Cooldown timers - prevent gestures from triggering too frequently
    last_hang_time = 0   # Last time hang loose gesture was triggered
    last_rock_time = 0   # Last time rock on gesture was triggered
    last_vol_time = 0    # Last time volume was changed
    last_pinch_time = 0  # Last time pinch was detected

    # Track if we're currently dragging with pinch
    dragging = False

    # Print startup messages
    print("Starting hand gesture control...")
    print("Press ESC to exit")

    while True:
        # Read a frame from the camera
        # success = True if frame was captured successfully
        # img = the actual image/frame
        success, img = cap.read()
        
        # If camera read failed, skip this iteration
        if not success:
            continue

        # Flip image horizontally for mirror effect (more intuitive)
        img = cv2.flip(img, 1)
        
        # Get frame dimensions
        # h = height in pixels
        # w = width in pixels  
        # c = number of color channels (3 for BGR)
        h, w, c = img.shape

        # Convert from BGR (OpenCV format) to RGB (MediaPipe format)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hands
        results = hands.process(imgRGB)

        # Initialize empty landmark list
        lmList = []
        
        # Default gesture label
        gesture_label = "NONE"
        
        # Get current time for cooldown checks
        now = time.time()
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Get the first detected hand (index 0)
            handLms = results.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Extract all 21 landmark positions
            for id, lm in enumerate(handLms.landmark):
                cx = int(lm.x * w)  # X position in pixels
                cy = int(lm.y * h)  # Y position in pixels
                # Add to list: [landmark_id, x_position, y_position]
                lmList.append([id, cx, cy])

        # Only process gestures if hand was detected (lmList not empty)
        if len(lmList) != 0:
        
            fingers = fingers_up(lmList)

           
            if is_picks_up(fingers):
                gesture_label = "PICKS UP!"  # Set label for display
                
                # Check if enough time has passed since last trigger (cooldown)
                if now - last_hang_time > config.HANG_COOLDOWN:
                    open_myutep()  
                    last_hang_time = now  # Update last trigger time

            elif is_rock_roll(fingers):
                gesture_label = "ROCK ON"
                
                # Check cooldown
                if now - last_rock_time > config.ROCK_COOLDOWN:
                    open_spotify()  # Open Spotify app
                    last_rock_time = now

  
            elif is_open_palm(fingers):
                gesture_label = "VOLUME UP"
                
                # Check cooldown
                if now - last_vol_time > config.VOLUME_COOLDOWN:
                    change_volume("UP")  # Increase system volume
                    last_vol_time = now

        
            elif is_fist(fingers):
                gesture_label = "VOLUME DOWN"
                
                # Check cooldown
                if now - last_vol_time > config.VOLUME_COOLDOWN:
                    change_volume("DOWN")  # Decrease system volume
                    last_vol_time = now
            
            # Get index finger tip position (landmark 8)
            ix = lmList[8][1]  # X coordinate
            iy = lmList[8][2]  # Y coordinate
            
            # Process movement through our smooth mouse controller
            # This handles: coordinate mapping, edge dampening, and smoothing
            mouse_controller.process_movement(ix, iy, w, h)
            
            # Calculate distance between thumb tip and index tip
            # Returns: distance in pixels, and coordinates of both tips
            pinch_distance, (x1, y1), (x2, y2) = detect_pinch(lmList)

          
            if pinch_distance < config.RELEASE_THRESHOLD:
                # Choose color based on distance
                # Green if pinched, yellow if close
                if pinch_distance < config.PINCH_THRESHOLD:
                    color = (0, 255, 0)  # Green = pinched
                else:
                    color = (0, 255, 255)  # Yellow = getting close
                
                # Draw line connecting thumb and index
                cv2.line(img, (x1, y1), (x2, y2), color, 3)
                
                cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)  # Thumb
                cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)  # Index
            
            # Start drag/click when pinch is detected
            if (pinch_distance < config.PINCH_THRESHOLD and  # Fingers are close enough
                not dragging and  # Not already dragging
                now - last_pinch_time > config.PINCH_COOLDOWN):  # Cooldown passed
                
                pyautogui.mouseDown()  # Press mouse button down
                dragging = True  # Set dragging state to true
                last_pinch_time = now  # Update last pinch time
                gesture_label = "PINCH (DRAG)"  # Update display label

            # Release drag when fingers spread apart
            elif pinch_distance > config.RELEASE_THRESHOLD and dragging:
                pyautogui.mouseUp()  # Release mouse button
                dragging = False  # Set dragging state to false
                gesture_label = "RELEASE"  # Update display label
        
        # Display FPS (frames per second) if enabled in config
        if config.DISPLAY_FPS:
            # Calculate FPS
            cTime = time.time()  # Current time
            fps = 1 / (cTime - pTime) if pTime else 0  # Frames per second calculation
            pTime = cTime  # Update previous time
            
            # Draw FPS text on screen
            cv2.putText(
                img,  # Image to draw on
                f'FPS: {int(fps)}',  # Text to display
                (10, 40),  # Position (top-left area)
                cv2.FONT_HERSHEY_PLAIN,  # Font type
                2,  # Font scale (size)
                (0, 0, 0),  # Color (black)
                2  # Thickness
            )

        # Display current gesture if enabled in config
        if config.DISPLAY_GESTURE:
            cv2.putText(
                img,  # Image to draw on
                f'Gesture: {gesture_label}',  # Current gesture text
                (10, 80),  # Position (below FPS)
                cv2.FONT_HERSHEY_PLAIN,  # Font type
                2,  # Font scale
                (0, 0, 0),  # Color (black)
                2  # Thickness
            )
        
        cv2.imshow("Hand Gesture Control", img)

        # Wait 1 millisecond for key press
        # Check if ESC key (ASCII 27) was pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break  # Exit the main loop
  
    # Release the camera
    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()  # Start the program