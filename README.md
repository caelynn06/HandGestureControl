# HandGestureControl
Workshop for GWC semester project that shows students the basics of MediaPipe and OpenCV

## Set up your virtual enviroment!
### Create Virtual Environment:  
 python -m venv .venv
  
  
### Activate enviroment:  

Mac:  
 source myenv/bin/activate


Linux:  
 source .venv/bin/activate

  
Windows:  
 .venv\Scripts\activate

## Install neccessary libraries:  
 pip install opencv-python mediapipe pyautogui numpy



# config.py
Configuration settings for gesture control


Set the camera index:
```python
CAMERA_INDEX = 0  
```
MediaPipe settings:  
```python
MAX_HANDS = 2  
MODEL_COMPLEXITY = 1  
MIN_DETECTION_CONFIDENCE = 0.8  
MIN_TRACKING_CONFIDENCE = 0.8   
```
Gesture cooldowns:  
```python
HANG_COOLDOWN = 10.0  
ROCK_COOLDOWN = 10.0  
VOLUME_COOLDOWN = 0.1  
PINCH_COOLDOWN = 0.15  
```
Pinch detection thresholds:
```python
PINCH_THRESHOLD = 55  
RELEASE_THRESHOLD = 80  
```
Mouse smoothing settings:
```python
MOUSE_BUFFER_SIZE = 7  
MOUSE_EXPONENTIAL_WEIGHT = 0.25  
EDGE_DAMPENING_MARGIN = 0.15  
```
Display settings:
```python
DISPLAY_FPS = True  
DISPLAY_GESTURE = True
```


# gestures.py
Set up the gestures we will be using:


Create a list of the fingers that are up: 
```python
def fingers_up(lmList): 
    fingers = []

    # Thumb → compare x positions
    fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

    tipIDs = [8, 12, 16, 20] # landmark tips of all fingers
    for tip in tipIDs:
        fingers.append(1 if lmList[tip][2] < lmList[tip-2][2] else 0)

    return fingers
```
Check if hand is in fist position: 
```python
def is_fist(fingers):
    return fingers == [0, 0, 0, 0, 0]
```
Check if hand is open:
```python
def is_open_palm(fingers):
    return fingers == [1, 1, 1, 1, 1]
```
Check if hand is in picks up position:
```python
def is_picks_up(fingers):
    return fingers == [1, 0, 0, 0, 1]
```
Check if hand is in rock in roll position:
```python
def is_rock_roll(fingers):
    return fingers == [0, 1, 0, 0, 1]
```
Calculate the distance between thumb and index finger: 
```python
def detect_pinch(lmList, thumb_id=4, index_id=8):
    import math
    x1, y1 = lmList[thumb_id][1], lmList[thumb_id][2]
    x2, y2 = lmList[index_id][1], lmList[index_id][2]
    return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)
```
# actions.py
Create actions for the gestures:


Import all necessary libraries:
```python
import subprocess # for running system commands
import platform # for detecting operating system
import os # for operating system interaction
import pyautogui # for keyboard control
```

Open Utep website with picks up:
```python
def open_myutep(): 
    url = "https://my.utep.edu" # feel free to change the link!
    system = platform.system()  # Detect which operating system we're running on

    try:
        if system == "Windows":
            subprocess.Popen(['start', url], shell=True)
        elif system == "Darwin": # macOS
            subprocess.Popen(['open', url])
        else:
            subprocess.Popen(['xdg-open', url])
        print("Opened my UTEP")
    except Exception as e:
        print("Error opening UTEP:", e)thon
import subprocess # for running system commands
import platform # for detecting operating system
import os # for operating system interaction
import pyautogui # for keyboard control
```
Open spotify with rock gesture:
```python
ef open_spotify(): # os system calls spotify
    print("Opening Spotify")
    try:
        os.system("start spotify") # feel free to customize!
    except Exception as e:
        print("Error opening Spotify:", e)
```
Change the volume with palm:
```python
def change_volume(direction):
    system = platform.system()
    try:
        if system in ("Windows", "Darwin"):
            pyautogui.press("volumeup" if direction == "UP" else "volumedown")
        else:
            step = "5%+" if direction == "UP" else "5%-" # increase/decrease volume by 5%
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", step])
            # "@DEFAULT_SINK@" means the default audio output device

    except Exception as e:
        print("Volume error:", e)

```
# mouse_smoother.py
Mouse smoothing and movement control  


Import necessary libraries:  
```python
from collections import deque  # Double-ended queue for efficient buffer operations
import pyautogui  # For controlling the mouse cursor
import numpy as np  # For mathematical operations (not used here but common in similar code)
```
 Smooth mouse movement using moving average and exponential smoothing.   
    This class stores recent mouse positions and calculates a smooth average.   

    
  Two smoothing techniques are combined:   
  1. Moving Average - average of last N positions   
  2. Exponential Smoothing - weighted average favoring recent positions  


  Initialize the smoothing system.  
        
  Args:  
            buffer_size: How many recent positions to store (default: 5)  
                        Larger = smoother but more lag  
                        Smaller = more responsive but jittery  
            exponential_weight: Weight for new positions (0.0 to 1.0, default: 0.3)  
                               Higher = more responsive  
                               Lower = smoother  
```python
 
    def __init__(self, buffer_size=5, exponential_weight=0.3):
        # Store the buffer size
        self.buffer_size = buffer_size
        
        # Store the exponential smoothing weight
        self.exp_weight = exponential_weight
        
        # Create a deque (double-ended queue) to store positions
        # maxlen=buffer_size means old positions automatically drop off when full
        # This is more efficient than a regular list for this purpose
        self.position_buffer = deque(maxlen=buffer_size)
        
        # Store the current smoothed position
        # None at start because we haven't received any positions yet
        self.smooth_x = None
        self.smooth_y = None
```
 Add a new mouse position to the smoothing buffer.  
        This should be called every frame with the raw cursor position.  
        
  Args:  
            x: Target x coordinate (in screen pixels)  
            y: Target y coordinate (in screen pixels)  
```python   
    def add_position(self, x, y):
        # Append the new position to our buffer as a tuple (x, y)
        # If buffer is full, the oldest position is automatically removed
        self.position_buffer.append((x, y))
```
Calculate and return the smoothed mouse position.  
        This uses both moving average and exponential smoothing.  
        
  Returns:  
            Tuple (smooth_x, smooth_y) - the smoothed coordinates  
            or (None, None) if buffer is empty  
```python       
    def get_smoothed_position(self):
        # Check if buffer has any positions
        if not self.position_buffer:
            # No positions stored yet, return None
            return None, None
        
        # STEP 1: Calculate Moving Average
        # Sum all x-coordinates in buffer
        # pos[0] gets the x-coordinate from each position tuple
        sum_x = sum(pos[0] for pos in self.position_buffer)
        
        # Calculate average x by dividing sum by number of positions
        avg_x = sum_x / len(self.position_buffer)
        
        # Do the same for y-coordinates
        # pos[1] gets the y-coordinate from each position tuple
        sum_y = sum(pos[1] for pos in self.position_buffer)
        avg_y = sum_y / len(self.position_buffer)
        
        # STEP 2: Apply Exponential Smoothing
        # This gives extra weight to the previous smoothed position
        
        # Check if this is the first position we're calculating
        if self.smooth_x is None:
            # No previous smooth position, so just use the average
            self.smooth_x = avg_x
            self.smooth_y = avg_y
        else:
            # We have a previous smooth position, so blend it with new average
            # Formula: new_smooth = (weight × new_average) + ((1-weight) × old_smooth)
            # This creates momentum - the cursor "remembers" where it was going
            self.smooth_x = self.exp_weight * avg_x + (1 - self.exp_weight) * self.smooth_x
            self.smooth_y = self.exp_weight * avg_y + (1 - self.exp_weight) * self.smooth_y
        
        # Return the final smoothed coordinates
        return self.smooth_x, self.smooth_y
```
Handle mouse movement and clicking with advanced smoothing.  
This class manages the entire pipeline from hand position to cursor position.  


 
  Initialize the mouse controller.  
        
  Args:  
            screen_width: Width of screen in pixels  
            screen_height: Height of screen in pixels  
            edge_margin: Fraction of screen to apply edge dampening (default: 0.15)  
                        0.15 means outer 15% on each side has reduced sensitivity  
```python
class MouseController:
    def __init__(self, screen_width, screen_height, edge_margin=0.15):
        # Store screen dimensions
        self.screen_w = screen_width
        self.screen_h = screen_height
        
        # Store edge margin for dampening
        self.edge_margin = edge_margin
        
        # Create a MouseSmoothing object with specific settings
        # buffer_size=7 stores last 7 positions
        # exponential_weight=0.25 gives 25% weight to new positions
        self.smoother = MouseSmoothing(buffer_size=7, exponential_weight=0.25)
```
  Process hand position and move mouse cursor smoothly.  
        This is the main function that gets called every frame.  
        
  Args:  
            hand_x: X position of hand landmark in camera frame  
            hand_y: Y position of hand landmark in camera frame  
            frame_width: Width of camera frame in pixels  
            frame_height: Height of camera frame in pixels  
```python 
    def process_movement(self, hand_x, hand_y, frame_width, frame_height):

        # STEP 1: Normalize coordinates to 0-1 range
        # This makes the values independent of camera resolution
        # Divide hand position by frame size to get fraction (0.0 to 1.0)
        norm_x = hand_x / frame_width   # 0.0 = left edge, 1.0 = right edge
        norm_y = hand_y / frame_height  # 0.0 = top edge, 1.0 = bottom edge
        
        # STEP 2: Apply edge dampening
        # Reduce sensitivity near screen edges to prevent overshoot
        # This makes it easier to reach screen corners and edges
        norm_x = self._apply_edge_dampening(norm_x)
        norm_y = self._apply_edge_dampening(norm_y)
        
        # STEP 3: Map to screen coordinates
        # Multiply normalized values (0-1) by screen dimensions
        # This converts to actual pixel positions on screen
        target_x = norm_x * self.screen_w  # Calculate target x position
        target_y = norm_y * self.screen_h  # Calculate target y position
        
        # STEP 4: Add to smoothing buffer
        # Send the target position to our smoother
        self.smoother.add_position(target_x, target_y)
        
        # STEP 5: Get smoothed position
        # Retrieve the smoothed coordinates from the buffer
        smooth_x, smooth_y = self.smoother.get_smoothed_position()
        
        # STEP 6: Move the actual mouse cursor
        # Only move if we have valid smoothed coordinates
        if smooth_x is not None and smooth_y is not None:
            # Try to move the mouse
            try:
                # Move cursor to smoothed position
                # duration=0 means move instantly (no pyautogui animation)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            except:
                # If move fails (invalid coordinates, etc.), just skip
                # Using bare except is normally bad practice, but okay here
                # since we just want to ignore any movement errors
                pass
   ```
 
  Reduce sensitivity near edges (0.0 or 1.0).
  This is a private helper method (indicated by _ prefix).
        
  Edge dampening prevents the cursor from overshooting at screen boundaries.
        Without this, it's very hard to reach screen corners precisely.
        
  Args:
            value: Normalized coordinate (0.0 to 1.0)
        
  Returns:
            Dampened coordinate value (still 0.0 to 1.0 range)
```python             
    def _apply_edge_dampening(self, value):
    
        # Check if we're in the LEFT edge zone (less than edge_margin)
        # For example, if edge_margin=0.15, this catches values < 0.15
        if value < self.edge_margin:
            # Apply dampening formula for left edge
            # This compresses the left 15% of movement into a smaller range
            # Result: slower movement near left edge
            return value * (1 - self.edge_margin) / self.edge_margin
            
        # Check if we're in the RIGHT edge zone (greater than 1-edge_margin)
        # For example, if edge_margin=0.15, this catches values > 0.85
        elif value > (1 - self.edge_margin):
            # Apply dampening formula for right edge
            # This compresses the right 15% of movement into a smaller range
            # Result: slower movement near right edge
            return self.edge_margin + (value - (1 - self.edge_margin)) * (1 - self.edge_margin) / self.edge_margin
            
        # If we're not in an edge zone, return value unchanged
        # This is the center 70% of the screen (if edge_margin=0.15)
        return value
```
# main.py
```python
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
```
Disable PyAutoGUI failsafe feature  
Normally, moving mouse to corner triggers failsafe - we don't want that  
```python
pyautogui.FAILSAFE = False
```
Main application loop - this runs the entire program


Intitial set up, basic same code as section!
```python
def main():

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
```
Cooldown timers - prevent gestures from triggering too frequently

```python
    last_hang_time = 0   # Last time hang loose gesture was triggered
    last_rock_time = 0   # Last time rock on gesture was triggered
    last_vol_time = 0    # Last time volume was changed
    last_pinch_time = 0  # Last time pinch was detected
```
Track if we're currently dragging with pinch and print startup messages

```python
    dragging = False

    print("Starting hand gesture control...")
    print("Press ESC to exit")
```
Set up Open CV
```python
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
```
Check for hand detection
```python
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
```
Check if list is empty, if not, continue with gestures
```python
        # Only process gestures if hand was detected (lmList not empty)
        if len(lmList) != 0:
        
            fingers = fingers_up(lmList)

  ```
Gesture == picks up
```python       
            if is_picks_up(fingers):
                gesture_label = "PICKS UP!"  # Set label for display
                
                # Check if enough time has passed since last trigger (cooldown)
                if now - last_hang_time > config.HANG_COOLDOWN:
                    open_myutep()  
                    last_hang_time = now  # Update last trigger time
```
Gesture == rock n roll
```python
            elif is_rock_roll(fingers):
                gesture_label = "ROCK ON"
                
                # Check cooldown
                if now - last_rock_time > config.ROCK_COOLDOWN:
                    open_spotify()  # Open Spotify app
                    last_rock_time = now
```
Gesture == open palm
```python  
            elif is_open_palm(fingers):
                gesture_label = "VOLUME UP"
                
                # Check cooldown
                if now - last_vol_time > config.VOLUME_COOLDOWN:
                    change_volume("UP")  # Increase system volume
                    last_vol_time = now

```
Gesture == fist
```python        
            elif is_fist(fingers):
                gesture_label = "VOLUME DOWN"
                
                # Check cooldown
                if now - last_vol_time > config.VOLUME_COOLDOWN:
                    change_volume("DOWN")  # Decrease system volume
                    last_vol_time = now
 ```
Gesture == index, Begin mouse movement
```python              
            # Get index finger tip position (landmark 8)
            ix = lmList[8][1]  # X coordinate
            iy = lmList[8][2]  # Y coordinate
            
            # Process movement through our smooth mouse controller
            # This handles: coordinate mapping, edge dampening, and smoothing
            mouse_controller.process_movement(ix, iy, w, h)
```
Calculate distance between thumb tip and index tip
```python               
            # Returns: distance in pixels, and coordinates of both tips
            pinch_distance, (x1, y1), (x2, y2) = detect_pinch(lmList)
```
Pinch logic for drag and drop
```python           
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
```
Set up FPS
```python
        # Display FPS (frames per second) if enabled in config
        if config.DISPLAY_FPS:
            # Calculate FPS
            cTime = time.time()  # Current time
            fps = 1 / (cTime - pTime) if pTime else 0  # Frames per second calculation
            pTime = cTime  # Update previous time
 ```
Display gesture and FPS on screen
```python         
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
```
Finish and program
```python
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

```
