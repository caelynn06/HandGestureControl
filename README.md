# HandGestureControl
Workshop for GWC semester project that shows students the basics of MediaPipe and OpenCV

## Set up your virtual enviroment!
### Create Virtual Environment:  
- python -m venv .venv
  
  
### Activate enviroment:  

Mac:  
- source myenv/bin/activate


Linux:  
- source .venv/bin/activate

  
Windows:  
- .venv\Scripts\activate

## Install neccessary libraries:  
- pip install opencv-python mediapipe pyautogui numpy



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


