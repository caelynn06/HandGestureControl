# config.py
"""
Configuration settings for gesture control
"""

# Camera settings
CAMERA_INDEX = 0 # which camera to use (change if needed)

# MediaPipe settings
MAX_HANDS = 2 # detect max of 2 hands
MODEL_COMPLEXITY = 1 # affects accuracy vs speed trade off (0 lite model, 1 complex)
MIN_DETECTION_CONFIDENCE = 0.8 # minimum confidence level required to detect hand
MIN_TRACKING_CONFIDENCE = 0.8 # minimum confidence level required to track hand

# Gesture cooldowns (seconds)
HANG_COOLDOWN = 10.0 
ROCK_COOLDOWN = 10.0
VOLUME_COOLDOWN = 0.1
PINCH_COOLDOWN = 0.15

# Pinch detection thresholds (pixels)
PINCH_THRESHOLD = 55
RELEASE_THRESHOLD = 80

# Mouse smoothing settings
# Number of previous positions to store for averaging
# Higher value = smoother but more lag
# Lower value = more responsive but jittery
MOUSE_BUFFER_SIZE = 7
MOUSE_EXPONENTIAL_WEIGHT = 0.25 # Weight for exponential smoothing (0.0 to 1.0)
EDGE_DAMPENING_MARGIN = 0.15 # Reduces sensitivity near screen edges to prevent cursor from getting stuck


# Display settings
DISPLAY_FPS = True # displays FPS on top left
DISPLAY_GESTURE = True # display name of gesture