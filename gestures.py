# gestures.py
"""
Hand gesture recognition functions
"""

def fingers_up(lmList): # create a list of the fingers that are up, similar to previous sections
    """Detect which fingers are up based on hand landmarks"""
    fingers = []

    # Thumb → compare x positions
    fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)

    tipIDs = [8, 12, 16, 20] # landmark tips of all fingers
    for tip in tipIDs:
        fingers.append(1 if lmList[tip][2] < lmList[tip-2][2] else 0)

    return fingers


def is_fist(fingers):
    """Check if hand is in fist position"""
    return fingers == [0, 0, 0, 0, 0]


def is_open_palm(fingers):
    """Check if hand is in open palm position"""
    return fingers == [1, 1, 1, 1, 1]

def is_picks_up(fingers):
    """Check if hand is in picks up position (thumb + pinky)"""
    return fingers == [1, 0, 0, 0, 1]


def is_rock_roll(fingers):
    """Check if hand is in rock on position (index + pinky)"""
    return fingers == [0, 1, 0, 0, 1]


def detect_pinch(lmList, thumb_id=4, index_id=8):
    """
    Calculate distance between thumb and index finger
    Returns: distance in pixels
    """
    import math
    x1, y1 = lmList[thumb_id][1], lmList[thumb_id][2]
    x2, y2 = lmList[index_id][1], lmList[index_id][2]
    return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)