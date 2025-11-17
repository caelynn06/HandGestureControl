# section3_finger_detection.py
"""
SECTION 3: Finger Detection
Learn to detect which fingers are up or down

Learning Objectives:
- Detect finger states (up/down)
- Understand finger tip vs base comparison
- Count raised fingers
- Create a finger state array
"""

import cv2
import mediapipe as mp

def fingers_up(lmList): 
    """
    Detect which fingers are raised
    Returns: [thumb, index, middle, ring, pinky] where 1=up, 0=down
    """
    fingers = []
    
    # THUMB: Special case - check horizontal position
    # If thumb tip (4) is to the RIGHT of thumb IP joint (3), it's up
    # (This works for right hand; flip for left hand if needed)
    if lmList[4][1] > lmList[3][1]: 
        fingers.append(1)
    else:
        fingers.append(0)
    
    # OTHER FINGERS: Check if tip is ABOVE the pip joint
    # Tip IDs: [8=index, 12=middle, 16=ring, 20=pinky]
    tipIds = [8, 12, 16, 20]
    
    for tip in tipIds:
        # If tip's Y coordinate is LESS than pip's Y (tip is higher), finger is up
        if lmList[tip][2] < lmList[tip - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers


def main():
    print("=" * 50)
    print("SECTION 3: Finger Detection")
    print("=" * 50)
    print("\nHow finger detection works:")
    print("- Compare finger tip position to base joint")
    print("- If tip is above base, the finger is UP")
    print("- If tip is below base, the finger is DOWN")
    print("\nTry different hand poses:")
    print("-  Fist (all fingers down)")
    print("-  Open palm (all fingers up)")
    print("-  Pointing (only index up)")
    print("-  Peace sign (index + middle up)")
    print("\nPress ESC to exit\n")
    
    cap = cv2.VideoCapture(0) # turn on camera
    
    mpHands = mp.solutions.hands # same as previos sections
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks: # Same as prev.
            handLms = results.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Extract landmarks
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h) # find their x and y coordinates
                lmList.append([id, cx, cy])
            
            # Detect which fingers are up
            fingers = fingers_up(lmList)
            total_fingers = fingers.count(1)
            
            # Print to console every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"\nFinger States: {fingers}")
                print(f"  Thumb:  {'UP ' if fingers[0] else 'DOWN '}")
                print(f"  Index:  {'UP ' if fingers[1] else 'DOWN '}")
                print(f"  Middle: {'UP ' if fingers[2] else 'DOWN '}")
                print(f"  Ring:   {'UP ' if fingers[3] else 'DOWN '}")
                print(f"  Pinky:  {'UP ' if fingers[4] else 'DOWN '}")
                print(f"Total fingers up: {total_fingers}")
            
            # Display finger count
            cv2.putText(img, f"Fingers Up: {total_fingers}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Display individual finger states
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            y_pos = 100
            for i, (name, state) in enumerate(zip(finger_names, fingers)):
                color = (0, 255, 0) if state else (0, 0, 255)
                status = "UP" if state else "DOWN"
                cv2.putText(img, f"{name}: {status}", (10, y_pos + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Highlight finger tips based on state
            tip_ids = [4, 8, 12, 16, 20]
            for i, tip_id in enumerate(tip_ids):
                x, y = lmList[tip_id][1], lmList[tip_id][2]
                color = (0, 255, 0) if fingers[i] else (0, 0, 255)
                cv2.circle(img, (x, y), 10, color, cv2.FILLED)
            
            # TEST CASE: Recognize common gestures
            gesture = "Unknown"
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "FIST "
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "OPEN PALM "
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "POINTING "
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "PEACE "
            elif fingers == [1, 0, 0, 0, 1]:
                gesture = "PICKS UP"
            elif fingers == [0, 1, 0, 0, 1]:
                gesture = "ROCK ON "
            elif fingers == [1, 1, 0, 0, 0]:
                gesture = "GUN "
            
            cv2.putText(img, f"Gesture: {gesture}", (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        else:
            cv2.putText(img, "Show your hand!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(img, "Press ESC to exit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Section 3: Finger Detection", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nSection 3 complete!")
    print("Next: Section 4 - Distance Calculation\n")


if __name__ == "__main__":
    main()