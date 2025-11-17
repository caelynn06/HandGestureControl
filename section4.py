# section4_distance_calculation.py
"""
SECTION 4: Distance Calculation (Pinch Detection)
Learn to calculate distance between landmarks

Learning Objectives:
- Calculate distance between two points
- Use Pythagorean theorem
- Detect pinch gesture
- Understand thresholds
"""

import cv2
import mediapipe as mp
import math

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    Uses Pythagorean theorem: distance = squareroot((x2-x1)^2 + (y2-y1)^2)
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def main():
    print("=" * 50)
    print("SECTION 4: Distance Calculation")
    print("=" * 50)
    print("\nLearning distance calculation:")
    print("- Pythagorean theorem: d = squareroot((x2-x1)^2+ (y2-y1)^2)")
    print("- math.hypot() does this calculation")
    print("- Used for pinch detection (thumb + index distance)")
    print("\nTry these:")
    print("- Bring thumb and index finger together (pinch)")
    print("- Spread them apart")
    print("- Watch the distance value change")
    print("\nPress ESC to exit\n")
    
    cap = cv2.VideoCapture(0)
    
    mpHands = mp.solutions.hands # same as prev
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    
    # Distance thresholds
    PINCH_THRESHOLD = 50 # in pixels
    CLOSE_THRESHOLD = 100
    
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            # Extract landmarks
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            # Get thumb tip (4) and index finger tip (8) positions
            thumb_tip = (lmList[4][1], lmList[4][2])
            index_tip = (lmList[8][1], lmList[8][2])
            
            # Calculate distance
            distance = calculate_distance(thumb_tip, index_tip)
            
            # Print to console every 15 frames
            frame_count += 1
            if frame_count % 15 == 0:
                print(f"\nThumb position: {thumb_tip}")
                print(f"Index position: {index_tip}")
                print(f"Distance: {distance:.2f} pixels")
                
                if distance < PINCH_THRESHOLD: 
                    print("-> Status: PINCHED!")
                elif distance < CLOSE_THRESHOLD:
                    print("-> Status: CLOSE")
                else:
                    print("-> Status: FAR APART")
            
            # Determine status and color
            if distance < PINCH_THRESHOLD:
                status = "PINCHED!"
                color = (0, 255, 0)  # Green
            elif distance < CLOSE_THRESHOLD:
                status = "CLOSE"
                color = (0, 255, 255)  # Yellow
            else:
                status = "FAR"
                color = (0, 0, 255)  # Red
            
            # Draw line between thumb and index
            cv2.line(img, thumb_tip, index_tip, color, 3)
            
            # Draw circles on fingertips
            cv2.circle(img, thumb_tip, 12, color, cv2.FILLED)
            cv2.circle(img, index_tip, 12, color, cv2.FILLED)
            
            # Draw midpoint
            mid_x = (thumb_tip[0] + index_tip[0]) // 2 # mid x is thumb tip0 + index tip0 // 2
            mid_y = (thumb_tip[1] + index_tip[1]) // 2 # mid y is thumb tip1 + index tip1 // 2
            cv2.circle(img, (mid_x, mid_y), 8, (255, 0, 255), cv2.FILLED)
            
            # Display distance and status
            cv2.putText(img, f"Distance: {int(distance)} px", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, f"Status: {status}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display thresholds
            cv2.putText(img, f"Pinch Threshold: < {PINCH_THRESHOLD}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(img, f"Close Threshold: < {CLOSE_THRESHOLD}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Visual indicator bar
            bar_length = 300
            bar_thickness = 30
            bar_x, bar_y = 10, h - 80
            
            # Background bar
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_thickness), 
                         (50, 50, 50), cv2.FILLED)
            
            # Distance indicator (capped at 300 for visualization)
            indicator_width = int(min(distance, 300))
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + indicator_width, bar_y + bar_thickness), 
                         color, cv2.FILLED)
            
            # Threshold markers
            pinch_marker = int((PINCH_THRESHOLD / 300) * bar_length)
            close_marker = int((CLOSE_THRESHOLD / 300) * bar_length)
            cv2.line(img, (bar_x + pinch_marker, bar_y), 
                    (bar_x + pinch_marker, bar_y + bar_thickness), (0, 255, 0), 2)
            cv2.line(img, (bar_x + close_marker, bar_y), 
                    (bar_x + close_marker, bar_y + bar_thickness), (0, 255, 255), 2)
            
            # TEST CASE: Print detailed calculation
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):
                print("\n" + "="*50)
                print("DETAILED DISTANCE CALCULATION:")
                print("="*50)
                print(f"Thumb tip (landmark 4): x={thumb_tip[0]}, y={thumb_tip[1]}")
                print(f"Index tip (landmark 8): x={index_tip[0]}, y={index_tip[1]}")
                print(f"\nCalculation:")
                print(f"  x_diff = {index_tip[0]} - {thumb_tip[0]} = {index_tip[0] - thumb_tip[0]}")
                print(f"  y_diff = {index_tip[1]} - {thumb_tip[1]} = {index_tip[1] - thumb_tip[1]}")
                print(f"  distance = squareroot(x_diff^2 + y_diff^2)")
                print(f"  distance = squareroot({(index_tip[0] - thumb_tip[0])**2} + {(index_tip[1] - thumb_tip[1])**2})")
                print(f"  distance = {distance:.2f} pixels")
                print("="*50)
            
            if key == 27:  # ESC
                break
                
        else:
            cv2.putText(img, "Show your hand!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(img, "Press 'D' for detailed calc | ESC to exit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Section 4: Distance Calculation", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nSection 4 complete! ")
    print("Next: Section 5 - Mouse Control\n")


if __name__ == "__main__":
    main()