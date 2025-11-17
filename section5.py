# section5_mouse_control.py
"""
SECTION 5: Mouse Control
Learn to control the mouse cursor with hand movement

Learning Objectives:
- Map hand coordinates to screen coordinates
- Use numpy.interp() for coordinate mapping
- Move mouse cursor with pyautogui
- Understand coordinate systems
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Disable failsafe (allows cursor to go to corners)
pyautogui.FAILSAFE = False


def main():
    print("=" * 50)
    print("SECTION 5: Mouse Control")
    print("=" * 50)
    print("\nLearning mouse control:")
    print("- Map camera coordinates -> screen coordinates")
    print("- numpy.interp() for smooth mapping")
    print("- pyautogui.moveTo() to move cursor")
    print("\nCoordinate systems:")
    print("- Camera: (0,0) at top-left, (640,480) at bottom-right")
    print(f"- Screen: (0,0) at top-left, {pyautogui.size()} at bottom-right")
    print("\nMove your INDEX FINGER to control the cursor!")
    print("\nPress ESC to exit\n")
    
    cap = cv2.VideoCapture(0)
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    
    # Get screen dimensions
    screen_w, screen_h = pyautogui.size()
    print(f"Screen size: {screen_w} x {screen_h}")
    
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
            
            # Get index finger tip position (landmark 8)
            index_x = lmList[8][1]
            index_y = lmList[8][2]
            
            # Map camera coordinates to screen coordinates
            # np.interp(value, [input_min, input_max], [output_min, output_max])
            screen_x = np.interp(index_x, [0, w], [0, screen_w])
            screen_y = np.interp(index_y, [0, h], [0, screen_h])
            
            # Print coordinates every 20 frames
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"\nCamera coords: ({index_x}, {index_y})")
                print(f"Screen coords: ({int(screen_x)}, {int(screen_y)})")
                print(f"Mapping: ({index_x}/{w}) -> ({int(screen_x)}/{screen_w})")
            
            # Move the mouse cursor
            try:
                pyautogui.moveTo(screen_x, screen_y, duration=0)
            except Exception as e:
                print(f"Mouse error: {e}")
            
            # Highlight index finger tip
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 20, (255, 0, 255), 2)
            
            # Draw crosshair
            cv2.line(img, (index_x - 25, index_y), (index_x + 25, index_y), (255, 0, 255), 2)
            cv2.line(img, (index_x, index_y - 25), (index_x, index_y + 25), (255, 0, 255), 2)
            
            # Display coordinates
            cv2.putText(img, f"Camera: ({index_x}, {index_y})", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Screen: ({int(screen_x)}, {int(screen_y)})", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display screen dimensions
            cv2.putText(img, f"Screen Size: {screen_w} x {screen_h}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Visual mapping guide
            # Draw frame corners with labels
            corner_color = (0, 255, 255)
            cv2.circle(img, (0, 0), 10, corner_color, cv2.FILLED)
            cv2.putText(img, "(0,0)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
            
            cv2.circle(img, (w-1, 0), 10, corner_color, cv2.FILLED)
            cv2.putText(img, f"({w},0)", (w-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
            
            cv2.circle(img, (0, h-1), 10, corner_color, cv2.FILLED)
            cv2.putText(img, f"(0,{h})", (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
            
            cv2.circle(img, (w-1, h-1), 10, corner_color, cv2.FILLED)
            cv2.putText(img, f"({w},{h})", (w-100, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1)
            
            # TEST CASE: Print detailed mapping
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                print("\n" + "="*60)
                print("COORDINATE MAPPING DETAILS:")
                print("="*60)
                print(f"Camera frame size: {w} x {h}")
                print(f"Screen size: {screen_w} x {screen_h}")
                print(f"\nIndex finger position in camera: ({index_x}, {index_y})")
                print(f"\nX-axis mapping:")
                print(f"  Input range: [0, {w}]")
                print(f"  Output range: [0, {screen_w}]")
                print(f"  Formula: screen_x = (index_x / {w}) * {screen_w}")
                print(f"  Result: screen_x = ({index_x} / {w}) * {screen_w} = {screen_x:.1f}")
                print(f"\nY-axis mapping:")
                print(f"  Input range: [0, {h}]")
                print(f"  Output range: [0, {screen_h}]")
                print(f"  Formula: screen_y = (index_y / {h}) * {screen_h}")
                print(f"  Result: screen_y = ({index_y} / {h}) * {screen_h} = {screen_y:.1f}")
                print("="*60)
            
            if key == 27:  # ESC
                break
                
        else:
            cv2.putText(img, "Show your hand to control mouse!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(img, "Press 'M' for mapping details | ESC to exit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Section 5: Mouse Control", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nSection 5 complete!")
    print("Next: Section 6 - Complete System\n")


if __name__ == "__main__":
    main()