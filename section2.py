# section2_landmarks.py
"""
SECTION 2: Understanding Hand Landmarks
Learn about the 21 hand landmarks and their coordinates

Learning Objectives:
- Understand landmark indexing (0-20)
- Extract x, y coordinates
- Print landmark positions
- Identify key landmarks (thumb tip, index tip, etc.)
"""

import cv2 # import OpenCV
import mediapipe as mp # import MediaPipe

# Hand landmark indices -- See image for better reference!
LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
}

def main():
    print("=" * 50)
    print("SECTION 2: Understanding Landmarks")
    print("=" * 50)
    print("\nHand has 21 landmarks (index 0-20):")
    print("- 0: Wrist")
    print("- 1-4: Thumb (base to tip)")
    print("- 5-8: Index finger (base to tip)")
    print("- 9-12: Middle finger")
    print("- 13-16: Ring finger")
    print("- 17-20: Pinky finger")
    print("\nPress ESC to exit\n")
    
    cap = cv2.VideoCapture(0) # turn on camera
    
    mpHands = mp.solutions.hands # initialize MediaPipe (same as section 1)
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mpDraw = mp.solutions.drawing_utils
    
    # Track which landmark to highlight
    highlight_index = 0
    frame_count = 0 
    
    while True: # start camera loop
        success, img = cap.read() # read frame from webcam
        if not success: # if frame not captured, continue
            continue
        
        img = cv2.flip(img, 1) # mirror image
        h, w, c = img.shape # extract height, width, and channels of the frame
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        results = hands.process(imgRGB) # run hand detecion model
        
        if results.multi_hand_landmarks: # if hand is detected, else on line 109!
            handLms = results.multi_hand_landmarks[0] # get first detected hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Draw 21 landmarks + connections on the image

            
            # Extract all landmarks into a list
            lmList = []
            for id, lm in enumerate(handLms.landmark): # loop through all 21 landmarks
                # lm.x and lm.y are normalized (0-1), convert to pixel coordinates
                cx = int(lm.x * w) # Convert normalized x (0–1) to pixel x
                cy = int(lm.y * h) # Convert normalized y (0–1) to pixel y
                lmList.append([id, cx, cy])  # Save landmark: [ID, x, y]
            
            frame_count += 1 # increment frame count
            
            # Highlight rotating landmark
            highlight_index = (frame_count // 30) % 21  # Change every 30 frames
            highlight_id, highlight_x, highlight_y = lmList[highlight_index]
            
            # Draw large circle on highlighted landmark
            cv2.circle(img, (highlight_x, highlight_y), 15, (0, 255, 255), cv2.FILLED)
            
            landmark_name = LANDMARK_NAMES.get(highlight_id, f"ID {highlight_id}") # display the landmark info

            info_text = f"Landmark {highlight_id}: {landmark_name}" # create text labels showing landwark and coordinates
            coord_text = f"Position: ({highlight_x}, {highlight_y})"
            
            # draw landmark name on screen
            cv2.putText(img, info_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, coord_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # TEST CASE: Print all landmarks to console (press 'p')
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        else: # else hand not shown
            cv2.putText(img, "Show your hand!", (10, 40),  
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # same as section one
        cv2.imshow("Section 2: Understanding Landmarks", img)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nSection 2 complete! ✓")
    print("Next: Section 3 - Finger Detection\n")


if __name__ == "__main__":
    main()