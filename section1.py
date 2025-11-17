# section1_hand_detection.py
"""
SECTION 1: Basic Hand Detection
Learn how to detect hands and display landmarks using MediaPipe

Learning Objectives:
- Set up camera capture
- Initialize MediaPipe Hands
- Detect hand landmarks
- Display landmarks on screen
"""

import cv2 # imports openCV
import mediapipe as mp # import mediapipe

def main():
    print("=" * 50)
    print("SECTION 1: Basic Hand Detection")
    print("=" * 50)
    print("\nWhat you'll learn:")
    print("- Capture video from webcam")
    print("- Detect hands using MediaPipe")
    print("- Draw hand landmarks on screen")
    print("\nPress ESC to exit\n")
    
    # Initialize camera (0 is usually the default webcam, if it doesnt work, try 1)
    cap = cv2.VideoCapture(0) 
    
    mpHands = mp.solutions.hands # get the hands module from MediaPipe (does all the heavy lifting!)
    hands = mpHands.Hands(
        max_num_hands=2,               # Detect only 1 hand for simplicity
        min_detection_confidence=0.7,  # Minimum confidence to detect a hand
        min_tracking_confidence=0.7    # Minimum confidence to track a hand
    )
    mpDraw = mp.solutions.drawing_utils # get drawing utilities to draw landmarks on image
    
    print("Camera initialized. Show your hand to the camera!")
    
    while True: # loops the camera
        # Read frame from camera
        success, img = cap.read()
        if not success: # if camera fails
            print("Failed to read from camera")
            continue
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(imgRGB)
        
        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Get the first detected hand
            handLms = results.multi_hand_landmarks[0]
            
            # Draw the hand landmarks and connections
            mpDraw.draw_landmarks(
                img, # the image to draw on
                handLms, # the hand landmarks to draw on
                mpHands.HAND_CONNECTIONS, # draw hands connecting landmarks
                mpDraw.DrawingSpec(color=(0, 0, 0), thickness=1),    # Landmarks (The 0,0,0 is the color, feel free to customize!)
                mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1)   # Connections
            )
            
            # Display text showing hand is detected
            cv2.putText(img, "HAND DETECTED", (10, 40),  
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
             #(frame you want to draw on, text string to display, (x positon, y position), font style, font scale,)

        else:
            # Display text showing no hand detected
            cv2.putText(img, "No hand detected", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Display instructions
        cv2.putText(img, "Press ESC to exit", (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the image on a window
        cv2.imshow("Section 1: Hand Detection", img)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break 
    
    cap.release() # release the camera
    cv2.destroyAllWindows() # close all openCV windows
    print("\nSection 1 complete! ")
    print("Next: Section 2 - Understanding Landmarks\n")


if __name__ == "__main__": # call the main function to start the program
    main()