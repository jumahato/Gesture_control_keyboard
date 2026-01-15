import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Screen resolution (change as needed)
screen_width, screen_height = pyautogui.size()

# Variable to store the state of the click
clicking = False

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Find the hands and their landmarks
    hands, img = detector.findHands(img)

    # If a hand is detected
    if hands:
        # Get the first hand detected
        hand = hands[0]

        # Get the landmarks for the index finger and thumb tips
        lmList = hand["lmList"]
        index_finger_tip = lmList[8]  # Index finger tip is landmark 8
        thumb_tip = lmList[4]  # Thumb tip is landmark 4

        # Get the x, y position of the index finger tip
        x, y = index_finger_tip[0], index_finger_tip[1]

        # Map the coordinates to the screen size
        screen_x = np.interp(x, [0, 640], [0, screen_width])
        screen_y = np.interp(y, [0, 480], [0, screen_height])

        # Move the mouse cursor
        pyautogui.moveTo(screen_x, screen_y)

        # Calculate the distance between the index finger and thumb
        distance, _, _ = detector.findDistance(lmList[8][:2], lmList[4][:2])

        # Check if the distance is less than a threshold (indicating a click)
        if distance < 20 and not clicking:
            clicking = True
            pyautogui.click()
        elif distance >= 20:
            clicking = False

    # Display the frame
    cv2.imshow("Image", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
