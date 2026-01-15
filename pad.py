import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Create a blank image for the drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Define color and thickness for drawing and erasing
draw_color = (255, 0, 255)
eraser_color = (0, 0, 0)
draw_thickness = 5
eraser_thickness = 30

# Initialize previous points
prev_x, prev_y = 0, 0
prev_erase_x, prev_erase_y = 0, 0

while True:
    # Capture the frame
    success, img = cap.read()
    if not success:
        break

    # Flip the image for mirror effect
    img = cv2.flip(img, 1)

    # Find the hand and landmarks
    hands, img = detector.findHands(img)

    if hands:
        # Get the first hand's landmark positions
        lmList = hands[0]['lmList']

        # Get the index finger tip position (point 8)
        x, y = lmList[8][0], lmList[8][1]

        # Check which fingers are up
        fingers = detector.fingersUp(hands[0])

        if fingers[1] and not fingers[2]:
            # Drawing mode: only index finger is up
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(img, (prev_x, prev_y), (x, y), draw_color, draw_thickness)
            cv2.line(imgCanvas, (prev_x, prev_y), (x, y), draw_color, draw_thickness)

            prev_x, prev_y = x, y

        elif fingers[1] and fingers[2]:
            # Eraser mode: both index and middle fingers are up
            if prev_erase_x == 0 and prev_erase_y == 0:
                prev_erase_x, prev_erase_y = x, y

            cv2.line(img, (prev_erase_x, prev_erase_y), (x, y), eraser_color, eraser_thickness)
            cv2.line(imgCanvas, (prev_erase_x, prev_erase_y), (x, y), eraser_color, eraser_thickness)

            prev_erase_x, prev_erase_y = x, y

        else:
            prev_x, prev_y = 0, 0
            prev_erase_x, prev_erase_y = 0, 0

    # Create a transparent effect
    alpha = 0.5  # Transparency factor
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.addWeighted(img, 1 - alpha, imgCanvas, alpha, 0)

    # Display the image
    cv2.imshow("Drawing Pad", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
