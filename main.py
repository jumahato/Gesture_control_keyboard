import cv2
import numpy
from cvzone.HandTrackingModule import HandDetector

class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size

    def draw(self, img, color=(0, 0, 0, 100)):  # Default to black with transparency
        overlay = img.copy()
        cv2.rectangle(overlay, self.pos, (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), color, cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, self.text, (self.pos[0] + 20, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        return img


# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 100)  # Height

# Hand detector (allow detection of up to 2 hands)
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Keyboard layout including Caps Lock
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M", "<-", "Caps"]]  # Added Caps Lock button

# Create buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == "Caps":
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key, size=[120, 85]))  # Larger size for Caps Lock
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Variables to store the typed text and Caps Lock state
finalText = ""
capsLock = False

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for a more natural interaction
    hands, img = detector.findHands(img, flipType=False)

    # Draw all buttons
    for button in buttonList:
        # Default button color is transparent black
        color = (0, 0, 0, 100)
        if button.text == "Caps" and capsLock:
            color = (0, 255, 0, 100)  # Change color when Caps Lock is active
        img = button.draw(img, color)

    # Check for hand presence
    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            x, y = lmList[8][0:2]

            # Keyboard input
            for button in buttonList:
                bx, by = button.pos
                bw, bh = button.size

                # Check if the index finger tip is within the button's boundary
                if bx < x < bx + bw and by < y < by + bh:
                    # Highlight the button when pressed
                    cv2.rectangle(img, button.pos, (bx + bw, by + bh), (100, 100, 100),
                                  cv2.FILLED)  # Lighter gray for pressed button
                    cv2.putText(img, button.text, (bx + 20, by + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

                    # If distance between index and middle finger is small, register key press
                    x1, y1 = lmList[8][:2]
                    x2, y2 = lmList[12][:2]
                    l, _, _ = detector.findDistance((x1, y1), (x2, y2), img)
                    if l < 30:
                        if button.text == "<-":
                            if finalText:
                                finalText = finalText[:-1]
                        elif button.text == "Caps":
                            capsLock = not capsLock
                        else:
                            if capsLock:
                                finalText += button.text.upper()
                            else:
                                finalText += button.text.lower()
                        print("Pressed", button.text)
                        cv2.rectangle(img, button.pos, (bx + bw, by + bh), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (bx + 20, by + 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                        cv2.waitKey(300)  # Debounce delay

    overlay = img.copy()
    cv2.rectangle(overlay, (50, 350), (1200, 450), (0, 0, 0), cv2.FILLED)
    alpha = 0.6  # Transparency factor
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
