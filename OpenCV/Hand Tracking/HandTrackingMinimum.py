import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks) 

    # lm.x and lm.y gives a ratio, this needs to be multiplied by image height and width to get a pixel position

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # All 21 landmark points
                # print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED) # Print a purple circle onto wrist (landmark 0)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # We draw into BGR img because that is what we are displaying

    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)