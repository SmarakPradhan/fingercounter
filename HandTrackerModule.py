import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, detect_mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.detect_mode = detect_mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.detect_mode, self.maxHands,
                                        self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True,id_lm = 0):
        id = 0
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    #for any particle id, draw a circle and fill it to highlight it
                    if id == id_lm:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    ptime = 0
    ctime = 0
    cam = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,id_lm=0)
        if len(lmList) != 0:
            print(lmList[5])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (255, 0, 255), 1                                                         )

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()