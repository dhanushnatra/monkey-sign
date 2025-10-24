from cvzone import HandTrackingModule,FaceDetectionModule
import cv2
from enum import Enum

class MonkeyState(Enum):
    Idle="idle"
    THINK="think"
    NO_THINK="noThink"


faceMesh = FaceDetectionModule.FaceDetector(minDetectionCon=0.8)

Hands = HandTrackingModule.HandDetector(maxHands=1)



def monkey_state(frame)->MonkeyState|None:
    _,faceBoxes= faceMesh.findFaces(frame,draw=False)
    hands,_ = Hands.findHands(frame,draw=False)
    if len(faceBoxes)>0 and len(hands)==0:
        return MonkeyState.Idle
    elif len(faceBoxes)>0 and len(hands)>0:
        h_x,h_y,h_z=hands[0]["lmList"][8]
        x,y,w,h=faceBoxes[0]["bbox"]
        if h_x>x and h_y>y and h_x<(x+w) and h_y<(y+h) :
            return MonkeyState.THINK
        else:
            return MonkeyState.NO_THINK
    else:
        return None