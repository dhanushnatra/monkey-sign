import cv2
from Detect import monkey_state,MonkeyState


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size= (640,480)


think = cv2.imread("images/Think.jpg")
think= cv2.resize(think,size)
noThink = cv2.imread("images/noThink.jpg")
noThink= cv2.resize(noThink,size)
neutral = cv2.imread("images/neutral.jpg")
neutral= cv2.resize(neutral,size)
noFace = cv2.imread("images/noFace.jpg")
noFace= cv2.resize(noFace,size)

webcam = cv2.VideoCapture(0)

def get_monkey(frame):
    state = monkey_state(frame)
    if state == MonkeyState.THINK:
        return think
    elif state == MonkeyState.NO_THINK:
        return noThink
    elif state == MonkeyState.Idle:
        return neutral
    else:
        return noFace





while True:
    ret, frame = webcam.read()
    frame = cv2.resize(frame, size)
    if not ret:
        break
    out = get_monkey(frame)
    cv2.imshow("Output", out)
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()
