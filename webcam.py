import cv2
import glob

i = len(glob.glob('facial_rec\*.jpg'))
cam = cv2.VideoCapture(0) # device 0. If not work, try with 1 or 2


if not cam.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,2)

    cv2.imshow('My App!', frame)

    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
    elif key==ord("s"):
        cv2.imwrite(f'facial_rec\{i:03d}.jpg',frame)
        i+=1

cam.release()
cv2.destroyAllWindows()