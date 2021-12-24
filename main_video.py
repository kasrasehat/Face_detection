
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        gray_roi = gray[y:y+h ,x:x+w]
        frame_roi = frame[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

video_capture = cv2.VideoCapture(0)
while True:
    _ ,frame = video_capture.read()
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
    canvas = detect(gray, frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyWindow()
