# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detector(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        gray_roi = gray[y:y+h ,x:x+w]
        frame_roi = frame[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        for (xa,ya,wa,ha) in eyes:
            cv2.rectangle(frame_roi, (xa, ya), (xa+wa, ya+ha), (255, 0, 0), 2)

    return frame



if __name__ == '__main__':
    image= cv2.imread('photo_2021-10-15_00-38-36.jpg')
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    a = detector(image1 , image)
    cv2.imshow('image', a)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
