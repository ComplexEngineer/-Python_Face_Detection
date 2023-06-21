import numpy as np
import cv2


print("Start...")
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

#esp_ip = "192.168.43.219"
#capture = cv2.VideoCapture(f"rtsp://10.2.4.39:554/stream/main")

while(True):
    ret, image = capture.read()
    image=cv2.resize(image,(160,100),interpolation = cv2.INTER_AREA)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayImage)

    if len(faces) == 0:
        image=cv2.resize(image,(660,400),interpolation = cv2.INTER_AREA)
        cv2.rectangle(image, ((0, image.shape[0] - 25)), (60, image.shape[0]), (255, 255, 0), -1)
        cv2.putText(image, "LEN: 0" , (0, image.shape[0] - 10),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Face Detect', image)

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        image=cv2.resize(image,(660,400),interpolation = cv2.INTER_AREA)
        cv2.rectangle(image, ((0, image.shape[0] - 25)), (60, image.shape[0]), (255, 255, 0), -1)     
        cv2.putText(image, "LEN: " + str(faces.shape[0]), (0, image.shape[0] - 10),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Face Detect', image)
        
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break

capture.release()
cv2.destroyAllWindows()
