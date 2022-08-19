import urllib.request
import cv2
import os
import numpy as np
import imutils

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  
sub_data = 'Brij'
url='http://192.168.1.11:8080/shot.jpg'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
    
(width, height) = (130, 100)   

face_cascade = cv2.CascadeClassifier(haar_file)
count = 1

while count < 31:
    print(count)
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width=450)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
    count += 1
	
    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyAllWindows()
