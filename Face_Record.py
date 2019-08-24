import numpy as np
import os
import pandas as pd
import cv2
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
print("Enter Your Name : ")
name=input()

face_list=[]
count=50
while(count):
    areas=[]
    ret, image=cap.read()
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces=classifier.detectMultiScale(gray)
    for face in faces :
        x, y, w ,h =face
        area= w*h
        areas.append((area,face))
    areas=sorted(areas,reverse=True)
    if len(areas)>0:
        face=areas[0][1]
        x, y, w, h=face
        face_img=gray[y:y+h,x:x+w]
        face_img=cv2.resize(face_img,(100,100))
        cv2.imshow("FACE",face_img)
        face_list.append(face_img.flatten())
        count-=1
        print("{} Record Entered".format(50-count))
    if cv2.waitKey(1) > 30:
            break

face_list = np.array(face_list)
print(face_list.shape)
name_list = np.full((len(face_list),1),name)
Final = np.hstack([name_list,face_list])
if os.path.exists("Face_Record.npy"):
    old_records=np.load("Face_Record.npy")
    Final=np.vstack([old_records,Final])
np.save("Face_Datanpy",Final)
cap.release()
cv2.destroyAllWindows()




