import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyodbc
path='face_recognition\images'
images=[]
personName=[]
myList=os.listdir(path)


for current in myList:
    current_img=cv2.imread(f'{path}/{current}')
    images.append(current_img)
    personName.append(os.path.splitext(current)[0])
    


def faceEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown=faceEncodings(images)
print("All encodings complete\n")

def markAttendance(name):
    with open('face_recognition\markattendence.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
            
        if name not in nameList:
            time_now=datetime.now()
            tstr=time_now.strftime('%H:%M:%S')
            dstr=time_now.strftime('%d:%m:%Y')
            f.writelines(f'{name},{tstr},{dstr}')
    

cap=cv2.VideoCapture(0)
while True:
    ret,frame =cap.read()
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    
    facesCurrentFrame=face_recognition.face_locations(faces)
    encodesCurrentFrame=face_recognition.face_encodings(faces,facesCurrentFrame)
    
    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=personName[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1=4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,0),2)
            markAttendance(name)
            
    

    cv2.imshow("Camera",frame)
    if cv2.waitKey(1)==13:
        break
    
cap.release()
cv2.destroyAllWindows()
