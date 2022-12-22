import cv2
import numpy as np 
import csv
import time

from sklearn import svm
import pandas as pd

cap = cv2.VideoCapture (1)
img = cap.set (cv2.CAP_PROP_FRAME_WIDTH, 480)
img = cap.set (cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Database: Gerbang Logika AND
#Membaca data dari fie
FileDB = 'database warna.txt'
Database = pd.read_csv(FileDB, sep=", ", header=0)
print (Database)

#x= Data, y = Target
x = Database [[u'B', u'G', u'R']]
y = Database.Target

clf = svm.SVC ()
clf.fit(x, y)

fpsLimit = 1
startTime = time.time()

while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)
    for x in range (330,340,1):
        for y in range (220,260,1):
            color = img [x,y]
            colorB = img [y,x,0]
            colorG = img [y,x,1]
            colorR = img [y,x,2]

print ('B G R = ', color)
cv2.imshow("Color Tracking", img)
if clf.predict([color]) == 'Abu-abu':
    print ("Abu-abu")
elif clf.predict([color])=='Hitam' :
    print ("Hitam") 
elif clf.predict ( [color]) == 'Coklat':
    print ("Coklat")
elif clf.predict ( [color]) == 'Milo': 
    print ("Milo")

k = cv2.waitKey(30) & 0xff
if k == 27:
     break

cap.release()
cv2.destroyAllWindows()
