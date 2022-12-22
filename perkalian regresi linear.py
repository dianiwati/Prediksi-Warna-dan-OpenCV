import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#database
# x = Data, y = Target
#x = [[1], [3], [5], [7], [9]]
#y = [2, 4, 10, 14, 18]
FileDB = 'kalian.txt'
Database = pd.read_csv(FileDB, sep=" ", header = 0)

print ("---------------------------------")
print (Database)
#x = Data, y = Target
x = Database[[u'x']] #ciri1, ciri2, dst
y = Database.Target

regr = LinearRegression ().fit (x, y)
regr.score (x, y)

#Data uji
predict = np.array([[1100 ]])

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict) 
print ("Output = ", regr.predict (predict))
