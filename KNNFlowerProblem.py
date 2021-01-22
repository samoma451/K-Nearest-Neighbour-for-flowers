import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv("iris.data")

#Lengths and widths are independent, species dependent
x = iris.iloc[:,0:4]
y = iris.iloc[:,-1]

#iris.shape to get no. of rows

rowRoot = int(round(np.sqrt(150)))
knn = KNeighborsClassifier(n_neighbors=rowRoot)
knn.fit(x,y)

yPredict = knn.predict(x)

yPrediction = pd.DataFrame(data = [yPredict, y.values])#please note the difference between yPredict and yPrediction
print(yPrediction.transpose()+"\n")

#We can also insert values to get a prediction like this
knn.predict([[1,2,3,4]])#where the inputted arguments correspond to the sepalLength,sepalWidth,PetalLength,PetalWidth

#now to train the data to handle unseen data
from sklearn.model_selection import train_test_split as tts
xTrain,xTest,yTrain,yTest = tts(x,y,test_size=0.3,random_state=2)#test_size gives the proportion of the origanal set to be used
#& random_state for reproducability

knn.fit(xTrain,yTrain)
yTestPredict = knn.predict(xTest)
print(yTestPredict+"\n")

#now to check against our correct data
predictionOutput = pd.DataFrame([yTestPredict, yTest.values])
print(predictionOutput.transpose()+"\n")

