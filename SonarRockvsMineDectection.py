# Dependencies
from re import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Data Processing
# As we don't have any data here, so we have put header to be none.
sonar_datasets = pd.read_csv('SonarDatasets.csv',header=None)
print(sonar_datasets.head()) #Display first 5 Rows.
print(sonar_datasets.shape) #Display number of rows and columns in tupple
print(sonar_datasets.describe()) #Display Some statistical measures of the data.
print(sonar_datasets[60].value_counts()) #Displays number of rock and mine data present.
print(sonar_datasets.groupby(60).mean()) #Display mean value of all Mine and Rocks present.




# Splitting of data and labels.
x = sonar_datasets.drop(columns=60,axis=1)
y= sonar_datasets[60]
print(x)
print(y)




#Splitting in train and test datas.
# Stratify= y:spilitting data based on Rock and Mine
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1) 
print(x.shape,x_train.shape,x_test.shape)




# Model Training using Logistic Regression.
model = LogisticRegression()

# Traing the Logistic Regression model with training data.
model.fit(x_train,y_train)


# Model Evaluation.
# Acurracy on training data.
x_train_prediction = model.predict(x_train)
train_data_acurracy = accuracy_score(x_train_prediction , y_train)
print("Accuracy on training data: ",train_data_acurracy)

# Acurracy on training data.
x_test_prediction = model.predict(x_test)
test_data_acurracy = accuracy_score(x_test_prediction , y_test)
print("Accuracy on training data: ",test_data_acurracy)



# Making a Sonar Prediction System
input_data = (0.0114,0.0222,0.0269,0.0384,0.1217,0.2062,0.1489,0.0929,0.1350,0.1799,0.2486,0.2973,0.3672,0.4394,0.5258,0.6755,0.7402,0.8284,0.9033,0.9584,1.0000,0.9982,0.8899,0.7493,0.6367,0.6744,0.7207,0.6821,0.5512,0.4789,0.3924,0.2533,0.1089,0.1390,0.2551,0.3301,0.2818,0.2142,0.2266,0.2142,0.2354,0.2871,0.2596,0.1925,0.1256,0.1003,0.0951,0.1210,0.0728,0.0174,0.0213,0.0269,0.0152,0.0257,0.0097,0.0041,0.0050,0.0145,0.0103,0.0025)

# changing the input data to a numpy array as processing of numpy array is faster.
numpy_input_data = np.asarray(input_data)
# Reshape np array as we predict for 1 instance.
input_data_reshape  = numpy_input_data.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)

if(prediction[0]=='M'):
    print("The Object is a Mine")
else:
    print("The Object is a Rock")
