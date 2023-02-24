## Importing the Dependecies 
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import binarize

from sklearn.metrics import mean_squared_error



## Data Colection and Preprocessing
data = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data.isnull().sum()

data.head()



## Binarization of Data
y = data['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

x = data.drop(["quality"],axis=1)



## Splitting the Data 
xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.2, random_state=10, stratify=y)



## Training The Model

model =RandomForestClassifier()

model.fit(xtn,ytn)



## Model Evaluation Through Accuracy Score
y_pred = model.predict(xtt)

ascore = accuracy_score(ytt,y_pred)

ascore

rmse = mean_squared_error(ytt,y_pred, squared=False)

print(f"The accuracy score of the model is {ascore} ")
print(f"The RMSE of the model is {rmse} ")

##prediction
input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)


if (prediction[0]==1):
  print('The Quality of the Wine is Good.')
else:
  print('The Quality of the Wine is Bad.')
