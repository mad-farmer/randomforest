#PART 1: PREDICTING PETROL CONSUMPTION
import pandas as pd

#importing csv file
df=pd.read_csv("petrol_consumption.csv")
df.head()

#preparing data for training
y=df.iloc[:,4].values.reshape(-1,1) #Petrol_Consumption
x=df.iloc[:,0:4].values #other columns

#creating train and test values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=20)

#Scaling features
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Prediction
from sklearn.ensemble import RandomForestRegressor
rg=RandomForestRegressor(n_estimators=200,random_state=0) 
#RandomForestRegressor model
rg.fit(x_train,y_train) 
#Fitting RandomForestRegressor to train values
y_pred=rg.predict(x_test)
 #Prediction y_test values(Petrol_Consumption)








#%%














