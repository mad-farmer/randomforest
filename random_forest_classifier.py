#%% PART 2 CLASSIFICATION 
#BANK CURRENCY NOTE IS AUTHENTIC OR NOT BASED
import pandas as pd

#importing csv file
df2=pd.read_csv("bill_authentication.csv")

#x and y values
x=df2.iloc[:,0:4].values
y=df2.iloc[:,4].values.reshape(-1,1) #classes

#creating train and test values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=0)

#Scaling features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Classificaiton
from sklearn.ensemble import RandomForestClassifier 
regressor = RandomForestClassifier(
        n_estimators=20, random_state=0)
#RandomForestClassifier model
regressor.fit(x_train, y_train)
#Fitting RandomForestClassifier to train values
y_pred = regressor.predict(x_test)
#Classification y_Test values 

#Accuracy and Evaluating 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))





























