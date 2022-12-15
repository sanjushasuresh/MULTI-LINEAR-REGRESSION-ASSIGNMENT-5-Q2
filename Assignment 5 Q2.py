# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:02:49 2022

@author: SANJUSHA
"""

import numpy as np
import pandas as pd
df = pd.read_csv("ToyotaCorolla.csv", encoding='latin1')
df.isnull().sum() # There are no null values
df.shape
df.duplicated()
df[df.duplicated()]
df=df.drop([113],axis=0) # Duplicate is removed
df.shape
TC=df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
TC.head()     
df1=TC.rename({'Age_08_04':'Age','Quarterly_Tax':'QT','cc':'CC'},axis=1)
df1
df1.corr()
df1.corr().to_csv("MLR.csv ")
# Age has high negative correlation with Price, i.e. -0.8762 
# X varibles are not dependent on each other


# Splitting the variables
Y = df1[["Price"]]
# X = df1[["Age"]] # Model 1
# X = df1[["Age","Weight"]] # Model 2
# X = df1[["Age","Weight","KM"]] # Model 3
# X = df1[["Age","Weight","KM","HP"]] # Model 4
X = df1[["Age","Weight","KM","HP","QT"]] # Model 5
# X = df1[["Age","Weight","KM","HP","QT","Doors"]] # Model 6
# X = df1[["Age","Weight","KM","HP","QT","Doors","CC"]] # Model 7
# X = df1[["Age","Weight","KM","HP","QT","Doors","CC","Gears"]] # Model 8


import matplotlib.pyplot as plt
plt.scatter(x = X["Age"],y = Y, color= 'black')
plt.show()
# Strong negative correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["Weight"],y = Y, color= 'black')
plt.show()
# Moderate positive correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["KM"],y = Y, color= 'black')
plt.show()
# Moderate negative correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["HP"],y = Y, color= 'black')
plt.show()
# Weak positive correlation
import matplotlib.pyplot as plt
plt.scatter(x = X["QT"],y = Y, color= 'black')
plt.show()
import matplotlib.pyplot as plt
plt.scatter(x = X["Doors"],y = Y, color= 'black')
plt.show()
import matplotlib.pyplot as plt
plt.scatter(x = X["CC"],y = Y, color= 'black')
plt.show()
import matplotlib.pyplot as plt
plt.scatter(x = X["Gears"],y = Y, color= 'black')
plt.show()
# QT, Doors, CC, Gears show very weak positive correlation


# Boxplot
df1.boxplot("Age",vert=False)
Q1=np.percentile(df1["Age"],25)
Q3=np.percentile(df1["Age"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR) # There are outliers
LW=Q1-(2.5*IQR)
df1["Age"]<LW
df1[df1["Age"]<LW].shape
df1["Age"]=np.where(df1["Age"]>UW,UW,np.where(df1["Age"]<LW,LW,df1["Age"])) 
# Replacing the outliers with UW and LW values 


df1.boxplot("KM",vert=False)
Q1=np.percentile(df1["KM"],25)
Q3=np.percentile(df1["KM"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df1["KM"]>UW
df1[df["KM"]>UW].shape
df1["KM"]=np.where(df1["KM"]>UW,UW,np.where(df1["KM"]<LW,LW,df1["KM"]))  


df1.boxplot("HP",vert=False)
Q1=np.percentile(df1["HP"],25)
Q3=np.percentile(df1["HP"],75)
IQR=Q3-Q1
UW=Q3+(1.5*IQR)
LW=Q1-(1.5*IQR)
df1["HP"]>UW
df1[df1["HP"]>UW].shape
df1["HP"]=np.where(df1["HP"]>UW,UW,np.where(df1["HP"]<LW,LW,df1["HP"]))


df1.boxplot("CC",vert=False)
Q1=np.percentile(df1["CC"],25)
Q3=np.percentile(df1["CC"],75)
IQR=Q3-Q1
UW=Q3+(2.0*IQR)
LW=Q1-(2.0*IQR)
df1["CC"]>UW
df1[df1["CC"]>UW].shape
df1["CC"]=np.where(df1["CC"]>UW,UW,np.where(df1["CC"]<LW,LW,df1["CC"]))
# Outliers are 80

df1.boxplot("Doors",vert=False)

df1.boxplot("Gears",vert=False)
Q1=np.percentile(df1["Gears"],25)
Q3=np.percentile(df1["Gears"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df1["Gears"]>UW
df1[df1["Gears"]>UW].shape
df1["Gears"]<LW
df1[df1["Gears"]<LW].shape
df1["Gears"]=np.where(df1["Gears"]>UW,UW,np.where(df1["Gears"]<LW,LW,df1["Gears"]))


df1.boxplot("QT",vert=False)
Q1=np.percentile(df1["QT"],25)
Q3=np.percentile(df1["QT"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df1["QT"]>UW
df1[df1["QT"]>UW].shape
df1["QT"]<LW
df1[df1["QT"]<LW].shape
df1["QT"]=np.where(df1["QT"]>UW,UW,np.where(df1["QT"]<LW,LW,df1["QT"]))


df1.boxplot("Weight",vert=False)
Q1=np.percentile(df1["Weight"],25)
Q3=np.percentile(df1["Weight"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df1["Weight"]>UW
df1[df1["Weight"]>UW].shape
df1["Weight"]<LW
df1[df1["Weight"]<LW].shape
df1["Weight"]=np.where(df1["Weight"]>UW,UW,np.where(df1["Weight"]<LW,LW,df1["Weight"]))

df1.duplicated()
df1[df1.duplicated()]


# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

# Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)
Y_predtrain=LR.predict(X_train)
Y_predtest=LR.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score
mse1 = mean_squared_error(Y_train,Y_predtrain) 
# MSE1 = 1811733.3395
mse2 = mean_squared_error(Y_test,Y_predtest)
# MSE2 = 1794484.8539

rmse1 = np.sqrt(mse1).round(2)
# RMSE1 = 1346.01
rmse2 = np.sqrt(mse2).round(2)
# RMSE2 = 1339.58

r2_score1 = r2_score(Y_train,Y_predtrain)
# r2_score1 = 0.8611748
r2_score2 = r2_score(Y_test,Y_predtest)
# r2_score2 = 0.8611689


# Model Validation - Multicollinearity, KFold
# pip install statsmodels
import statsmodels.api as sma
X_new = sma.add_constant(X)
lm = sma.OLS(Y,X_new).fit()
lm.summary()
# p value of Doors, CC, Gears is >0.05. Therefore it has multicollinearity issues

from sklearn.model_selection import KFold, cross_val_score
k=3
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(LR, X_train, Y_train, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)


# Model Deletion - Cooks Distance
# Suppress scientific notation
import numpy as np
np.set_printoptions(suppress=True)
# Create instance of influence
influence = lm.get_influence()
# Obtain Cook's distance for each observation
cooks = influence.cooks_distance
# Display Cook's distances
print(cooks)

import matplotlib.pyplot as plt
plt.scatter(df1.Age, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.Weight, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.KM, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.HP, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.QT, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.Doors, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.CC, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(df1.Gears, cooks[0])
plt.xlabel('X')
plt.ylabel('Cooks Distance')
plt.show()


# Inference : Here Model 5 where Y=df1[["Price"]] and X = df1[["Age","Weight","KM","HP","QT"]]
# is selected, since its r2 for train and test are 0.86117 and 0.86116 respectively, P value of   
# variables is <0.05 and the model has no collinearity issues. So it is better than other models  
# also for less expense and more profit.
 
