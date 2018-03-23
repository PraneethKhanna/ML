
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict,KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

BikeData = pd.read_csv('C:/Users/neon/Desktop/UTD slides/ML/bikeRentalHourlyTrain.csv')
Bike_testData= pd.read_csv('C:/Users/neon/Desktop/UTD slides/ML/bikeRentalHourlyTest.csv')

Bike_testData["dteday"] = Bike_testData["dteday"] .apply(lambda x: re.sub('-', '', x))
BikeData["dteday"]  = BikeData["dteday"] .apply(lambda x: re.sub('-', '', x))

BikeData=BikeData.iloc[:1000]
Bike_testData=Bike_testData[:100]
Bike_test_Data=Bike_testData.iloc[:,3:15]
Bike_train_Data=BikeData.iloc[:,3:15]


scaler = StandardScaler()
Bike_train_Data= scaler.fit_transform(Bike_train_Data)
Bike_test_Data= scaler.transform(Bike_test_Data)
#put test_size=0 to train whole data
#x_train,x_test,y_train,y_test=train_test_split(BikeData,Bike_train_Data["cnt"], test_size=0, random_state=0)

nn_classifier = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=100,activation='identity',random_state=0)
nn_model = nn_classifier.fit(Bike_train_Data , BikeData["cnt"])

knn_classifier = KNeighborsRegressor(n_neighbors=7,p=1)
knn_model = knn_classifier.fit(Bike_train_Data , BikeData["cnt"])

regr_classifier = linear_model.Lasso(alpha=1.0,fit_intercept=True, normalize=True,random_state=0)
regr_model=regr_classifier.fit(Bike_train_Data, BikeData["cnt"])

ridge_classifier = Ridge(alpha=1.0,normalize=True,random_state=0)
ridge_model=ridge_classifier.fit(Bike_train_Data, BikeData["cnt"])

cv = KFold(n_splits=5, shuffle=True)

cv_nn_prediction = cross_val_predict(nn_classifier, Bike_train_Data, BikeData["cnt"], cv=cv)
cv_knn_prediction = cross_val_predict(knn_classifier, Bike_train_Data, BikeData["cnt"], cv=cv)
cv_regr_prediction = cross_val_predict(regr_classifier, Bike_train_Data, BikeData["cnt"], cv=cv)
cv_ridge_prediction = cross_val_predict(ridge_classifier, Bike_train_Data, BikeData["cnt"], cv=cv)


Testcv_nn_prediction = cross_val_predict(nn_classifier, Bike_test_Data, Bike_testData["cnt"], cv=cv)
Testcv_knn_prediction = cross_val_predict(knn_classifier, Bike_test_Data, Bike_testData["cnt"], cv=cv)
Testcv_regr_prediction = cross_val_predict(regr_classifier, Bike_test_Data, Bike_testData["cnt"], cv=cv)
Testcv_ridge_prediction = cross_val_predict(ridge_classifier, Bike_test_Data, Bike_testData["cnt"], cv=cv)

'''
nn_train_prediction = nn_model.predict(Bike_test_Data)
knn_train_prediction = knn_model.predict(Bike_test_Data)
regr_train_prediction = regr_model.predict(Bike_test_Data)
ridge_train_prediction = ridge_model.predict(Bike_test_Data)
'''

print("MSE for train")
print(mean_squared_error(BikeData["cnt"], cv_nn_prediction))
print(mean_squared_error(BikeData["cnt"], cv_knn_prediction))
print(mean_squared_error(BikeData["cnt"], cv_regr_prediction))
print(mean_squared_error(BikeData["cnt"], cv_ridge_prediction))

print("MSE for test")
print(mean_squared_error(Bike_testData["cnt"], Testcv_nn_prediction))
print(mean_squared_error(Bike_testData["cnt"], Testcv_knn_prediction))
print(mean_squared_error(Bike_testData["cnt"], Testcv_regr_prediction))
print(mean_squared_error(Bike_testData["cnt"], Testcv_ridge_prediction))

