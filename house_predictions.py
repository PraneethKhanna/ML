from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import pandas as np
from sklearn import preprocessing
from sklearn.metrics import r2_score
le = preprocessing.LabelEncoder()

#The dataset has 81 columns with 1460rows
house_raw = pd.read_csv("C:/Users/pxb171530/Downloads/DEMO/House_train.csv", sep="," ) 
print(house_raw.describe([.1, .5, .8]))

# most of the places doesn't have pools,Del columns POOLQC,POOLAREA
house = house_raw.drop(columns = ["Id","Alley","PoolQC","Fence","MiscVal","MiscFeature","PoolArea"])

#80% of the data is of same type thus dropping columns which are not effective for model prediction
house = house.drop(columns = ["BsmtFinSF2","BsmtFinType2","3SsnPorch","ScreenPorch","BsmtHalfBath","LowQualFinSF",
                              "KitchenAbvGr","SaleType","FireplaceQu","SaleCondition","EnclosedPorch","Functional","GarageCond"])

house=house.dropna(subset=["LotFrontage"])
#agricultural, commercial, Residential
house["MSZoning"]=le.fit_transform(house["MSZoning"])

# streer gravel or paved
house["Street"]=le.fit_transform(house["Street"])

#rgular , irregular, moderalety or slightly irregular
house["LotShape"]=le.fit_transform(house["LotShape"])

#banked , flat sized , hillside
house["LandContour"]=le.fit_transform(house["LandContour"])

#Utilities - electricity, water, gas 
house["Utilities"]=le.fit_transform(house["Utilities"])

#Lot Config  inside Lot , corner Lot
house["LotConfig"]=le.fit_transform(house["LotConfig"])

#gentle slope, moderate , severe
house["LandSlope"]=le.fit_transform(house["LandSlope"])

#Neighborhood 
house["Neighborhood"]=le.fit_transform(house["Neighborhood"])

#Condition1 Proximity to various conditions
house["Condition1"]=le.fit_transform(house["Condition1"])

#Condition2: Proximity to various conditions (if more than one is present)
house["Condition2"]=le.fit_transform(house["Condition2"])

#BldgType: Type of dwelling single family, duplex
house["BldgType"]=le.fit_transform(house["BldgType"])

#HouseStyle one story/two story
house["HouseStyle"]=le.fit_transform(house["HouseStyle"])

#RoofStyle Type of roof
house["RoofStyle"]=le.fit_transform(house["RoofStyle"])

#Roof material
house["RoofMatl"]=le.fit_transform(house["RoofMatl"])

#Exterior1st: Exterior covering on house
house["Exterior1st"]=le.fit_transform(house["Exterior1st"])

#Exterior1st: Exterior covering on house,Brick Face,Cement Board,Plywood
house["MSZoning"]=le.fit_transform(house["MSZoning"])

#Exterior2nd: Exterior covering on house (if more than one material)
house["Exterior2nd"]=le.fit_transform(house["Exterior2nd"])

#MasVnrType: Masonry veneer type
house=house.dropna(subset=["MasVnrType"])
house["MasVnrType"]=le.fit_transform(house["MasVnrType"])

#ExterQual: Evaluates the quality of the material on the exterior
house["ExterQual"]=le.fit_transform(house["ExterQual"])

#ExterCond: Evaluates the present condition of the material on the exterior
house["ExterCond"]=le.fit_transform(house["ExterCond"])

#Foundation: Type of foundation
house["Foundation"]=le.fit_transform(house["Foundation"])

house=house.dropna(subset=["BsmtExposure"])
#BsmtQual: Evaluates the height of the basement,Good (90-99 inches)
house["BsmtQual"]=le.fit_transform(house["BsmtQual"])

#BsmtCond: Evaluates the general condition of the basement
house["BsmtCond"]=le.fit_transform(house["BsmtCond"])

#BsmtExposure: Refers to walkout or garden level walls,Average Exposure,Mimimum Exposure
house["BsmtExposure"]=le.fit_transform(house["BsmtExposure"])

#BsmtFinType1: Rating of basement finished area
house["BsmtFinType1"]=le.fit_transform(house["BsmtFinType1"])

#Heating: Type of heating
house["Heating"]=le.fit_transform(house["Heating"])

#HeatingQC: Heating quality and condition
house["HeatingQC"]=le.fit_transform(house["HeatingQC"])

#Exterior1st: Exterior covering on house
house["Exterior1st"]=le.fit_transform(house["Exterior1st"])
#CentralAir: Central air conditioning
house["CentralAir"]=le.fit_transform(house["CentralAir"])

house=house.dropna(subset=["Electrical"])
#Electrical: Electrical system,fuse 60 AMP Fuse Box and mostly Romex wiring (Fair)
house["Electrical"]=le.fit_transform(house["Electrical"])

#KitchenQual: Kitchen quality
house["KitchenQual"]=le.fit_transform(house["KitchenQual"])

#Functional: Home functionality (Assume typical unless deductions are warranted)
house["MSZoning"]=le.fit_transform(house["MSZoning"])

house=house.dropna(subset=["GarageType"])
#GarageType: Garage location
house["GarageType"]=le.fit_transform(house["GarageType"])

#GarageFinish: Interior finish of the garage
house["GarageFinish"]=le.fit_transform(house["GarageFinish"])

#GarageQual: Garage quality
house["GarageQual"]=le.fit_transform(house["GarageQual"])

#PavedDrive: Paved driveway
house["PavedDrive"]=le.fit_transform(house["PavedDrive"])

print(house.describe([.1, .5, .8]))

house = house.drop(columns = ["MSZoning","Street","LandContour","Utilities","LandSlope",
                              "GarageQual","PavedDrive","Condition1","Condition2","BldgType",
                              "RoofMatl","BsmtCond","Heating","CentralAir","Electrical"])

#random_state=3,2,1
x_train,x_test,y_train,y_test=train_test_split(house.iloc[:,:-1],house.iloc[:,-1], test_size=0.2, random_state=1)
#Scaling the input features
sc_p = preprocessing.StandardScaler()
sc_q = preprocessing.StandardScaler()

x_train = sc_p.fit_transform(x_train)
x_test = sc_p.transform(x_test)

y_train = y_train.astype(float)
y_test = y_test.astype(float)
y_train = (sc_q.fit_transform(y_train.values.reshape(-1, 1)))
y_test = (sc_q.transform(y_test.values.reshape(-1, 1)))


from sklearn.decomposition import PCA
sc2 = preprocessing.StandardScaler()
house_scaled2 = sc2.fit_transform(house)
pca_2 = PCA(n_components = 2)
x_pca = pca_2.fit_transform(house_scaled2[:,:-1])
variance_features2 = pca_2.explained_variance_ratio_


from sklearn.neighbors import KNeighborsRegressor
K_model = KNeighborsRegressor()
K_model.fit(x_train,y_train)
predictions = K_model.predict(x_test)
MSE =  mean_squared_error(y_test, predictions)
print()
print("Mean squared error and r2 score of predicted values are:")
print(MSE)
print(r2_score(y_test, predictions))
print()

y_test_inv_scaled = sc_q.inverse_transform(y_test)
result_inv_scaled = sc_q.inverse_transform(predictions)

from sklearn.model_selection import cross_val_predict,KFold
cv = KFold(n_splits=10)
sc = preprocessing.StandardScaler()
house_scaled = sc.fit_transform(house)
results = cross_val_predict(estimator = K_model, X = house_scaled[:,:-1], y = house_scaled[:,-1], cv = cv)
MSE =  mean_squared_error(house_scaled[:,-1], results)
print("Mean squared error and r2 score of cross_validated prediction values are:")
print(MSE)
print(r2_score(house_scaled[:,-1], results))


import matplotlib.pyplot as plt
plt.plot(x_pca, results, 'ro')
plt.show()

plt.plot(x_pca, house_scaled[:,-1], 'ro')
plt.show()



