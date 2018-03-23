from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# First we read in the dataset
dataset = pd.read_csv("C:/Users/neon/Desktop/UTD slides/ML/carTrainData.csv")

dataset1=dataset.iloc[:,:6]
dataset2=dataset.iloc[:,6:7]

#encoding the words using numbers
le.fit(['vhigh','high','med','low','small','big','more','acc','unacc','good','vgood','2','3','4','5more'])
df=pd.DataFrame(le.transform(dataset1.V1))
df.columns=(['vect1'])
df['vect2']=le.transform(dataset1.V2)
df['vect3']=le.transform(dataset1.V3)
df['vect4']=le.transform(dataset1.V4)
df['vect5']=le.transform(dataset1.V5)
df['vect6']=le.transform(dataset1.V6)
target=le.transform(dataset2.V7)
df= df.astype(float)
# converting type to float
target=target.astype(float)

#put test_size=0 to train complete dataset
x_train,x_test,y_train,y_test=train_test_split(df,target, test_size=0.25, random_state=0)

# Here we create the Random Forest Classifier  
clf = RandomForestClassifier()
trained_model = clf.fit(x_train, y_train)


#checking predictions
train_predictions = trained_model.predict(x_test)
confusion_matrix(y_test, train_predictions)
print( " Confusion matrix ",confusion_matrix(y_test, train_predictions))
print("accuracy: ",accuracy_score(y_test, train_predictions))
print(classification_report(y_test, train_predictions))



#use below for seperate training set available
'''
filename=input('Enter the test data directory: ')
x_test = pd.read_csv(filename)
dataset3=x_test.iloc[:,:6]
dataset4=x_test.iloc[:,6:7]

df2=pd.DataFrame(le.transform(dataset3.V1))
df2.columns=(['vect1'])
df2['vect2']=le.transform(dataset3.V2)
df2['vect3']=le.transform(dataset3.V3)
df2['vect4']=le.transform(dataset3.V4)
df2['vect5']=le.transform(dataset3.V5)
df2['vect6']=le.transform(dataset3.V6)
testtarget=le.transform(dataset4.V7)
#print (df2)
df2= df2.astype(float)
# Check accuracy on test set
predictions = trained_model.predict(df2)
confusion_matrix(testtarget, predictions)

print( " Confusion matrix ",confusion_matrix(testtarget, predictions))
print("accuracy: ",accuracy_score(testtarget, predictions))
#print("classification report: ")
print(classification_report(testtarget, predictions))
'''
