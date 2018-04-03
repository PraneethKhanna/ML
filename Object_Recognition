import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from PIL import Image
import glob
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#use sift features from images
sift= cv2.xfeatures2d.SIFT_create()

BOW = cv2.BOWKMeansTrainer(40)
label_list=[]
image_list ={1: [], 2: []}


#appending sift features into bag of words
for filename in glob.glob('file name of training images/*.jpg'):
    im=Image.open(filename)
    pic=cv2.imread(filename)
    label=float(filename.split("\\")[1][0])
    image_list[label].append(im)
    gim=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    key,vec= sift.detectAndCompute(gim, None)
    BOW.add(vec)
    label_list.append(label)
dictionary = BOW.cluster()

bowDiction = cv2.BOWImgDescriptorExtractor(sift,cv2.BFMatcher(cv2.NORM_L2))

bowDiction.setVocabulary(dictionary)

train_desc = []
for filename in glob.glob('file name of training images/*.jpg'): 
    im=Image.open(filename)
    pic=cv2.imread(filename)
    gim=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    train_desc.extend(bowDiction.compute(gim, sift.detect(gim)))
    
'''Using opencv svm model
svm = cv2.ml.SVM_create()
svm.train(np.array(train_desc), cv2.ml.ROW_SAMPLE,np.array(label_list))
'''
data=pd.DataFrame()
data['label']=label_list       

svc=SVC(gamma=0.01,C=10,kernel='sigmoid')
svc.fit(train_desc,data['label'])

clf = RandomForestClassifier(max_depth=100, random_state=0,n_estimators =30)
clf.fit(train_desc, data['label'])
#To measure accuracy of  the train model
'''
result=[]
result2=[]
for filename in glob.glob('filename of training images/*.jpg'): 
    im=Image.open(filename)
    test=cv2.imread(filename)
    test=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    testdata=bowDiction.compute(test, sift.detect(test))
    result.append(svc.predict(testdata))
    result2.append(clf.predict(testdata))
accuracy_score(data['label'], result)
accuracy_score(data['label'], result2) 


#you can create a pickle file of the trained model
#from sklearn.externals import joblib
#joblib.dump(clf, 'filename.pkl')
   
'''

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        
        gr=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        testdata=bowDiction.compute(gr, sift.detect(gr))
        result= clf.predict(testdata)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,400)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(frame,str(result), (10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        
        cv2.imshow("Detection", frame)
        print(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

