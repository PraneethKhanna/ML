import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import pandas as pd
from PIL import Image
import glob
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

#sift= cv2.xfeatures2d.SIFT_create()
sift=cv2.xfeatures2d.SURF_create()


descriptors_unclustered = []

BOW = cv2.BOWKMeansTrainer(40)
label_list=[]
image_list ={1: [], 2: []}

for filename in glob.glob('C:/Users/pxb171530/Downloads/DEMO/Mytrain/*.jpg'):
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
for filename in glob.glob('C:/Users/pxb171530/Downloads/DEMO/Mytrain/*.jpg'): 
    im=Image.open(filename)
    pic=cv2.imread(filename)
    gim=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    train_desc.extend(bowDiction.compute(gim, sift.detect(gim)))

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
for filename in glob.glob('C:/Users/pxb171530/Downloads/Mytrain/*.jpg'): 
    im=Image.open(filename)
    test=cv2.imread(filename)
    test=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    testdata=bowDiction.compute(test, sift.detect(test))
    result.append(svc.predict(testdata))
    result2.append(clf.predict(testdata))
accuracy_score(data['label'], result)
accuracy_score(data['label'], result2) 
#from sklearn.externals import joblib
svc= joblib.load('C:/Users/pxb171530/Downloads/MyTrainModel (1).pkl')
   
'''
zz=3
#joblib.dump(clf,'C:/Users/pxb171530/Downloads/clf.pkl')
cap = cv2.VideoCapture('C:/Users/pxb171530/Downloads/DEMO/t9.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        
        gr=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        z=[]
        y=[]
        windowsize_r=int(frame.shape[0]/zz)
        windowsize_c=int(frame.shape[1]/zz)
        for r in range(0,zz):   
            for c in range(0,zz):      
                a=r*windowsize_r
                b=c*windowsize_c
                window = gr[a:a+windowsize_r,b:b+windowsize_c]
                
                t1=bowDiction.compute(window, sift.detect(window))
                t1=pd.DataFrame(t1)
                #t1.fillna(1)
                '''
                if(np.isnan(t1).any()):
                    print("gotcha")
                    break
                    '''
                   
                z.append(int(clf.predict(t1)))
                y.append(clf.predict_proba(t1).ravel().tolist())
                
        
        box=pd.DataFrame(np.array(y))
       
        result="NO temoc"
        if(len(box)==0):
            continue
        if(max(box[0])>=0.6):   
            box_id=box[0].idxmax()        
            print(box_id)
            p=int(box_id/zz)
            q=int(box_id%zz)
            cv2.rectangle(frame,(q*windowsize_c,p*windowsize_r),(q*windowsize_c+windowsize_c,p*windowsize_r+windowsize_r),(0,255,0),3)
            result="Temoc"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,400)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(frame,str(result), (10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
       
        cv2.imshow("Detection", frame)
#        print(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

'''

imgC=cv2.imread("C:/Users/pxb171530/Downloads/1(1).jpg")
img=cv2.cvtColor(imgC,cv2.COLOR_BGR2GRAY)
z=[]
y=[]
windowsize_r=int(imgC.shape[0]/2)
windowsize_c=int(imgC.shape[1]/2)
for r in range(0,2):   
    for c in range(0,2):      
#        cv2.line(img, (int(img.shape[1]/c), 0), (int(img.shape[1]/c), img.shape[0]), (255, 0, 0), 5, 5)
#        cv2.line(img, (0, int(img.shape[0]/r)), (img.shape[1],int(img.shape[0]/r)), (255, 0, 0), 5, 5)
        a=r*windowsize_r
        b=c*windowsize_c
        window = img[a:a+windowsize_r,b:b+windowsize_c]
        t1=bowDiction.compute(window, sift.detect(window))
        z.append(int(clf.predict(t1)))
        y.append(clf.predict_proba(t1).ravel().tolist())
        
box=pd.DataFrame(np.array(y))
if(max(box[0])>=0.6):   
    box_id=box[0].idxmax()        
    print(box_id)
    p=int(box_id/2)
    q=int(box_id%2)
    cv2.rectangle(imgC,(q*windowsize_c,p*windowsize_r),(q*windowsize_c+windowsize_c,p*windowsize_r+windowsize_r),(0,255,0),3)
cv2.imshow("Detection", imgC)      
cv2.waitKey(0)  
cap.release()
cv2.destroyAllWindows()

'''

k=np.array([[1],[2],[3],[4],[5]])
k=np.nan * np.ones(shape=(1,5))

k[0][0]=1
k[0][1]=1
k[0][2]=1
k[0][3]=1
k=pd.DataFrame(k)
if k.isnull().values.any():
    print(1)
    
                    










