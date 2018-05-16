
import numpy as np
import cv2

from keras.models import load_model
model = load_model('C:/Users/pxb171530/Downloads/DEMO/2layered_pickel.pkl')


screen = cv2.VideoCapture('C:/Users/pxb171530/Downloads/DEMO/dog1.mp4')
while(True):
    ret, frame = screen.read()
    if ret==True:
        
        
        testimg=cv2.resize(frame,(64,64))
        testimg1=cv2.resize(frame,(400,600))
        testimg= np.expand_dims(testimg,axis=0)
        
        result= model.predict(testimg)
        if(result==1):
            pred="dog detected"
        else:
            pred="cat detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,400)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(testimg1,str(pred), (10,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        
        cv2.imshow("Detection", testimg1)
        print(result)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break


screen.release()
cv2.destroyAllWindows()

'''Image iesting
testimg=  cv2.imread('C:/Users/pxb171530/Downloads/test1 (1).jpg')
testimg=cv2.resize(testimg,(64,64))
testimg= np.expand_dims(testimg,axis=0)
result=model.predict(testimg)
print(result)
'''
