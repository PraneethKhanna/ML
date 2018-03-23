import numpy as np
import cv2

while(1):
    #cv2.namedWindow("output1" , cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
    #cv2.namedWindow("output2" , cv2.WINDOW_NORMAL) 
    boundaries = [([30, 100, 50], [80, 255, 255])]
    screen= cv2.imread('C:/Users/neon/Desktop/UTD slides/VA/pool.jpg')
    fgbg=cv2.createBackgroundSubtractorMOG2()
    nxt = cv2.cvtColor(screen,cv2.COLOR_BGR2HSV)
    nxt=cv2.medianBlur(nxt,5)
    #cv2.resizeWindow("output1",900,600)
    #cv2.imshow("output", nxt) 
    #cv2.imshow('output',frame2)
    for (lower, upper) in boundaries:
    
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply
        # the mask
    maski = cv2.inRange(nxt, lower, upper)
    
    kernel = np.ones((5,5),np.uint8)
    mask1 = cv2.erode(maski,kernel,iterations = 2)
    mask2 = cv2.dilate(mask1,kernel,iterations = 2)
    cnts=cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(screen,[box],0,(0,0,255),2)
    cv2.drawContours(screen,[hull],0,(255,255,255),2)
    cv2.rectangle(screen,(x,y),(x+w,y+h),(0,255,0),2)
    area=cv2.contourArea(c)
    '''
    ((x, y), l,w) = cv2.boundingRect(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    cv2.circle(screen, (int(x), int(y)), int(l),int(w),(0, 255, 255), 2)
    cv2.rectangle(screen,)
    cv2.circle(screen, center, 5, (0, 0, 255), -1)
''' 
	# update the points queue
	#pts.appendleft(center)
    
    #cv2.drawContours(image, contours, 3, 255, 3)
    #img2 = cv2.drawContours(image, contours, -1, 255, 3)
    #cv2.imshow("mask",maski)
    #output = cv2.bitwise_and(nxt, nxt, mask = maski)
    
    #colored=cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
    #fgmask=fgbg.apply(output)
    #kernel = np.ones((5,5),np.uint8)
    #fgmask1 = cv2.erode(fgmask,kernel,iterations = 2)
    #final1 = cv2.bitwise_and(nxt, nxt, mask = fgmask)
    #cv2.imshow("output1", np.hstack([fgmask,fgmask1]))
    #cv2.imshow('output1',np.hstack([img,img2]))
    #cv2.imshow('output2',img)
    cv2.imshow('output',screen)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break

cv2.destroyAllWindows()