
import numpy as np
import cv2

peopleout,peoplein = 0,0
line = 300 

cap = cv2.VideoCapture('peopleCount.mp4') #Open video file

#Create the background substractor
#fgbg = cv2.BackgroundSubtractorMOG()

#Create the background substractor
#******Creates a GMG Background Subtractor.
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

#******Creates a CNT Background Subtractor.
#fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()

#******Creates mixture-of-gaussian background subtractor.
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


frame_empty = cv2.imread("frame_empty.jpeg")

contours_previous = []
contours_now = []

fgmask = fgbg.apply(frame_empty)

sayac = 0
while(cap.isOpened()):
            
        ret, frame = cap.read() #read a frame  
        
        fgmask = fgbg.apply(frame) #Use the substractor
        thresh = cv2.dilate(fgmask, None, iterations=2)                  
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        contours_now = []
        # loop over the contours
        for c in cnts:
        		
            if cv2.contourArea(c) < 1000:
                continue
             
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            
            contours_now.append([x,y])
        
        
        if ( len(contours_previous) == 0 ): 
            contours_previous = contours_now 
            continue
        
        
        closest_contour_list = [] 

        for i in range ( len(contours_now) ):
            
            minimum = 1000000
               
            
            for k in range ( len(contours_previous) ):
                
                diff_x = contours_now[i][0] - contours_previous[k][0]
                diff_y = contours_now[i][1] - contours_previous[k][1]
                
                distance = diff_x * diff_x + diff_y * diff_y
                
                if ( distance < minimum ):
                    minimum =  distance
                    closest_contour = k
                    
            closest_contour_list.append(closest_contour)
            
         
        for i in range ( len(contours_now) ):
            
            y_previous = contours_previous[closest_contour_list[i]][1]
 
            if ( contours_now[i][1] < line and y_previous > line ):
                peopleout = peopleout + 1 
        
            if ( contours_now[i][1] > line and y_previous < line ):
                peoplein = peoplein + 1 
        
        contours_previous = contours_now     
            
        # Draw line
        cv2.line(frame,(0,line),(frame.shape[1],line),(0,255,255),2)
        
        
        cv2.putText(frame, "out: " + str(peopleout) ,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame, "in: " + str(peoplein) ,(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)        
        cv2.imshow('Frame',frame)
        #cv2.imshow('Background Substraction',fgmask)    

        
        #Abort and exit with 'Q' or ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows