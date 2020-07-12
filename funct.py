# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:29:55 2020

@author: eternal_demon
"""

import numpy as np
import cv2

def coordinates(image):
    
    temp =[]
  
    # Reading image 
    font = cv2.FONT_HERSHEY_COMPLEX 
    img2 = cv2.imread(image, cv2.IMREAD_COLOR) 
  
    # Reading same image in another  
    # variable and converting to gray scale. 
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
    
    # Converting image to a binary image 
    # ( black and white only image). 
    _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_OTSU) 
  
    # Detecting contours in image. 
    contours, _= cv2.findContours(threshold, cv2.RETR_TREE, 
                               cv2.CHAIN_APPROX_SIMPLE) 
    ymin=1000
    loc=0
    # Going through every contours found in the image.
    
    for cnt in contours : 
        hull = cv2.convexHull(cnt)  
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
  
        # draws boundary of contours. 
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)  
  
         # Used to flatted the array containing 
         # the co-ordinates of the vertices. 
        n = approx.ravel()  
        i = 0
  
        for j in n : 
            if(i % 2 == 0): 
                x = n[i] 
                y = n[i + 1] 
                if(y<ymin):
                    ymin=y
                    loc=i
                    # String containing the co-ordinates. 
                string = str(x) + " " + str(y)  
  
                if(i == 0): 
                    # text on topmost co-ordinate. 
                    cv2.putText(img, "", (x, y), 
                                font, 0.5, (0, 0, 0))  
                    if((x>=30 and x<=290) and (y>=30 and y<=210)):
                        temp.append((x,y))
                else: 
                # text on remaining co-ordinates. 
                    cv2.putText(img, string, (x, y),  
                          font, 0.5, (0, 0, 255)) 
                    if((x>=30 and x<=290) and (y>=30 and y<=210)):
                        temp.append((x,y))
            i = i + 1
  
    
    #cv2.drawContours(img,[cnt],0,(0,255,0),2)
    #img = cv2.drawContours(img,[hull],0,(0,0,255),2)
    # Showing the final image. 
    #cv2.imshow('image2', img)  

    # Exiting the window if 'q' is pressed on the keyboard. 
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows
    x_sum=0
    y_sum=0
    x_mean=0
    y_mean= 0
    #print(hull[0][0][0])
    #print(hull.shape)
    for (x,y) in temp:
        x_sum = x_sum + x
        y_sum = y_sum + y
    if(len(temp)>0):
        x_mean,y_mean = (x_sum/len(temp)), (y_sum/len(temp))
    else:
        x_mean,y_mean=0,0

    x,y = x_mean,y_mean
    #print((x,y))
    return [[int(x),int(y)]] 


def rescaleobj(userscreenx,userscreeny):
    screenx = 1366
    screeny = 768
    mulx= userscreenx/screenx
    muly= userscreeny/screeny
    objcoordinates=[]
    get1=[]
    get2=[]
    get3=[]
    get4=[]
    get1.append(round(mulx*194))
    get1.append(round(muly*345))
    objcoordinates.append(get1)
    get2.append(round(mulx*665))
    get2.append(round(muly*345))
    objcoordinates.append(get2)
    get3.append(round(mulx*194))
    get3.append(round(muly*395))
    objcoordinates.append(get3)
    get4.append(round(mulx*663))
    get4.append(round(muly*395))
    objcoordinates.append(get4)
    return objcoordinates

def rescaleocr(userscreenx,userscreeny):
    screenx = 1366
    screeny = 768
    mulx= userscreenx/screenx
    muly= userscreeny/screeny
    ocrcoordinates=[]
    get1=[]
    get2=[]
    get3=[]
    get4=[]
    get1.append(round(mulx*685))
    get1.append(round(muly*344))
    ocrcoordinates.append(get1)
    get2.append(round(mulx*1157))
    get2.append(round(muly*344))
    ocrcoordinates.append(get2)
    get3.append(round(mulx*686))
    get3.append(round(muly*395))
    ocrcoordinates.append(get3)
    get4.append(round(mulx*1156))
    get4.append(round(muly*393))
    ocrcoordinates.append(get4)
    return ocrcoordinates