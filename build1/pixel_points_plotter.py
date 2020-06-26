import cv2
import numpy as np
cap = cv2.VideoCapture('pedestrians.mp4') 
ix, iy = -1,-1

def mouse_pixel_tracker(event, x, y, flags, param):
    global ix
    global iy

    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x 
        iy = y
        print(x, " ", y)

def circle_onscreen(frame, x, y):
    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_pixel_tracker) 

while(1):
    ret, img = cap.read()
    circle_onscreen(img, ix, iy) 
    cv2.imshow('frame', img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()



# compiled and coded by GunH-colab
# The above .py is for printing the pixel points clicked on a video
