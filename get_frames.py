import cv2
import torch
import torchvision

video_path = "/input/sample video.mp4"
cap = cv2.VideoCapture(video_path)
count = 0
if (cap.isOpened()==False):
    print("there was an error while opening video")

while cap.isOpened()==True:
    ret,frame = cap.read()
    if ret==True:
        cv2.write(f"/src/frames/{count}.png",frame)
        count+=1

    else:
        break

