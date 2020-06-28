import numpy as np
import cv2
import math
import imutils
import os
import time
import matplotlib.pyplot as plt

class social_distancing_detector:
    def __init__(self,video_path,out_path,weight_path,cfg_path,label_name_path,thresh):
        self.video_path = video_path
        self.out_path = out_path
        self.weight_path = weight_path
        self.cfg_path = cfg_path
        self.label_name_path = label_name_path
        self.thresh = thresh
  
    def Check(self, a,  b):
        dist = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)       
        if 0 < dist < self.thresh:
            return True
        else:
            return False
  
    def get_fps(self):
        video = cv2.VideoCapture(self.video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        return fps

    def setup(self):
        global net, ln, LABELS
        LABELS = open(self.label_name_path).read().strip().split("\n")  
        net = cv2.dnn.readNet(self.cfg_path, self.weight_path)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return LABELS,net,ln

    def ImageProcess(self,image):
        (H, W) = (None, None)
        frame = image.copy()
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),(0,0,0), swapRB=True, crop=False)
        LABELS,net,ln = self.setup()
        net.setInput(blob)
        starttime = time.time()
        layerOutputs = net.forward(ln)
        stoptime = time.time()
        print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime-starttime))) 
        confidences = []
        outline = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                maxi_class = np.argmax(scores)
                confidence = scores[maxi_class]
                if LABELS[maxi_class] == "person":
                    if confidence > 0.5:
                        print("there is confidence")
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        outline.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))

        box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.5)

        if len(box_line) > 0:
            flat_box = box_line.flatten()
            pairs = []
            center = []
            status = [] 
            for i in flat_box:
                (x, y) = (outline[i][0], outline[i][1])
                (w, h) = (outline[i][2], outline[i][3])
                center.append([int(x + w / 2), int(y + h / 2)])
                status.append(False)

            for i in range(len(center)):
                for j in range(len(center)):
                    if i!=j:
                        close = self.Check(center[i], center[j])

                        if close:         
                            status[i] = True
                            status[j] = True
            index = 0

            for i in flat_box:
                (x, y) = (outline[i][0], outline[i][1])
                (w, h) = (outline[i][2], outline[i][3])
                if status[index] == True:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                    print("its there")
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print("its here too")
                index += 1

        processedImg = frame.copy()
        return processedImg

    def process_video(self):
        create = None
        frameno = 0
        cap = cv2.VideoCapture(self.video_path)
        time1 = time.time()
        while(True):
          ret,frame = cap.read()
          if ret==False:
            break

          current_img = frame.copy()
          current_img = imutils.resize(current_img, width=480)
          frame_shape = current_img.shape

          frameno+=1
          processed_img = self.ImageProcess(current_img)
          if create is None:
            fps = self.get_fps()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(self.out_path, fourcc,fps, (processed_img.shape[1], processed_img.shape[0]), True)
          create.write(processed_img)
        time2 = time.time()
        print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))


#model for obtaining resulted video after detection
