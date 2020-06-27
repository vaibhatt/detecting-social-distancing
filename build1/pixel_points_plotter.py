import numpy
import cv2

class get_points:
    def __init__(self,video_path):
        self.video_path = video_path
        self.ix = -1
        self.iy = -1
    
    def mouse_pixel_tracker(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix = x 
            self.iy = y
            print(x, " ", y)
    def circle_onscreen(self,frame):
        x = self.ix
        y = self.iy
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouse_pixel_tracker)
        while(1):
            ret, img = cap.read()
            self.circle_onscreen(img) 
            cv2.imshow('frame', img)

            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
