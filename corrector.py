import os
import cv2
from PIL import Image
import numpy as np

class corrector(object):

    def __WaitForFace__(self):
        width, height = 800, 600

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        while(True):
            ret, frame = cap.read()
            img = Image.fromarray(frame, 'RGB')

            cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()