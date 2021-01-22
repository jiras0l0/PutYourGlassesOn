#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow SavedModel export package from Lobe.
"""
import argparse
import os
import cv2
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import corrector as corrector
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import queue


MODEL_DIR = os.path.join(os.path.dirname(__file__), "./model")  # default assume that our export is in this file's parent directory


class Model(object):
    def __init__(self, model_dir=MODEL_DIR):
        # make sure our exported SavedModel folder exists
        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        # placeholder for the tensorflow session
        self.session = None

    def load(self):
        self.cleanup()
        # create a new tensorflow session
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        # load our model into the session
        tf.compat.v1.saved_model.loader.load(sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):
        # load the model if we don't have a session
        if self.session is None:
            self.load()
        # get the image width and height
        width, height = image.size
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        if "Image" not in self.inputs:
            raise ValueError("Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # create the feed dictionary that is the input to the model
        # first, add our image to the dictionary (comes from our signature.json file)
        feed_dict = {self.inputs["Image"]["name"]: [image]}

        # list the outputs we want from the model -- these come from our signature.json file
        # since we are using dictionaries that could have different orders, make tuples of (key, name) to keep track for putting
        # the results back together in a dictionary
        fetches = [(key, output["name"]) for key, output in self.outputs.items()]

        # run the model! there will be as many outputs from session.run as you have in the fetches list
        outputs = self.session.run(fetches=[name for _, name in fetches], feed_dict=feed_dict)
        # do a bit of postprocessing
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def cleanup(self):
        # close our tensorflow session if one exists
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()

def __KO_LAYER__(image):
    start_point = (50, 50) 
    end_point = (image.shape[1] - 50 , image.shape[0] - 50)

    overlay = image.copy()
    output = image.copy()

    alpha = 0.4

    cv2.rectangle(overlay, start_point, end_point,
		(0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
		0, output)

    return output
    #return cv2.rectangle(image, start_point, end_point, (0, 0, 255), -1)


def __WaitForFace__():

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while(True):
            ret, frame = capture.read()
            img = Image.fromarray(frame, 'RGB')

            #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window', frame)
            
            # Our operations on the frame come here
            result = model.predict(img)
            q.put(result['Prediction'])

            print("queue size", q.qsize(), 'prediction=', result['Prediction'])

            countGlassOn = 0

            if(q.qsize() >= 50):
                while(q.empty() != True):
                    if(q.get() == 'GlassOn'):
                        countGlassOn += 1

                if((countGlassOn / 40) > 0.9):
                    capture.release()
                    cv2.destroyAllWindows()
                    scheduler.resume()
                    return

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()

def checkGlasses(): 

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = Image.fromarray(frame, 'RGB')
        
        # Our operations on the frame come here
        result = model.predict(img)
        q.put(result['Prediction'])

        print("queue size", q.qsize(), 'prediction=', result['Prediction'])

        countGlassOff = 0

        if(q.qsize() >= 50):
            while(q.empty() != True):
                if(q.get() == 'GlassOff'):
                    countGlassOff += 1

            if((countGlassOff / 40) > 0.8):
                scheduler.pause()
                cap.release()
                __WaitForFace__()
                cv2.destroyAllWindows()

            break

    cap.release()
    cv2.destroyAllWindows()

try:
    q = queue.Queue()
    model = Model()
    model.load()
    scheduler = BlockingScheduler()
    scheduler.add_job(checkGlasses, 'interval', seconds=20, id='some_job_id')
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass