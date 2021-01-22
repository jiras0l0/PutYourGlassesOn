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


        
def __Rectangle__(image):

    print(image.shape)

    # Start coordinate, here (5, 5) 
    # represents the top left corner of rectangle 
    start_point = (50, 50) 
    
    # Ending coordinate, here (220, 220) 
    # represents the bottom right corner of rectangle 
    end_point = (image.shape[1] - 50 , image.shape[0] - 50) 
    
    # Blue color in BGR 
    color = (100, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 1
    
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
    image = cv2.rectangle(image, start_point, end_point, color, thickness) 

    return image


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


if __name__ == "__main__":
    

    model = Model()
    model.load()
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('rtsp://admin:bazar@192.168.1.55:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
    

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = Image.fromarray(frame, 'RGB')
        # Our operations on the frame come here
        result = model.predict(img)
    
        if(result['Prediction'] == 'Sans masque'):
            print(result)

            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            
            # org 
            org = (200 , 30) 
            
            # fontScale 
            fontScale = 1
            
            # Blue color in BGR 
            color = (0, 0, 255) 
            
            # Line thickness of 2 px 
            thickness = 2
            frame = cv2.putText(frame, result['Prediction'], org, font,  fontScale, color, thickness, cv2.LINE_AA) 

            frame = __KO_LAYER__(frame)
           

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()