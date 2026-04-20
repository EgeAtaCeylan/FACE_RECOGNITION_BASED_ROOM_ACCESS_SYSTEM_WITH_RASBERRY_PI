import cv2
import os
import pickle
import numpy as np
import socket
import struct
import io
from PIL import Image
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_vggface import utils
from keras.models import load_model

from numpy import asarray
import pandas as pd
import keras

from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

from keras.models import Model
from keras_vggface import VGGFace

# Load the trained face recognition model that was created by the training script.
# This model is used later to predict which known person appears in the frame.
model = load_model('final-face-recognition-model.h5')

# Input size expected by the trained model.
image_width = 224
image_height = 224

# Load the saved label mapping so model output indices can be converted back to names.
labelsFilename = 'person-labels.pickle'
with open(labelsFilename, "rb") as f:
    classDictionary = pickle.load(f)

# Build a list of class names from the loaded dictionary.
classList = [value for _, value in classDictionary.items()]
print(classList)

# Load OpenCV's Haar Cascade face detector.
# The code uses this to detect whether the frame contains 0, 1, or multiple faces.
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
)

# Create a TCP server socket that listens for the Raspberry Pi room client.
server_socket = socket.socket()
server_socket.bind(("0.0.0.0", 9000))
server_socket.listen(0)

# Wait until the room client connects, then create a file-like object for reading bytes.
conn = server_socket.accept()[0]
connection = conn.makefile('rb')

# Access lists for rooms.
# In the current code, room2access is the list that is actually checked.
room1access = ['John']
room2access = ['Mark', 'David']

# Face image size used before prediction.
imageWidth = 224
imageHeight = 224

try:
    count = 0
    while True:
        try:
            while True:
                # Read the length of the incoming image first.
                # The client sends this length as a packed unsigned long.
                imageLen = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]

                # A zero length means the client has finished sending frames.
                if not imageLen:
                    break

                # Read the raw image bytes into an in-memory stream.
                imageStream = io.BytesIO()
                imageStream.write(connection.read(imageLen))

                # Reset stream position, open the image with PIL, then convert it to a NumPy array.
                imageStream.seek(0)
                image = Image.open(imageStream)
                image = asarray(image)

                # Convert the frame to RGB because the later face-processing pipeline expects it.
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces in the current frame.
                detectedFaces = facecascade.detectMultiScale(
                    rgb, scaleFactor=1.3, minNeighbors=5
                )
                print("faces: ", len(detectedFaces))

                # Create a writable stream back to the connected client.
                datasend = conn.makefile('w')

                # Case 1: No face found.
                # Message format: "0 -1"
                if len(detectedFaces) == 0:
                    data = "0 -1"
                    datasend.write(data)
                    datasend.flush()

                # Case 2: More than one face found.
                # Message format: "2 2"
                elif len(detectedFaces) > 1:
                    data = "2 2"
                    datasend.write(data)
                    datasend.flush()

                # Case 3: Exactly one face found.
                else:
                    for (x, y, w, h) in detectedFaces:
                        # Crop the detected face region from the frame.
                        roiRgb = rgb[y:y + h, x:x + w]

                        # Resize the face to the model's required input size.
                        size = (imageWidth, imageHeight)
                        resizedImage = cv2.resize(roiRgb, size)

                        # Convert the face into a batch tensor and preprocess it for VGGFace.
                        faceToBePredicted = keras.utils.img_to_array(resizedImage)
                        faceToBePredicted = np.expand_dims(faceToBePredicted, axis=0)
                        faceToBePredicted = utils.preprocess_input(faceToBePredicted, version=1)

                        # Run the face recognition model.
                        predictionResults = model.predict(faceToBePredicted)
                        print(predictionResults[0])

                        # Decide whether this should be treated as a known face.
                        # The project uses a score threshold instead of always forcing a class choice.
                        knownFace = False
                        for predictionValue in predictionResults[0]:
                            if predictionValue > 300.0:
                                knownFace = True

                        if knownFace:
                            # Select the name corresponding to the highest score.
                            name = classList[predictionResults[0].argmax()]

                            # Check whether the recognized person has permission to enter this room.
                            if name in room2access:
                                # Access granted.
                                # Message format: "1 <name>"
                                data = '1 ' + name
                                print("Predicted face: " + classList[predictionResults[0].argmax()])
                                datasend.write(data)
                                datasend.flush()
                            else:
                                # Person is known, but not authorized.
                                # Message format: "0 <name>"
                                data = '0 ' + name
                                datasend.write(data)
                                datasend.flush()

                        else:
                            # Face is treated as unknown.
                            # Message format: "0 0"
                            name = "UnKnown Face"
                            print("Unknown Face")
                            data = '0 0'
                            datasend.write(data)
                            datasend.flush()

        finally:
            # Exit the outer loop cleanly after the client disconnects.
            break

finally:
    # Always close the connection and server socket so the program exits safely.
    connection.close()
    server_socket.close()
