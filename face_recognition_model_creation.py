import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import pandas as pd
import keras

from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras_vggface import VGGFace
from keras.optimizers import Adam

# Root folder that contains one subfolder per person.
# Each subfolder is expected to contain training photos for that identity.
trainingFolderName = 'TrainingPhotos'

# Image size expected by the model.
imageWidth = 224
imageHeight = 224

# Haar Cascade face detector used to find and crop a single face from each image.
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

imagesDir = os.path.join(".", trainingFolderName)

# Prepare a label mapping from person name to numeric ID.
currentID = 0
personLabelswID = {}

# Walk through all images in the training dataset.
for root, _, files in os.walk(imagesDir):
    for file in files:
        count = 0
        print(file)

        # Only process supported image files.
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            print(path)

            # Use the folder name as the person's label.
            currentPersonName = os.path.basename(root).replace(" ", ".").lower()

            # Assign a numeric label ID the first time a person name is seen.
            if not currentPersonName in personLabelswID:
                personLabelswID[currentPersonName] = currentID
                currentID += 1

            # Read the image and convert it to a NumPy array for face detection.
            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            arrayOfImage = np.array(imgtest, "uint8")

            # Detect faces in the original training photo.
            faces = faceDetector.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

            print(len(faces))
            if len(faces) != 1:
                print("NOT EXACTLY 1 FACE IS DETECTED DISCARDING THE PHOTO")

            # Remove the original file.
            # If exactly one face is found, a cropped replacement image is saved below.
            os.remove(path)

            for (x_, y_, w, h) in faces:
                # Crop the detected face region.
                roi = arrayOfImage[y_: y_ + h, x_: x_ + w]

                # Resize the face crop to the model input size.
                size = (imageWidth, imageHeight)
                resizedImage = cv2.resize(roi, size)
                arrayOfImage = np.array(resizedImage, "uint8")

                # Save the cropped face back to the same path.
                finalFaceImage = Image.fromarray(arrayOfImage)
                finalFaceImage.save(path)

# Create a data generator that applies preprocessing during training.
TrainingDatas = ImageDataGenerator(preprocessing_function=preprocess_input)

# Build a generator that reads images from the TrainingPhotos directory.
# Subfolder names automatically become class labels.
TrainingDataGenerator = TrainingDatas.flow_from_directory(
    imagesDir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Count how many people/classes are in the dataset.
NO_CLASSES = len(TrainingDataGenerator.class_indices.values())
print("NUM CLASSES", NO_CLASSES)

# Load VGGFace without its original top classification layers.
# This acts as the pretrained feature extractor.
baseFaceDetectionModel = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

# Add a custom classification head on top of the pretrained backbone.
x = baseFaceDetectionModel.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# Final output layer uses softplus, following the project's design choice
# for score-based known/unknown handling.
preds = Dense(NO_CLASSES, activation='softplus')(x)

# Build the final trainable model.
finalCustomTrainedModel = Model(baseFaceDetectionModel.input, preds)

# Freeze earlier layers so the pretrained low-level face features stay unchanged.
for layer in finalCustomTrainedModel.layers[:19]:
    layer.trainable = False

# Fine-tune later layers and the custom head.
for layer in finalCustomTrainedModel.layers[19:]:
    layer.trainable = True

# Compile the model for training.
finalCustomTrainedModel.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on the prepared face dataset.
finalCustomTrainedModel.fit(TrainingDataGenerator, batch_size=1, verbose=1, epochs=25)

# Save the trained model so the server can load it later for inference.
finalCustomTrainedModel.save('final-face-recognition-model.h5')

# Invert the class index mapping so prediction indices can be converted back to names.
dictionaryOfPersons = TrainingDataGenerator.class_indices
dictionaryOfPersons = {
    value: key for key, value in dictionaryOfPersons.items()
}
print(dictionaryOfPersons)

# Save the label mapping to disk for use by the server.
memberLabels = 'person-labels.pickle'
with open(memberLabels, 'wb') as f:
    pickle.dump(dictionaryOfPersons, f)

# Free the model from memory when training is complete.
del finalCustomTrainedModel
