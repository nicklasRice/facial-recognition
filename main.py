import cv2 as cv
import numpy as np

from train import Train

model = cv.face.LBPHFaceRecognizer()
train = Train(r'C:\Users\nickl\COP4930\final_project\images')
images = np.array(train.images, dtype='object')
labels =  np.array(train.labels)
model.train(images, labels)
np.save('features.npy', images)
np.save('labels.npy', labels)
model.save('trained_model.yml')