import cv2 as cv
import numpy as np

from train import Train
from image_preprocessor import ImagePreprocessor

# model = cv.face.EigenFaceRecognizer.create()
# train = Train('C:\\Users\\nickl\\COP4930\\final_project\\images', ImagePreprocessor())
# images = np.array(train.images)
# labels =  np.array(train.labels)
# np.save('features.npy', images)
# np.save('labels.npy', labels)
# model.train(images, labels)
# model.save('trained_model.yml')

model = cv.face.EigenFaceRecognizer.create()
cv.face.EigenFaceRecognizer.read(model, 'trained_model.yml')
img = cv.imread("C:\\Users\\nickl\\COP4930\\final_project\\images\\0\\0.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(model.predict(gray))