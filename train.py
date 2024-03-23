import pathlib
import cv2 as cv

from image_preprocessor import ImagePreprocessor

class Train:
    path = None
    image_preprocessor = None
    images = []
    labels = []

    def __init__(self, path, image_preprocessor=ImagePreprocessor()) -> None:
        self.path = pathlib.Path(path)
        self.image_preprocessor = image_preprocessor
        

    def label_data(self):
        for f in self.path.iterdir():
            for i in f.iterdir():
                image = cv.imread(i)
                image = self.image_preprocessor.process(image)
                self.images.append(image)
                self.labels.append()
