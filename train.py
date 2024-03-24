import pathlib
import cv2 as cv

class Train:
    path = None
    image_preprocessor = None
    images = []
    labels = []
    label_names = []


    def __init__(self, path, image_preprocessor=None) -> None:
        self.path = pathlib.Path(path)
        self.image_preprocessor = image_preprocessor
        self._label_data()
        

    def _label_data(self):
        for i, f in enumerate(self.path.iterdir()):
            if i == 10:
                break
            self.label_names.append(f.name)
            for im in f.iterdir():
                image = cv.imread(str(im))
                if (self.image_preprocessor):
                    image = self.image_preprocessor.process(image)
                self.images.append(image)
                self.labels.append(i)

    def cross_validate(self, model):
        pass