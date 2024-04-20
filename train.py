import cv2 as cv

class Train:
    data_paths = []
    model = None
    image_preprocessor = None
    train_x = []
    train_y = []
    test_x = []
    test_y = []


    def __init__(self, paths, model, image_preprocessor, split) -> None:
        self.data_paths = paths
        self.image_preprocessor = image_preprocessor
        self.model = model
        images, labels = self._label_data()

        

    def _label_data(self, split):
        images = []
        labels = []
        name_to_label = {}
        i = 0
        for path in self.data_paths:
            for f in path.iterdir():
                if f.name not in name_to_label:
                    name_to_label[f.name] = i
                    i = i + 1
                label = name_to_label[f.name]
                for im in f.iterdir():
                    image = cv.imread(str(im))
                    if (self.image_preprocessor):
                        image = self.image_preprocessor.process(image)
                    self.images.append(image)
                    self.labels.append(i)
        return images, labels
    

    

    def cross_validate(self):
        pass