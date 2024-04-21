import cv2 as cv
from sklearn.model_selection import train_test_split, KFold
import numpy as np

class Train:
    data_paths = []
    image_preprocessor = None
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label_to_name = []
    dimensions = None


    def __init__(self, paths, image_preprocessor, split) -> None:
        self.data_paths = paths
        self.image_preprocessor = image_preprocessor
        images, labels = self._label_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=split,
                                                                                stratify=labels)
        self.X_train = np.array(self.X_train)
        self.dimensions = (self.X_train.shape[1], self.X_train.shape[2])
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

    def _label_data(self):
        images = []
        labels = []
        name_to_label = {}
        i = 0
        for path in self.data_paths:
            for f in path.iterdir():
                if f.name not in name_to_label:
                    self.label_to_name.append(f.name)
                    name_to_label[f.name] = i
                    i = i + 1
                label = name_to_label[f.name]
                for im in f.iterdir():
                    image = cv.imread(str(im))
                    if (self.image_preprocessor):
                        image = self.image_preprocessor.process(image)
                    images.append(image)
                    labels.append(label)
        return images, labels

    def cross_validate(self, model, folds, metric):
        res = []
        kf = KFold(n_splits=folds)
        for (train_indices, test_indices) in kf.split(self.X_train):
            model.train(self.X_train[train_indices], self.y_train[train_indices])
            true = self.y_train[test_indices]
            predicted = [model.predict(self.X_train[i])[0] for i in test_indices]
            res.append(metric(true, predicted))
        return res

    def train(self, model):
        model.train(self.X_train, self.y_train)
    
    def test(self, model, metric):
        predicted = [model.predict(x)[0] for x in self.X_test]
        return metric(self.y_test, predicted)