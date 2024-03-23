import cv2 as cv

class ImagePreprocessor:
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    def process(self, image):
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces_rect = self.haar_cascade.detectMultiScale(grayscale_image)
        if faces_rect:
            (x, y, w, h) = faces_rect[0]
            image = image[x:x+w, y:y+h]
        return image