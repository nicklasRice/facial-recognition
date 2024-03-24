import cv2 as cv

class ImagePreprocessor:
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    def process(self, image):
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces_rect = self.haar_cascade.detectMultiScale(grayscale_image)
        if len(faces_rect) > 0:
            (x, y, w, h) = faces_rect[0]
            face = grayscale_image[x:x+w, y:y+h]
            face = cv.resize(face, (image.shape[0], image.shape[1]), interpolation=cv.INTER_LINEAR)
            grayscale_image = face
        return grayscale_image