import mnist_loader
import image_processor

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class NumberRecognizer:
    def __init__(self):
        self.loader = mnist_loader.MnistLoader()
        self.processor = image_processor.ImageProcessor()
        self.error = []

    def vector_images(self):  # get processed train and test image
        x_train_raw = self.loader.load_image_set()[:50000]
        self.x_train = self.processor.process(x_train_raw)
        self.y_train = np.ravel(self.loader.load_label_set()[:50000])

        x_test_raw = self.loader.load_image_set(1)[:1000]
        self.x_test = self.processor.process(x_test_raw)
        self.y_test = np.ravel(self.loader.load_label_set(1))[:1000]

    def all_train(self):  # train all images
        self.svc = SVC(kernel='linear', C=1.0)
        print('fitting all data...')
        self.svc.fit(self.x_train, self.y_train)
        print('fit data complete')

        # y_pred = np.ravel(np.array(self.svc.predict(self.x_test)).reshape((-1, 1)))
        # print('Number of misclassified: %d' % (y_pred != self.y_test).sum())
        # print('Accuracy: %.3f' % accuracy_score(self.y_test, y_pred))

    def test_c_train(self, c):  # train all images
        self.svc = SVC(kernel='linear', C=c)
        print('fitting all data...%.1f' % c)
        self.svc.fit(self.x_train, self.y_train)
        print('fit data complete')

        y_pred = np.ravel(np.array(self.svc.predict(self.x_test)).reshape((-1, 1)))
        print('Number of misclassified: %d' % (y_pred != self.y_test).sum())
        print('Accuracy: %.3f' % accuracy_score(self.y_test, y_pred))
        self.error = [(y_pred[i], self.y_test[i], i) for i in range(len(self.y_test)) if y_pred[i] != self.y_test[i]]

    def predict(self, img):
        if img.shape == (28, 28):
            img = img.reshape(1, 28*28)
            y_pred = self.svc.predict(img)
            return y_pred
        else:
            print('Wrong image!')

    def recognize(self):  # main executor
        self.vector_images()
        self.all_train()

    def accuracy_plot(self):  # draw the accuracy plot | x:number of train samples | y:number of errors
        plt.plot(range(1000, len(self.error) * 1000, 1000), self.error)
        plt.xlabel('Epochs')
        plt.ylabel('Misclassified sample number')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    recognizer = NumberRecognizer()
    recognizer.vector_images()
    recognizer.test_c_train(1)
    print(recognizer.error)
