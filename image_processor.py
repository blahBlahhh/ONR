import numpy as np


class ImageProcessor:

    def process(self, images):  # turn (N, 28, 28) into normalized (N, 28^2)
        processed_images = []
        normalized_images = np.round(images/255)  # normalized to 1:white | 2:black
        for idx in range(len(images)):
            processed_images.append(normalized_images[idx].reshape((1, -1)))  # turn array into (N, 1, 28^2)
        processed_images = np.array(processed_images).squeeze()  # squeeze the second axes, to (N, 28^2)
        return processed_images


if __name__ == '__main__':
    processor = ImageProcessor()
    a = np.array([[[1, 2, 3], [1, 3, 2], [2, 3, 1]], [[1, 2, 3], [1, 3, 2], [2, 3, 1]]])
    # a.shape is (2, 3, 3)
    print(processor.process(a).shape)
