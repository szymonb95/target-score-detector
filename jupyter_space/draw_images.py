from matplotlib import pyplot as plt
import numpy as np
import cv2

def draw_images(images: list[tuple], rows: int, columns: int, figsize: tuple):
    fig = plt.figure(figsize=figsize)
    for index, (name, image, cmap) in enumerate(images):
        fig.add_subplot(rows, columns, index+1)
        plt.imshow(image, cmap=cmap)
        plt.title(name)
    plt.show()

def convert_image_to_plt(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
