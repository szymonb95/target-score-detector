from matplotlib import pyplot as plt
import numpy as np

def draw_images(images: list[tuple], rows: int, columns: int, figsize: tuple):
    fig = plt.figure(figsize=figsize)
    for index, (name, image, cmap) in enumerate(images):
        fig.add_subplot(rows, columns, index+1)
        plt.imshow(image, cmap=cmap)
        plt.title(name)
    plt.show()