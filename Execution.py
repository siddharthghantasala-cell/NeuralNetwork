import pickle
import random
from operator import indexOf
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

from PIL import Image


def main():
    plt.close()
    sample = np.array(
        Image.open(r"C:\Users\sidds\OneDrive\Desktop\Number.png")
        .convert("L")  # grayscale
        .resize((28, 28))  # resize to 28x28
    )
    mnist_network = pickle.load(open("mnist_network.p", "rb"))
    test = 255 - np.array(sample, dtype=np.int64).flatten()
    test = (test - test.min()) / (test.max() - test.min())
    mnist_network.forward(test.reshape(-1, 1))
    result = np.argmax(mnist_network.return_output())
    plt.imshow(sample)
    plt.title(f"The network thought that this was a {result}")
    plt.axis('off')
    plt.show()

    # samples = [
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1]
    # ]
    # network = pickle.load(open('xor_network.p', 'rb'))
    # for i in samples:
    #     network.forward(np.array(i).reshape(-1, 1))
    #     network.show_output(0)


if __name__ == "__main__":
    main()