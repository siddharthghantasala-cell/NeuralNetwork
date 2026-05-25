import pickle
import numpy as np
from MNISTTraining import MnistDataloader
from os.path import join



def main():
    input_path = 'MNIST/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    m_reader = MnistDataloader(
        training_images_filepath=training_images_filepath,
        training_labels_filepath=training_labels_filepath,
        test_images_filepath=test_images_filepath,
        test_labels_filepath=test_labels_filepath,
    )

    (x_train, y_train), (x_test, y_test) = m_reader.load_data()

    mnist_network = pickle.load(open("mnist_network.p", "rb"))
    print("testing...")
    test_i = 7
    # test_label = np.array([0 for _ in range(y_test[test_i])] + [1] + [0 for _ in range(10 - y_test[test_i])])
    test = np.array(x_test[test_i], dtype=np.int64).flatten()
    max_val = max(test)
    min_val = min(test)
    test = ((test - min_val) / (max_val - min_val))
    mnist_network.forward(test, batch_size=1)
    result = mnist_network.return_output()
    print("Network output: \n", np.resize(result, 10))
    print(f"<Execution> Expected output is {y_test[test_i]}")




if __name__ == "__main__":
    main()