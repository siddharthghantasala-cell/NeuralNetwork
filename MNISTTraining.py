import time

from Network import Network
from ExternalFunctions import *
from MNISTReader import MnistDataloader
import pickle

def main():
    mnist_network = Network(
        input_size=784,
        output_size=10,
        hidden_layer_size= 100,
        hidden_layer_count= 3,
        activation_function= relu,
        output_activation= softmax,
        initialization=he_initialization,
    )

    from os.path  import join

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

    # Flattening the datapoints into 1D arrays
    x_train_aug = []
    for dp in x_train:
        flattened = (np.array(dp, dtype=np.int64).flatten())
        max_val = max(flattened)
        min_val = min(flattened)
        x_train_aug.append((flattened - min_val) / (max_val - min_val))

    y_train_aug = []
    for label in y_train:
        y_train_aug.append(
            np.array([0 for _ in range(label)] + [1] + [0 for _ in range(10 - label - 1)])
        )

    print("training...")
    t0 = time.time()

    mnist_network.mini_batch_grad_desc(
        learning_rate=0.05,
        data=x_train_aug,
        epochs=25,
        labels=y_train_aug,
        batch_size=64,
        loss=cross_entropy
    )

    t1 = time.time()

    print("training complete!")
    print("elapsed time: ", round(t1 - t0, 2))

    pickle.dump(mnist_network, open('mnist_network.p', 'wb'))

    mnist_network.plot_loss()

    print("testing...")

    errors = 0

    mnist_network.plot_loss()

    for i in range(len(x_test)):
        test = np.array(x_test[i], dtype=np.int64).flatten()
        test = (test - test.min()) / (test.max() - test.min())
        mnist_network.forward(test.reshape(-1, 1))
        pred = np.argmax(mnist_network.return_output())
        if pred != y_test[i]:
            errors += 1
    print(f"accuracy: {(len(x_test) - errors) / len(x_test):.2%}")

if __name__ == "__main__":
    main()