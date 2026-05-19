import ActivationFunctions
from Network import Network
from ActivationFunctions import *
from MNISTReader import MnistDataloader

def main():
    # X = [
    #     np.array([0, 0]),
    #     np.array([0, 1]),
    #     np.array([1, 0]),
    #     np.array([1, 1]),
    # ]
    # y = [
    #     np.array([0]),
    #     np.array([1]),
    #     np.array([1]),
    #     np.array([1]),
    # ]
    #
    # network = Network(
    #     input_size=2,
    #     output_size=1,
    #     hidden_layer_size=0,
    #     hidden_layer_count=0,
    #     activation_function=tanh,
    #     output_activation=tanh
    # )
    #
    # print("training...")
    # network.train(0.1, X, 10000, y)
    #
    # print("\n--------------- testing ---------------\n")
    #
    # outputs = []
    # for entry in X:
    #     network.forward(entry)
    #     network.show_output()
    #     outputs.append(network.return_output())
    #
    # print()
    #
    # for i in range(len(outputs)):
    #     if outputs[i] >= 0.5:
    #         outputs[i] = 1
    #     else:
    #         outputs[i] = 0
    # print(outputs)
    mnist_network = Network(
        input_size=784,
        output_size=10,
        hidden_layer_size= 522,
        hidden_layer_count= 7,
        activation_function= sigmoid,
        output_activation= softmax,
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
    for dp in x_train[:10]:
        flattened = (np.array(dp, dtype=np.int64).flatten())
        max_val = max(flattened)
        min_val = min(flattened)
        x_train_aug.append((flattened - min_val) / (max_val - min_val))

    y_train_aug = []
    for label in y_train[:10]:
        y_train_aug.append(
            np.array([0 for _ in range(label)] + [1] + [0 for _ in range(10 - label)])
        )

    mnist_network.train(
        learning_rate=0.01,
        data=x_train_aug,
        epochs=1,
        test_data=y_train_aug,
    )

    test_i = 7
    # test_label = np.array([0 for _ in range(y_test[test_i])] + [1] + [0 for _ in range(10 - y_test[test_i])])
    test = np.array(x_test, dtype=np.int64).flatten()
    max_val = max(test)
    min_val = min(test)
    test = test - min_val / (max_val - min_val)
    mnist_network.forward(test)
    mnist_network.show_output()
    print(f"<Execution> Expected output is {y_test[test_i]}")

if __name__ == "__main__":
    main()