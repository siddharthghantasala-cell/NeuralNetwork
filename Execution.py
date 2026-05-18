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
        hidden_layer_size= 200,
        hidden_layer_count= 7,
        activation_function= relu,
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
        x_train_aug.append(np.array(dp).flatten())


    mnist_network.train(
        learning_rate=0.01,
        data=x_train_aug,
        epochs=1,
        test_data=y_train,
    )

if __name__ == "__main__":
    main()