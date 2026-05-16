from Network import Network
from ActivationFunctions import *

def main():
    X = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ]
    y = [
        np.array([0]),
        np.array([1]),
        np.array([1]),
        np.array([0]),
    ]

    network = Network(
        input_size=2,
        output_size=1,
        hidden_layer_size=3,
        hidden_layer_count=1,
        activation_function=sigmoid,
        output_activation=sigmoid
    )

    print("training...")
    network.train(0.1, X, 10000, y)

    print("\n--------------- testing ---------------\n")

    outputs = []
    for entry in X:
        network.forward(entry)
        network.show_output()
        outputs.append(network.return_output())

    print()

    print("After softmax : ")
    for i in range(len(outputs)):
        if outputs[i] >= 0.5:
            outputs[i] = 1
        else:
            outputs[i] = 0
    print(outputs)




if __name__ == "__main__":
    main()