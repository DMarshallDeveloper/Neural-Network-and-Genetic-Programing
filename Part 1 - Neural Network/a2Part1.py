import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from NeuralNetwork import Neural_Network
from NeuralNetworkBiased import Neural_Network_Biased
import sys


def encode_labels(labels):
    # encode 'Adelie' as 0, 'Gentoo' as 2, 'Chinstrap' as 1,...
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


def test(test_instances, test_labels, nn):
    # Test the network on the test data
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(test_labels)
    predictions = []
    for i, instance in enumerate(test_instances):
        predicted_class = nn.predict([instance])
        predictions.append(predicted_class)

    # Classify each instance
    num_correct = 0
    for i in range(len(predictions)):
        actual_class = np.where(onehot_encoded[i] == np.amax(onehot_encoded[i]))[0]
        if predictions[i] == actual_class:
            num_correct = num_correct + 1
    acc = num_correct / len(predictions)
    print("")
    print('Test Set accuracy: %0.2f%%' % (100 * acc))


def run(nn, instances, onehot_encoded, epochs, pd_data_ts):
    # Train for 100 epochs, on all instances.
    nn.train(instances, onehot_encoded, epochs)
    print('\nAfter training:')
    print('Final Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Final Output layer weights:\n', nn.output_layer_weights)

    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]
    # scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    # Compute and print the test accuracy
    test(test_instances, test_labels, nn)


if __name__ == '__main__':
    if (sys.argv[1:])[0].isdigit():
        argument = int((sys.argv[1:])[0])
    else:
        argument = 2
    data = pd.read_csv('Data/penguins307-train.csv')
    pd_data_ts = pd.read_csv('Data/penguins307-test.csv')
    # the class label is last!
    labels = data.iloc[:, -1]
    # separate the data from the labels
    instances = data.iloc[:, :-1]
    # scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2
    epochs = 100

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
    initial_hidden_layer_biases = np.array([-0.02, -0.20])
    initial_output_layer_biases = np.array([-0.33, 0.26, 0.06])

    if argument == 0:
        # Run first on a simple neural network with no biases
        nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                            learning_rate)
        print("Without Biases")
        run(nn, instances, onehot_encoded, epochs, pd_data_ts)
        print("")
        print("")
    elif argument == 1:
        # Run on a network that includes biases
        nn_biases = Neural_Network_Biased(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                                          initial_hidden_layer_biases, initial_output_layer_biases, learning_rate)
        print("With Biases")
        print("")
        run(nn_biases, instances, onehot_encoded, epochs, pd_data_ts)
    else:
        print("Error: invalid argument")
