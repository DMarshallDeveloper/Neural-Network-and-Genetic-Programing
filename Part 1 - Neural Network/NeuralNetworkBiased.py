import numpy as np
import math


class Neural_Network_Biased:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights,
                 hidden_layer_biases, output_layer_biases, learning_rate):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.hidden_layer_biases = hidden_layer_biases
        self.output_layer_biases = output_layer_biases

        self.learning_rate = learning_rate

    # Calculate neuron activation for an input
    def sigmoid(self, input_value):
        output = 1 / (1 + math.pow(math.e, -input_value))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(self.num_inputs):
                weighted_sum = weighted_sum + inputs[j] * self.hidden_layer_weights[j][i]
            weighted_sum = weighted_sum + self.hidden_layer_biases[i]
            output = self.sigmoid(weighted_sum)
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for j in range(self.num_hidden):
                weighted_sum = weighted_sum + hidden_layer_outputs[j] * self.output_layer_weights[j][i]
            weighted_sum = weighted_sum + self.output_layer_biases[i]
            output = self.sigmoid(weighted_sum)
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        #  Calculate output layer betas.
        output_layer_betas = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            output_layer_betas[i] = desired_outputs[i] - output_layer_outputs[i]
        # print('OL betas: ', output_layer_betas)

        # Calculate hidden layer betas.
        hidden_layer_betas = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                hidden_layer_betas[i] = hidden_layer_betas[i] + self.output_layer_weights[i][j] * output_layer_outputs[
                    j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]
        # print('HL betas: ', hidden_layer_betas)

        # Following section calculates the amount to change every single weight and bias by

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[
                    j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * hidden_layer_outputs[j] * (
                        1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]

        delta_output_layer_biases = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            delta_output_layer_biases[i] = self.learning_rate * output_layer_outputs[i] * (
                    1 - output_layer_outputs[i]) * output_layer_betas[i]

        delta_hidden_layer_biases = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            delta_hidden_layer_biases[i] = self.learning_rate * hidden_layer_outputs[i] * (
                    1 - hidden_layer_outputs[i]) * hidden_layer_betas[i]

        # Return the weights and biases we calculated, so they can be used to update all the weights and biases.
        return delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_layer_biases, delta_output_layer_biases

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_layer_biases,
                       delta_output_layer_biases):
        # Change every weight and bias
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                self.hidden_layer_weights[i][j] = self.hidden_layer_weights[i][j] + delta_hidden_layer_weights[i][j]
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                self.output_layer_weights[i][j] = self.output_layer_weights[i][j] + delta_output_layer_weights[i][j]
        for i in range(self.num_outputs):
            self.output_layer_biases[i] = self.output_layer_biases[i] + delta_output_layer_biases[i]

        for i in range(self.num_hidden):
            self.hidden_layer_biases[i] = self.hidden_layer_biases[i] + delta_hidden_layer_biases[i]

    def train(self, instances, desired_outputs, epochs):
        # train the neural network
        converged = False
        for epoch in range(epochs):
            if not converged:
                if (epoch+1) % 5 == 0 or epoch < 5:
                    print()
                    print('Epoch: ', epoch + 1)
                predictions = []
                for i, instance in enumerate(instances):
                    hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)

                    # Code for printing the first instance classification after the first feedforward pass and backpropagation
                    if (epoch == 0) and (i == 0):
                        print("Raw Values of first forward pass: ", output_layer_outputs)
                        initial_prediction = self.predict([instance])
                        if initial_prediction[0] == 0:
                            print("Predicted Class: Adelie")
                        elif initial_prediction[0] == 1:
                            print("Predicted Class: Gentoo")
                        elif initial_prediction[0] == 2:
                            print("Predicted Class: Chinstrap")

                    delta_output_layer_weights, delta_hidden_layer_weights, delta_hidden_layer_biases, delta_output_layer_biases, = self.backward_propagate_error(
                        instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])

                    predicted_class = self.predict([instance])
                    predictions.append(predicted_class)

                    # We use online learning, i.e. update the weights after every instance.
                    self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights,
                                        delta_hidden_layer_biases,
                                        delta_output_layer_biases)

                    # Code for printing the weights after the first backpropagation update
                    if (epoch == 0) and (i == 0):
                        print('Weights after performing BP for first instance only:')
                        print('Hidden layer weights:\n', self.hidden_layer_weights)
                        print('Output layer weights:\n', self.output_layer_weights)

                # Print new weights
                # print('Hidden layer weights \n', self.hidden_layer_weights)
                # print('Output layer weights  \n', self.output_layer_weights)

                num_correct = 0
                for i in range(len(predictions)):
                    actual_class = np.where(desired_outputs[i] == np.amax(desired_outputs[i]))[0]
                    if predictions[i] == actual_class:
                        num_correct = num_correct + 1

                acc = num_correct / len(predictions)
                if epoch != 0 and ((epoch+1) % 5 == 0 or epoch < 5):
                    print('Accuracy: %0.2f%%' % (100 * acc))
                if acc == 1:
                    converged = True
                    print("The network has converged on epoch ", epoch)

                if epoch != 0 and ((epoch+1) % 5 == 0 or epoch < 5):
                    print('Hidden layer weights:\n', self.hidden_layer_weights)
                    print('Output layer weights:\n', self.output_layer_weights)

    def predict(self, instances):
        # Predict the results for every instance
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            # print(output_layer_outputs)
            maximum = max(output_layer_outputs)
            predicted_class = output_layer_outputs.index(maximum)
            predictions.append(predicted_class)
        return predictions
