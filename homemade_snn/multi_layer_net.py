import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:
    """
    Neural Network with multi hidden layers.
    """

    def __init__(self, inputnodes, hiddenlayers, outputnodes, learningrate):
        # Set constants

        self.layers = [inputnodes] + hiddenlayers + [outputnodes]
        self.weights = []

        self.lr = learningrate

        # Generate weight matrices for each layer-to-layer
        for i, layer in enumerate(self.layers[1:]):
            self.weights.append(numpy.random.normal(0.0, pow(self.layers[i], -0.5), (layer, self.layers[i])))

        # Set activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Convert the inputs list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        outputs = [inputs]

        for weights in self.weights:
            weighted_inputs = numpy.dot(weights, outputs[-1])
            outputs.append(self.activation_function(weighted_inputs))

        
        output_errors = targets - outputs[-1]
        errors = [output_errors]

        for i, output in enumerate(reversed(outputs[1:])):

            errors.append(numpy.dot(self.weights[-(i+1)].T, errors[i]))
            x = self.lr * numpy.dot((errors[-2] * output * (1 - output)), numpy.transpose(outputs[-(i+2)]))
            self.weights[-(i+1)] += x

    def query(self, inputs_list):
        # Convert the inputs list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        outputs = [inputs]

        for weights in self.weights:
            weighted_inputs = numpy.dot(weights, outputs[-1])
            outputs.append(self.activation_function(weighted_inputs))

        return outputs[-1]


def main():
    # Set constants
    input_nodes = 784
    hidden_layers = [150, 150]
    output_nodes = 10
    learning_rate = .3

    # Create neural network
    global n
    n = NeuralNetwork(input_nodes, hidden_layers, output_nodes, learning_rate)

    # Open and parse training data into input nodes
    training_data_file = open("homemade_snn/mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # Train the neural network
    for i, record in enumerate(training_data_list):
        # Print progress
        if (i % 100 == 0):
            print(i)

        # Get pixel values, scaled to [.01, 1.0]
        all_values = record.split(',')
        scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
        # Create output nodes, scaled to [.01, .99]
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        # Train the network on this (input, output) set
        n.train(scaled_input, targets)

    # Test the neural network
    test_data_file = open("homemade_snn/mnist_dataset/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    
    for i, record in enumerate(test_data_list):
        # Print progress
        if (i % 100 == 0):
            print(i)
        
        # Get correct answer and pixel values scaled to [.01, 1.0]
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        # Query the network and get most probable choice
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        # print(label, "network's answer")

        # Append to list whether it was correct or not
        scorecard.append(int(label == correct_label))
    
    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)

if __name__ == "__main__":
    main()
