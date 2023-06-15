import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:
    """
    Neural Network designed to find vertical vs. horizontal lines.
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set constants
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Set activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        # Create weight matrices
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    def train(self, inputs_list, targets_list):
        # Convert the inputs list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Calculate signals in and out of hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals in and out of final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Find error (target-actual)
        output_errors = targets - final_outputs

        # Find hidden layer error (output_error split by weight recombined at nodes)
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update weights hidden -> output
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))

        # Update weights input -> hidden
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # Convert the inputs list into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals in and out of hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals in and out of final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def main():
    # Set constants
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = .3

    # Create neural network
    global n
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Open and parse training data into input nodes
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
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
    test_data_file = open("mnist_dataset/mnist_test.csv", "r")
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
