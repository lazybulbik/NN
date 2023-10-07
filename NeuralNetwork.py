import numpy as np


class NeuralNetwork:

    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_neurons, input_neurons))
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_neurons, hidden_neurons))
        self.bias_input_to_hidden = np.zeros((hidden_neurons, 1))
        self.bias_hidden_to_output = np.zeros((output_neurons, 1))

        self.e_loss = 0
        self.e_correct = 0
        self.learning_rate = 0.01

    def train(self, x_train, y_train, epochs=1):
        for epoch in range(epochs):
            print(f"Epoch №{epoch}")

            for image, label in zip(x_train, y_train):
                image = np.reshape(image, (-1, 1))
                label = np.reshape(label, (-1, 1))

                # print(image, ' ', label, end="\n\n")
                # Forward propagation (to hidden layer)
                hidden_raw = self.bias_input_to_hidden + self.weights_input_to_hidden @ image
                hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

                # Forward propagation (to output layer)
                output_raw = self.bias_hidden_to_output + self.weights_hidden_to_output @ hidden
                output = 1 / (1 + np.exp(-output_raw))

                # Loss / Error calculation
                self.e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                self.e_correct += int(np.argmax(output) == np.argmax(label))

                # Backpropagation (output layer)
                delta_output = output - label
                self.weights_hidden_to_output += -self.learning_rate * delta_output @ np.transpose(hidden)
                self.bias_hidden_to_output += -self.learning_rate * delta_output

                # Backpropagation (hidden layer)
                delta_hidden = np.transpose(self.weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
                self.weights_input_to_hidden += -self.learning_rate * delta_hidden @ np.transpose(image)
                self.bias_input_to_hidden += -self.learning_rate * delta_hidden

            # DONE

            # print some debug info between epochs
            print(f"Loss: {round((self.e_loss[0] / x_train.shape[0]) * 100, 3)}%")
            print(f"Accuracy: {round((self.e_correct / x_train.shape[0]) * 100, 3)}%")
            self.e_loss = 0
            self.e_correct = 0

    def go(self, data):
        data = np.reshape(data, (-1, 1))
        hidden_raw = self.bias_input_to_hidden + self.weights_input_to_hidden @ data
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid
        output_raw = self.bias_hidden_to_output + self.weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        return output

    def load_weights(self, weights_path):
        saved_weights = np.load(weights_path, allow_pickle=True)
        self.weights_input_to_hidden = saved_weights.item()['weights_input_to_hidden']
        self.weights_hidden_to_output = saved_weights.item()['weights_hidden_to_output']
        self.bias_input_to_hidden = saved_weights.item()['bias_input_to_hidden']
        self.bias_hidden_to_output = saved_weights.item()['bias_hidden_to_output']

    def upload_weights(self, weights_path):
        saved_weights = {
            'weights_input_to_hidden': self.weights_input_to_hidden,
            'weights_hidden_to_output': self.weights_hidden_to_output,
            'bias_input_to_hidden': self.bias_input_to_hidden,
            'bias_hidden_to_output': self.bias_hidden_to_output
        }
        np.save(weights_path, saved_weights)