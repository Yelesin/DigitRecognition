import numpy as np
import matplotlib.pyplot as plt
import scipy.special

class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        self.wih = np.random.normal(0.0, pow(self.inputNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.who = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        self.learningRate = learningRate
        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # Входи прихованого шару
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        # Входи вихідного шару
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # Помилка на виході
        output_error = targets - final_outputs
        # Помилка прихованого шару
        hidden_errors = np.dot(self.who.T, output_error)

        # Оновлення вагів між прихованим і вихідним шарами
        self.who += self.learningRate * np.dot((output_error * final_outputs * (1.0 - final_outputs)),
                                               np.transpose(hidden_outputs))
        # Оновлення вагів між вхідним і прихованим шарами
        self.wih += self.learningRate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                               np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        return final_outputs


if __name__ == "__main__":
    inp = 784  # Вхідний шар (28x28 пікселів)
    hidden = 100
    out = 10  # Вихідний шар (10 класів для цифр 0-9)
    lr = 0.3

    n = NeuralNetwork(inp, hidden, out, lr)

    # Читання даних
    data_file = open("mnist_train_100.csv", "r")
    data_list = data_file.readlines()
    data_file.close()
    epochs = 6

    for e in range(epochs):
        for record in data_list:
            all_values = record.split(',')
            inputs = (np.array(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01
            targets = np.zeros(out) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # Тестування моделі
    test_data_file = open("mnist_test_10.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.array(all_values[1:], dtype=np.float32) / 255.0 * 0.99) + 0.01

        outputs = n.query(inputs)
        label = np.argmax(outputs)
        scorecard.append(1 if label == correct_label else 0)

    print("Ефективність = ", sum(scorecard) / len(scorecard))
