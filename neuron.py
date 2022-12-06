import numpy as np


class Neuron:

    def __init__(self, weights, threshold, decay=0.9):
        """
        Parameters
        -----------
        - weights: [*weight_inputs, *weight_neurons]. (np.array)
        - threshold: int
        - decay: ent
        """
        self.weights = weights
        self.threshold = threshold
        self.decay = decay

        self.potential = 0

    def __call__(self, neuron_input):
        """
        Parameters
        ----------
        - neuron_input: array with all the inputs: [*inputs_value, *neurons_value]. (np.array)
        """
        # GET INPUT VALUE
        input_value = np.dot(self.weights, neuron_input.T)

        # UPDATE POTENTIAL
        self.update_potential(input_value)

        # GET OUTPUT VALUE
        # WHEN SPIKE
        if self.potential >= self.threshold:
            self.reset()
            return 1

        # WHEN NO SPIKE
        self.decay_potential()
        return 0

    def update_potential(self, input_value):
        self.potential += input_value

    def decay_potential(self):
        self.potential = self.potential * self.decay

    def reset(self):
        self.potential = 0