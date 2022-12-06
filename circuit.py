import numpy as np
import pandas as pd


class Circuit:

    def __init__(self, neurons):
        self.neurons = neurons
        self.t = 0

        self.MAX_ITERATION = 40
        self._history = []

    @property
    def history(self):
        columns = ["source"] + [f"neuron_{n + 1}" for n in range(len(self.neurons))]
        return pd.DataFrame(self._history, columns=columns)

    def __call__(self):
        source_value = self.get_source_value()
        neuron_value = [0] * len(self.neurons)
        state = np.array(source_value + neuron_value)

        while self.t < self.MAX_ITERATION:
            self._history.append(state)
            new_state = self.step(state)
            state = new_state

    def step(self, state):
        neuron_outputs = []
        for neuron in self.neurons:
            neuron_value = neuron(state)
            neuron_outputs.append(neuron_value)

        self.t += 1
        source_value = self.get_source_value()
        return np.array(source_value + neuron_outputs)

    def get_source_value(self):
        return [1] if self.t == 0 else [1]