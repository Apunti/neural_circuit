from circuit import Circuit
from neuron import Neuron
from utils import History

import numpy as np


def main():

    # DEFINE NEURONS
    neuron_weights = [
        np.array([0.8, 0, 0.5, -0.8]),
        np.array([0, 0.5, 0, 0]),
        np.array([0, 0, 0.5, 0])
    ]
    neuron_thresholds = [
        0.6,
        0.6,
        0.6
    ]

    neurons = [Neuron(weights, threshold) for weights, threshold in zip(neuron_weights, neuron_thresholds)]

    # INIT CIRCUIT
    circuit = Circuit(neurons)

    # SIMULATE CIRCUIT
    circuit()

    # PLOT NEURONS VALUES
    history = History(circuit.history)
    history.plot_neurons()


if __name__ == "__main__":
    main()
