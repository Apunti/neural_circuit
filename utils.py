import matplotlib.pyplot as plt


class History:

    def __init__(self, df):
        self.df = df

    def plot_neuron(self, neuron_number):
        column_name = f"neuron_{neuron_number}"
        self.plot_column(column_name)

    def plot_source(self):
        column_name = "source"
        self.plot_column(column_name)

    def plot_column(self, column_name, label=False):
        y = self.df[column_name]
        x = range(len(y))
        if label:
            plt.plot(x, y, label=column_name)
        else:
            plt.plot(x, y)

    def plot_neurons(self):
        column_names = [column for column in self.df.columns if "neuron" in column]
        plt.figure()
        for column_name in column_names:
            self.plot_column(column_name, label=True)
        plt.legend()
