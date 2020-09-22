
from tensorflow import keras
import matplotlib.pyplot as plt
import json
# import IPython.display as display
class PlotLosses(keras.callbacks.Callback):

    def __init__(self, path=None, showGraph=False ):
        self.path = path
        self.showGraph = showGraph

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.history = {}
        self.fig = plt.figure()
        
        self.logs = []
#         plt.show()

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        for l in logs:
            if l not in self.history:
                self.history[l]=[]
            self.history[l].append(logs.get(l))
        self.i += 1
        
#         display.clear_output(wait=True)
        for l in logs:
            plt.plot(self.x, self.history[l], label=l)
        plt.legend()
        if self.path is not None:
            plt.savefig(self.path)
        if self.showGraph:
            plt.show()

class SaveHistory(keras.callbacks.Callback):

    def __init__(self, path ):
        self.path = path

    def on_train_begin(self, logs={}):     
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        for l in logs:
            if l not in self.history:
                self.history[l]=[]
            self.history[l].append(logs.get(l))
        history_v2_dict = self.history
        json.dump(str(history_v2_dict), open(self.path, 'w'))
        
