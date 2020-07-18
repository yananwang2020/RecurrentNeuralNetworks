# 1. design model (input, output size, foward pass)
# 2. construct loss and optimizer
# 3. traning loop
#   - foward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import numpy as np
from pathlib import Path
import pickle

class RNN_Vanilla:
    'This is the simplest RNN class'

    def __init__(self, hiddensize, inputsize, lr):
        self.hidden_size = hiddensize
        self.input_size = inputsize
        self.learning_rate = lr
        
        # model parameters
        self.Wxh = np.random.randn(self.hidden_size, self.input_size)*0.01 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(self.input_size, self.hidden_size)*0.01 # hidden to output
        self.Bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.By = np.zeros((self.input_size, 1)) # output bias

    def LoadParam(self, filepath):
        if filepath.exists():
            with open(filepath, 'rb') as f:
                filedata = pickle.load(f)
                for var in vars(self):
                    setattr(self, var, filedata[var])
                return True

    def SaveParam(self, filepath):        
        with open(filepath, 'wb+') as f:
            pickle.dump(vars(self), f)

    def Epoch(self, inputdata, h_pre):
        'whole pass'
        input_X = inputdata[:-1]
        input_Y = inputdata[1:]
        Xs, Hs, Ss, Ps= {}, {}, {}, {}
        Hs[-1] = np.copy(h_pre)
        Loss = 0

        # forward pass
        for t in range(len(input_X)):
            Xs[t] = np.zeros((self.input_size, 1))
            Xs[t][input_X[t]] = 1
            Hs[t] = np.tanh(np.dot(self.Wxh, Xs[t]) + np.dot(self.Whh, Hs[t-1]) + self.Bh)
            Ss[t] = np.dot(self.Why, Hs[t]) + self.By
            Ps[t] = np.exp(Ss[t]) / np.sum(np.exp(Ss[t])) # softmax function
            Loss += -np.log(Ps[t][input_Y[t]][0]) # softmax loss

        # backward pass
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
        dHnext = np.zeros_like(Hs[0])

        for t in reversed(range(len(input_X))):
            dY = np.copy(Ps[t])
            dY[input_Y[t]] -= 1

            dBy += dY
            dWhy += np.dot(dY, Hs[t].T)
            dH = np.dot(self.Why.T, dY) + dHnext
            dHraw = (1 - Hs[t] * Hs[t]) * dH
            dBh += dHraw
            dWxh += np.dot(dHraw, Xs[t].T)
            dWhh += np.dot(dHraw, Hs[t-1].T)
            dHnext = np.dot(self.Whh.T, dHraw)

        # update weights
        for weight, dweight in zip([self.Wxh, self.Whh, self.Why, self.Bh, self.By],
                                    [dWxh, dWhh, dWhy, dBh, dBy]):
            weight -= self.learning_rate * dweight
        
        return Loss, Hs[len(input_X)-1]

    def Sample(h, seedidx, num):
        x = np.zeros((input_size, 1))
        x[seedidx] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.By)
            y = np.dot(self.Why, h) + self.By
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((input_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
