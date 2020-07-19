# 1. design model (input, output size, foward pass)
# 2. construct loss and optimizer
# 3. traning loop
#   - foward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import numpy as np
import torch
from .RNNBase import RNN_Base

class RNN_LSTM(RNN_Base):
    'This is the long short-time memery version of RNN class'
    'For sentiment recognition'

    def __init__(self, hiddensize, inputsize, lr):
        super(RNN_LSTM, self).__init__()

        self.Hidden_Size = hiddensize
        self.Input_Size = inputsize
        self.Learning_Rate = lr
        
        # model parameters
        self.Wf = torch.randn(self.Hidden_Size, self.Input_Size, requires_grad=True)
        self.Wi = torch.randn(self.Hidden_Size, self.Input_Size, requires_grad=True)
        self.Wa = torch.randn(self.Hidden_Size, self.Input_Size, requires_grad=True)
        self.Wo = torch.randn(self.Hidden_Size, self.Input_Size, requires_grad=True)
        
        self.Bf = torch.zeros(self.Hidden_Size, 1, requires_grad=True)
        self.Bi = torch.zeros(self.Hidden_Size, 1, requires_grad=True)
        self.Ba = torch.zeros(self.Hidden_Size, 1, requires_grad=True)
        self.Bo = torch.zeros(self.Hidden_Size, 1, requires_grad=True)

    def Epoch(self, input_x, output_y, h_pre):
        'whole pass'
        Input_X = torch.tensor(input_x)
        Input_Y = outputy

        C_pre = torch.zeros(self.Hidden_Size, 1)
        H_pre = h_pre
        
        # forward pass
        for t in range(len(Input_X)):
            HX = torch.cat(Input_X, H_pre)
            Ft = torch.sigmoid(torch.mm(self.Wf, HX) + self.Bf)
            It = torch.sigmoid(torch.mm(self.Wi, HX) + self.Bi)
            At = torch.tanh(torch.mm(self.Wa, HX) + self.Ba)
            Ct = torch.mm(C_pre, Ft) + torch.mm(It, At)
            Ot = torch.sigmoid(torch.mm(self.Wo, HX) + self.Bo)
            Ht = Ot * torch.tanh(Ct)
            Cpre = Ct

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
