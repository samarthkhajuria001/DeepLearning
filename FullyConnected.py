import numpy as np
from Layers.Base import BaseLayer
import sys
from Optimization.Optimizers import Sgd


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__(trainable=True)
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.weights_temp = np.random.uniform(0.0, 1.0, (self.input_size, self.output_size))
        self.bias = np.random.uniform(0.0, 1.0, (1, self.output_size))
        self.weights = np.vstack((self.weights_temp, self.bias))
        self.input_tensor = None

    def forward(self, input_tensor):
        input_tensor_temp = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        self.input_tensor = input_tensor_temp
        output = np.dot(input_tensor_temp, self.weights)

        return output

    def backward(self, error_tensor):
        self.weights_dw = np.dot(self.input_tensor.T, error_tensor)

        output_tensor = np.dot(error_tensor, self.weights.T)

        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.weights_dw)

        return output_tensor[:, 0:-1]

    def initialize(self, weights_initializer, bias_initializer):

        fan_in  = self.input_size
        fan_out = self.output_size
        self.weights[:self.input_size,:] = weights_initializer.initialize((fan_in, fan_out), fan_in, fan_out)
        self.weights[self.input_size:,:] = bias_initializer.initialize((1, fan_out), 1, fan_out)

    @property
    def gradient_weights(self):
        return self.weights_dw

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    
