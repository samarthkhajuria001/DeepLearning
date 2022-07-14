from Layers.Base import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__(trainable=False)
        self.input_size = None
        

    def forward(self, input_tensor):

        self.input_size = input_tensor.shape


        new_shape = None
        if len(input_tensor.shape) == 3:
            b, c, m = input_tensor.shape
            new_shape = (b, c * m)
        elif len(input_tensor.shape) == 4:
            b, c, m, n = input_tensor.shape
            new_shape = (b, c * m * n)

        final_input = np.reshape(input_tensor, newshape=new_shape)
        return final_input

    def backward(self, error_tensor):
        final_error = np.reshape(error_tensor, newshape=self.input_size)
        return final_error
    