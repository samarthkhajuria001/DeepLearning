import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        

    def forward(self,input_tensor):
        self.input_tensor=input_tensor

        return np.maximum(0,self.input_tensor)

    def backward(self,error_tensor):
        error_tensor_copy = error_tensor.copy()
        error_tensor_copy[self.input_tensor<=0] = 0
        return error_tensor_copy

