import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__(False)
        self.input_tensor=None

    def forward(self,input_tensor):
        self.input_tensor =input_tensor

        exp_values = np.exp(self.input_tensor -np.max(self.input_tensor ))
        self.output = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        return self.output


    def backward(self,output_tensor):
        #implement backward prop
        output = self.output
        x = output_tensor * output

        E_passed = output * (output_tensor - np.sum(x,axis=1 , keepdims=True))

        return E_passed
        # return output_tensor*(self.output*(1-self.output))
        
