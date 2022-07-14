import numpy as np

class Constant:

    def __init__(self,val=0.1):
        self.val=val
    
    def initialize(self,weights_shape,fan_in,fan_out):
        inp_tensor = np.zeros(weights_shape) + self.val
        return inp_tensor

class UniformRandom:
    def initialize(self,weights_shape,fan_in,fan_out):
        inp_tensor = np.random.uniform(0, 1, size=weights_shape)
        return inp_tensor

class Xavier:
    def initialize(self,weights_shape,fan_in,fan_out):
        sig = np.sqrt(2/(fan_in+fan_out))
        inp_tensor = np.random.normal(0, sig, size=weights_shape)
        return inp_tensor

class He:
    def initialize(self,weights_shape,fan_in,fan_out):
        sig = np.sqrt(2 / fan_in)
        inp_tensor = np.random.normal(0, sig, size=weights_shape)
        return inp_tensor
