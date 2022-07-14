import numpy as np

class BaseLayer:
    def __init__(self,trainable=False):
        self.trainable = trainable
        self.weights =[]