
import numpy as np
from turtle import forward
import copy
from Optimization.Loss import CrossEntropyLoss


class NeuralNetwork:

    def __init__(self,optimizer,weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer    = bias_initializer
        



    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        output = self.input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return self.loss_layer.forward(output, self.label_tensor)
    

    def backward(self):
        output = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            output = layer.backward(output)
        
        return output[:, 0:-1]
    
    def append_layer(self,layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)


    def train(self,iterations):
        for i in range(iterations):
            loss =  self.forward()
            self.loss.append(loss)
            self.backward()


    def test(self, input_tensor):

        output = input_tensor  
        for layer in self.layers: 
            output = layer.forward(output)

        return output

            



         
