from turtle import forward
import numpy as np
import math
class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor= None

    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        prediction_tensor_log = np.log(self.prediction_tensor + np.finfo(np.float64).eps)
        loss = np.sum(label_tensor*prediction_tensor_log)
        return -loss

    def backward(self,label_tensor):

        prediction_tensor = self.prediction_tensor+np.finfo(np.float64).eps
        error_tensor = np.array([x/y for (x,y) in zip(label_tensor,prediction_tensor)]).reshape(label_tensor.shape)
        return -error_tensor





