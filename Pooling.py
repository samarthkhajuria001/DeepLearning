import numpy as np
import sys
from Layers.Base import BaseLayer



class Pooling(BaseLayer):
    
    def __init__(self,stride_shape, pooling_shape):
        super().__init__(trainable=False)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.mask = None

    def forward(self, input_tensor):

        self.mask = {}
        self.input_shape = input_tensor.shape


        output_shape = (self.input_shape[0], self.input_shape[1], # b and c
                        int(1 + (self.input_shape[2] - self.pooling_shape[0]) / self.stride_shape[0]),
                        int(1 + (self.input_shape[3] - self.pooling_shape[1]) / self.stride_shape[1]))

        output = np.zeros(shape=output_shape)


        for y in range(output_shape[2]):
            for x in range(output_shape[3]):

                h = (y * self.stride_shape[0], y * self.stride_shape[0] + self.pooling_shape[0])
                w = (x * self.stride_shape[1], x * self.stride_shape[1] + self.pooling_shape[1])

                orient_slice = input_tensor[:, :, h[0]:h[1], w[0]:w[1]]

                res_sliced = np.reshape(orient_slice, (orient_slice.shape[0], orient_slice.shape[1], -1))
                max_of_v = res_sliced.max(axis=2, keepdims=True)


                output[:, :, y, x] = max_of_v[:, :, 0]


                mask = res_sliced == max_of_v
                self.mask[(y, x)] = mask.reshape(orient_slice.shape)

        return output

    def backward(self, error_tensor):
        gradient = np.zeros(self.input_shape)

        for i in range(error_tensor.shape[2]):
            for j in range(error_tensor.shape[3]):

                h = (i * self.stride_shape[0], i * self.stride_shape[0] + self.pooling_shape[0])
                w = (j * self.stride_shape[1], j * self.stride_shape[1] + self.pooling_shape[1])


                gradient[:, :, h[0]:h[1], w[0]:w[1]] += error_tensor[:, :, i:i+1, j:j+1] * self.mask[(i, j)]

        return gradient