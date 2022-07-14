from copy import deepcopy
import sys
from Layers.Base import BaseLayer
from scipy.signal import correlate, convolve
import numpy as np


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__(trainable=True)
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.Dim1D = False
        if len(convolution_shape) == 2:
            self.convolution_shape += (1,)
            self.stride_shape += (1,)
            self.Dim1D = True

        elif len(convolution_shape) == 3:
            if len(stride_shape) == 1:
                self.stride_shape = (stride_shape, stride_shape)

        else:
            raise ValueError("Please verify Dimensions! There may be a Problem.")
        self.weights = np.random.uniform(size=(self.num_kernels,) + self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None

        self._optimizers = None

        self.input_size = None
        self.inpsize_padd = None
        self.input_tensor = None
        self.is_padded = None

        self.os_new = None

        self.padd = None
        self.list_padded = None

    def forward(self, input_tensor):
        self.input_size = input_tensor.shape
        self.input_tensor = input_tensor
        if self.Dim1D:  # 1D
            self.input_size += (1,)
            self.input_tensor = input_tensor[:, :, :, np.newaxis]
        self.os_new = (self.input_size[0], self.num_kernels,
                                       self.input_size[2], self.input_size[3])
        self.padd = (self.convolution_shape[1] - 1, self.convolution_shape[2] - 1)
        self.list_padded = [(0, 0), (0, 0),
                             (np.ceil(self.padd[0] / 2).astype(int), np.floor(self.padd[0] / 2).astype(int)),
                             (np.ceil(self.padd[1] / 2).astype(int), np.floor(self.padd[1] / 2).astype(int))]

        self.is_padded = np.pad(self.input_tensor, self.list_padded, mode="constant", constant_values=0)
        self.inpsize_padd = self.is_padded.shape

        without_stride = np.zeros(self.os_new)
        for b in range(self.os_new[0]):
            for k in range(self.num_kernels):
                without_stride[b, k, :, :] = correlate(self.is_padded[b, :, :, :],
                                                      self.weights[k, :, :, :], mode='valid') + self.bias[k]

        final = without_stride[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        if self.Dim1D: final = final[:, :, :, 0]

        return final

    def backward(self, error_tensor):
        if self.Dim1D: error_tensor = error_tensor[:, :, :, np.newaxis]
        grad_inp = np.zeros(self.input_size)
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        errs = np.zeros(self.os_new)
        errs[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        errs_with_padd = np.pad(errs, self.list_padded, mode="constant", constant_values=0)
        final_shape_kernel = (self.convolution_shape[0], self.num_kernels,
                         self.convolution_shape[1], self.convolution_shape[2])

        final_w = np.zeros(final_shape_kernel)

        for c in range(self.convolution_shape[0]):
            for k in range(self.num_kernels):

                final_w[c, k, :, :] = self.weights[k, c, :, :]


            for b in range(self.input_size[0]):

                final_flip = np.flipud(final_w[c, :, :, :])
                grad_inp[b, c, :, :] = convolve(errs_with_padd[b, :, :, :], final_flip, mode="valid")


        self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))


        for b in range(self.input_size[0]):
            for c in range(self.input_size[1]):
                for k in range(self.num_kernels):

                    self.gradient_weights[k, c, :, :] += correlate(self.is_padded[b, c, :, :],
                                                                   errs[b, k, :, :], mode="valid")


        if self._optimizers is not None:
            self.weights = self._optimizers[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizers[1].calculate_update(self.bias, self.gradient_bias)

        if self.Dim1D: grad_inp = grad_inp[:, :, :, 0]

        return grad_inp

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(list(self.convolution_shape))
        fan_out = np.prod([self.num_kernels] + list(self.convolution_shape[1:]))
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, fan_out)

    @property
    def optimizer(self):
        return self._optimizers

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizers = [optimizer, deepcopy(optimizer)]

    @optimizer.getter
    def optimizer(self):
        return self._optimizers
