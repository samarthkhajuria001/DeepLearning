import numpy as np

class Sgd:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate


    def calculate_update(self,weight_tensor, gradient_tensor):
        updated_weights = weight_tensor- self.learning_rate*gradient_tensor
        return updated_weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu  = mu
        self.rho = rho

        self.vk = None
        self.rk = None

        self.t = 0 

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.vk is None:
            self.vk = np.zeros_like(gradient_tensor)
        if self.rk is None:
            self.rk = np.zeros_like(gradient_tensor)

        self.t += 1

        self.vk = self.mu  * self.vk + (1 - self.mu)  * gradient_tensor

        self.rk = self.rho * self.rk + (1 - self.rho) * gradient_tensor * gradient_tensor 

        v_hat_k = self.vk / (1.0 - self.mu  ** self.t)  
        r_hat_k = self.rk / (1.0 - self.rho ** self.t)  

        eps = np.finfo('float').eps
        weight_tensor -= self.learning_rate * (v_hat_k / (np.sqrt(r_hat_k) + eps))
        return weight_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_k_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v_k_tensor is None:  
            self.v_k_tensor = np.zeros_like(gradient_tensor)
        self.v_k_tensor = self.momentum_rate * self.v_k_tensor - self.learning_rate * gradient_tensor
        weight_tensor += self.v_k_tensor
        return weight_tensor


         
