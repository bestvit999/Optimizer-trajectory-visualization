import numpy as np
import math

class SGD:

    def __init__(self, lr=0.01, *args, **kwargs):
        self.lr = lr
        self.num_particle = 1
        self.name = "SGD"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        params = params - self.lr * grads
        """ for track """
        self.best = params
        return params



class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.num_particle = 1
        self.name = "Momentum"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros(params.shape)
        
        self.v = self.momentum * self.v - self.lr * grads
        params += self.v
        """ for track """
        self.best = params
        return params
        # if self.v is None:
        #     self.v = {}
        #     for key, val in params.items():                                
        #         self.v[key] = np.zeros_like(val)
                
        # for key in params.keys():
        #     self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
        #     params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.num_particle = 1
        self.name = "Nesterov"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros(params.shape)
        
        params += self.momentum * self.momentum * self.v
        params -= (1 + self.momentum) * self.lr * grads
        self.v *= self.momentum
        self.v -= self.lr * grads
        """ for track """
        self.best = params
        return params
        # if self.v is None:
        #     self.v = {}
        #     for key, val in params.items():
        #         self.v[key] = np.zeros_like(val)
            
        # for key in params.keys():
        #     params[key] += self.momentum * self.momentum * self.v[key]
        #     params[key] -= (1 + self.momentum) * self.lr * grads[key]
        #     self.v[key] *= self.momentum
        #     self.v[key] -= self.lr * grads[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01, *args, **kwargs):
        self.lr = lr
        self.h = None
        self.num_particle = 1
        self.name = "Adagrad"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = np.zeros(params.shape)
        
        self.h += np.multiply(grads, grads)
        params -= self.lr * grads / (np.sqrt(self.h) + 1e-7)
        """ for track """
        self.best = params
        return params
        # if self.h is None:
            # self.h = {}
            # for key, val in params.items():
            #     self.h[key] = np.zeros_like(val)
            
        # for key in params.keys():
        #     self.h[key] += grads[key] * grads[key]
        #     params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99, *args, **kwargs):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        self.num_particle = 1
        self.name = "RMSprop"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = np.zeros(params.shape)
        
        self.h *= self.decay_rate
        self.h += (1 - self.decay_rate) * np.multiply(grads, grads)
        params -= self.lr * grads / (np.sqrt(self.h) + 1e-7)
        """ for track """
        self.best = params
        return params

        # if self.h is None:
        #     self.h = {}
        #     for key, val in params.items():
        #         self.h[key] = np.zeros_like(val)
            
        # for key in params.keys():
        #     self.h[key] *= self.decay_rate
        #     self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
        #     params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, correct=True, *args, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.correct = correct
        self.num_particle = 1
        self.name = "Adam"
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros(params.shape)
            self.v = np.zeros(params.shape)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        self.m = self.beta1*self.m + (1-self.beta1)*grads
        self.v = self.beta2*self.v + (1-self.beta2)*(grads**2)
        # self.m += (1 - self.beta1) * (grads - self.m)
        # self.v += (1 - self.beta2) * (grads**2 - self.v)

        hat_m = self.m / (1 - self.beta1 ** self.iter)
        hat_v = self.v / (1 - self.beta2 ** self.iter)
        
        if self.correct:
            params -= lr_t * hat_m / (np.sqrt(hat_v) + 1e-7)
        else:
            params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
        """ for track """
        self.best = params
        return params
        
class SA:
    def __init__(self, lr=0.001, cool_rate=0.999, temperature=1000, *args, **kwargs):
        self.lr = lr
        self.cool_rate = cool_rate
        self.temperature = temperature
        self.best = None
        self.obj_func = kwargs['obj_func']
        self.num_particle = 1
        self.name = "SA"
        
    def update(self, params, grads):
        if self.best is None:
            self.best = np.zeros(params.shape)
            self.best = params # inital position is `current best`
        
        """ transit """
        step = np.random.normal(loc=0.0, scale=1.0, size=params.shape)
        # (TODO) : clip
        tweak = params + np.random.random() * step
        """ have a probability to accept worse solution """
        tweak_fit = self.obj_func(tweak)
        params_fit = self.obj_func(params)
        if tweak_fit < params_fit or np.random.random() < np.exp((params_fit - tweak_fit) / self.temperature):
            params = tweak
        """ decrease temp """
        self.temperature *= self.cool_rate
        """ determine best """
        params_fit = self.obj_func(params)
        best_fit = self.obj_func(self.best)
        if params_fit < best_fit:
            self.best = params
        
        return self.best
        # return params

import copy
class PSO:
    class particle:
        def __init__(self, params):
            self.position = params + np.random.normal(0.0, 1.0, params.shape)
            self.velocity = np.random.normal(loc=0.0, scale=1.0, size=params.shape)
            self.fit = None
        def _update(self, movement):
            self.position += movement

    def __init__(self, params, lr=0.001, num_particle=10, *args, **kwargs):
        self.lr = lr
        self.num_particle = num_particle
        self.pop = [self.particle(params) for i in range(num_particle)] # initial population
        self.pop_fit = [None for i in range(num_particle)]
        self.p_best = [None for i in range(num_particle)]
        self.g_best = None
        self.obj_func = kwargs['obj_func']
        self.name = "PSO"
        """ create population """
        self.g_best = self.particle(params)
        self.g_best.fit = self.obj_func(self.g_best.position)
        for i in range(len(self.p_best)):
            self.pop[i].fit = self.obj_func(self.pop[i].position)
            self.p_best[i] = copy.deepcopy(self.pop[i])
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        w = np.random.uniform(0.4, 0.9)
        a = 0.8
        b = 1.6
        """ transit """
        for i in range(len(self.pop)):
            # new_pos += (w * v) + (r_1 * a * (p_best - pos)) + (r_2 * b * (g_best - pso))
            step = np.random.normal(loc=0.0, scale=1.0, size=params.shape)
            old_pos = self.pop[i].position
            self.pop[i]._update(w * self.pop[i].velocity +
                                np.random.random() * a * step +
                                np.random.random() * b * (self.g_best.position - self.pop[i].position))
            # self.pop[i]._update(w * self.pop[i].velocity +
            #                     np.random.random() * a * (self.p_best[i].position - self.pop[i].position) +
            #                     np.random.random() * b * (self.g_best.position - self.pop[i].position))
            """ evaluate all pop """
            self.pop[i].fit = self.obj_func(self.pop[i].position)
            
            """ update velocity """
            self.pop[i].velocity = self.pop[i].position - old_pos

            """ update personal best """
            if self.pop[i].fit < self.p_best[i].fit:
                self.p_best[i].fit = self.pop[i].fit

        """ update global best """
        current_best = sorted(self.pop, key = lambda particle : particle.fit)[0]
        current_best.fit = self.obj_func(current_best.position)
        if current_best.fit < self.g_best.fit:
            self.g_best = copy.deepcopy(current_best)
        
        """ return """
        self.best = self.g_best.position
        return self.g_best.position
        # return current_best.position

class GLAdam:

    """GLAdam """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9, total_iter=10000, correct=True, *args, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.iter = 0
        self.m = None
        self.v = None
        self.g_pos = None
        self.g = None
        self.g_fit = None
        self.l_pos = None
        self.l = None
        self.correct = correct
        self.num_particle = 1
        self.total_iter = total_iter
        self.obj_func = kwargs['obj_func']
        self.name = "GLAdam"
        # init
        self.m = np.zeros(params.shape)
        self.v = np.zeros(params.shape)
        self.g_pos = copy.deepcopy(params)
        self.g = np.random.random(params.shape) # momentum
        self.g_fit = self.obj_func(self.g_pos)
        self.l_pos = copy.deepcopy(params)
        self.l = np.zeros(params.shape) # adaptive
        """ for track """
        self.best = None
        
    def update(self, params, grads):
        
        """ Adam """
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        lambda_t = 1.5 + (self.iter / self.total_iter) * (1.999 - 1.5)
        
        self.m = self.beta1*self.m + (1-self.beta1)*grads
        self.v = self.beta2*self.v + (1-self.beta2)*(grads**2)
        # self.m += (1 - self.beta1) * (grads - self.m)
        # self.v += (1 - self.beta2) * (grads**2 - self.v)

        hat_m = self.m / (1 - self.beta1 ** self.iter)
        hat_v = self.v / (1 - self.beta2 ** self.iter)
        
        if self.correct:
            adam_move = hat_m / (np.sqrt(hat_v) + 1e-7)
        else:
            adam_move = self.m / (np.sqrt(self.v) + 1e-7)

        """ global movement """
        g_move = self.g_pos - params  # movement from `current` toward `global_best_position`
        self.g = self.beta3 * self.g + (1 - self.beta3) * g_move

        """ levy flight """
        sigma1 = np.power((math.gamma(1 + lambda_t) * np.sin((np.pi * lambda_t) / 2)) \
                    / math.gamma((1 + lambda_t) / 2) * np.power(2, (lambda_t - 1) / 2), 1 / lambda_t)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=grads.shape)
        v = np.random.normal(0, sigma2, size=grads.shape)
        levy = u / np.power(np.fabs(v), 1 / lambda_t)
        
        self.l = self.beta2 * self.l + (1 - self.beta2) * np.abs(params - self.l_pos)
        self.l_pos = copy.deepcopy(params)

        """ evaluate, determine, and update """
        alpha = (self.iter / self.total_iter)
        new_pos = params - lr_t * adam_move + levy + self.g # GLAdam
        # new_pos = params + levy + self.g # GL
        # new_pos = params - lr_t * adam_move + lr_t * levy # LAdam
        # new_pos = params - lr_t * adam_move # Adam
        # new_pos = params + (1 - alpha) * lr_t * levy # Levy
        if self.obj_func(new_pos) < self.g_fit:
            self.g_pos = new_pos
            self.g_fit = self.obj_func(new_pos)
        
        # print(self.g)
        """ for track """
        self.best = self.g_pos
        # return new_pos
        return self.g_pos