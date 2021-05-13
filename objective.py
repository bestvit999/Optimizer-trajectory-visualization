import autograd.numpy as np  # Thinly-wrapped numpy
from autograd.numpy import exp, cos, sqrt
class objectiveFunction:
    def __init__(self):
        self.name = None
        self.init_pos = None
        self.best_pos = None
        self.gp = None
        self.xrange = None
        self.yrange = None
        self.zrange = None
    
    def obj_func(x):
        pass

class rosenbrock(objectiveFunction):
    def __init__(self):
        self.name = 'rosenbrock'
        self.best_pos = '1. 1. 0.'
        self.gp = 'f(x, y) = 100*(y - x**2)**2 + (1 - x)**2'
        self.init_pos = [-2.2, 4.2]
        self.xrange = '[-3:3]'
        self.yrange = '[-1:5]'
        self.zrange = '[0:10000]'

    def obj_func(self, x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

class ackley(objectiveFunction):
    def __init__(self):
        self.name = 'ackley'
        self.best_pos = '0. 0. 0.'
        self.gp = 'f(x, y) = -20 * exp( -0.2 * sqrt(x ** 2 + y ** 2) / 2) - exp((cos(2 * 3.14 * x)+ cos(2 * 3.14 * y)) / 2) + 20 + exp(1)'
        self.init_pos = [-4.5, 4.2]
        self.xrange = '[-5:5.]'
        self.yrange = '[-5:5.]'
        self.zrange = '[0:100]'
    
    def obj_func(self, x):
        return -20 * exp( -0.2 * sqrt(x[0] ** 2 + x[1] ** 2) / 2) - exp((cos(2 * np.pi * x[0])+ cos(2 * np.pi * x[1])) / 2) + 20 + exp(1)

class sphere(objectiveFunction):
    def __init__(self):
        self.name = 'sphere'
        self.best_pos = '0. 0. 0.'
        self.gp = 'f(x, y) = x ** 2 + y ** 2'
        self.init_pos = [-4.5, 4.2]
        self.xrange = '[-5:5.]'
        self.yrange = '[-5:5.]'
        self.zrange = '[0:100]'

    def obj_func(self, x):
        return x[0] ** 2 + x[1] ** 2

class zakharov(objectiveFunction):
    def __init__(self):
        self.name = 'zakharov'
        self.best_pos = '0. 0. 0.'
        self.gp = 'f(x, y) = x ** 2 + y ** 2 + (0.5 * 1 * x + 0.5 * 2 * y) ** 2 + (0.5 * 1 * x + 0.5 * 2 * y) ** 4'
        self.init_pos = [-4.5, 4.2]
        self.xrange = '[-5:5.]'
        self.yrange = '[-5:5.]'
        self.zrange = '[0:100]'
    
    def obj_func(self, x):
        return x[0] ** 2 + x[1] ** 2 + \
               (0.5 * 1 * x[0] + 0.5 * 2 * x[1]) ** 2 + \
               (0.5 * 1 * x[0] + 0.5 * 2 * x[1]) ** 4

class booth(objectiveFunction):
    def __init__(self):
        self.name = 'booth'
        self.best_pos = '1. 3. 0.'
        self.gp = 'f(x, y) = (x + 2 * y  - 7)**2 + (2 * x  + y - 5) ** 2'
        self.init_pos = [-4.5, 4.2]
        self.xrange = '[-7.5:9.5]'
        self.yrange = '[-6:12.]'
        self.zrange = '[0:100]'
    
    def obj_func(self, x):
        return (x[0] + 2 * x[1]  - 7)**2 + (2 * x[0]  + x[1] - 5) ** 2

