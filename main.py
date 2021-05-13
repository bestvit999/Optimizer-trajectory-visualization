from autograd import value_and_grad
from optimizer import *
from objective import *
import copy

# objs = [rosenbrock(), ackley(), sphere(), zakharov(), booth()]
objs = [rosenbrock(), sphere(), zakharov(), booth()]
for obj in objs:
    obj_with_grad = value_and_grad(obj.obj_func)

    """ initial position """
    init_pos = np.array(obj.init_pos) # init position
    total_iter = 10000
    """ optimizers """
    opts = {
        "SGD" : SGD(lr=.001),
        "Momentum" : Momentum(lr=.001, momentum=0.9),
        "Nesterov" : Nesterov(lr=.001, momentum=0.9),
        "Adagrad" : AdaGrad(lr=.1),
        "RMSprop" : RMSprop(lr=.1, decay_rate=0.99),
        "Adam" : Adam(lr=.1, beta1=0.9, beta2=0.999),
        "SA" : SA(lr=.01, obj_func=obj.obj_func),
        "PSO" : PSO(lr=.01, params = init_pos, num_particle=10, obj_func=obj.obj_func),
        "GLAdam" : GLAdam(lr=.01, params = init_pos, total_iter=total_iter, obj_func=obj.obj_func),
    }

    for opt_key in opts:
        pos = copy.deepcopy(init_pos) # init position
        opt = opts[opt_key]

        """ training / minimize """
        trajectory_logs = []
        convergence_logs = []
        for step in range(total_iter//opt.num_particle):
            loss, grads = obj_with_grad(pos) # compute partial derivative
            new_pos = opt.update(pos, grads) # update weights
            pos = new_pos
            trajectory_logs.append([new_pos[0], new_pos[1], loss]) # print jratectory
            convergence_logs.append([opt.best[0], opt.best[1], obj.obj_func(opt.best)]) # print best-so-far loss

        """ save trajectory_logs / convergence_logs """
        import pandas as pd
        df = pd.DataFrame(trajectory_logs)
        df_con = pd.DataFrame(convergence_logs)
        df.to_csv('logs/{obj}/{optimizer}'.format(obj=obj.name, optimizer=opt.name), sep=' ', header=None, index=None)
        df_con.to_csv('convergence/{obj}/{optimizer}'.format(obj=obj.name, optimizer=opt.name), sep=' ', header=None, index=None)

        """ outout figure by GNUplot"""
        import PyGnuplot as gp
        gp.c('set term png size 600, 400 font "arial, 10" fontscale 1.0')
        gp.c('set output "trajectory/{obj}/{optimizer}.png"'.format(obj=obj.name, optimizer=opt.name))
        gp.c("set rmargin at screen 0.8")
        gp.c("set lmargin at screen 0.1")
        gp.c("set xrange {xrange}".format(xrange=obj.xrange))
        gp.c("set yrange {yrange}".format(yrange=obj.yrange))
        gp.c("set zrange {zrange}".format(zrange=obj.zrange))
        gp.c("{gp}".format(gp = obj.gp))
        # gp.c("f(x, y) = 100*(y - x**2)**2 + (1 - x)**2") # rosenbrock
        # gp.c("f(x, y) = -20 * exp( -0.2 * sqrt(x ** 2 + y ** 2) / 2) - exp((cos(2 * 3.14 * x)+ cos(2 * 3.14 * y)) / 2) + 20 + exp(1)") # ackley
        # ---- #
        # gp.c('set pm3d')
        # gp.c('set palette defined (0 "#2cd8f2",1 "#1af0d0",2"#e2f22c")')
        #gp.c('set palette defined (0 "#8ecae6",1 "#b8f2e6",2"#118ab2",3 "#2b9348",4 "#90be6d", 5 "#f9c74f",6 "#f8961e",7 "#ff6b6b",8 "#e01e37",9 "#7251b5")')
        # ---- #
        gp.c("set palette rgbformulae 33,13,10") # color
        gp.c("set isosample 51") # resolution
        gp.c("set view map")
        # g("set key outside")
        # g("set key top left")
        gp.c("splot f(x,y) w pm3d notitle 'rosenbrock', 'logs/{obj}/{optimizer}' u 1:2:(f($1,$2)) w l lc 'black' t '{optimizer}', \"< echo '{best_pos}'\" w point pointtype 9 lc 'white' t 'minimum'".format(obj=obj.name, optimizer=opt.name, best_pos=obj.best_pos))