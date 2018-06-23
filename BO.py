import A3C

import numpy as np
import matplotlib.pyplot as plt

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import os
import time
import csv
import argparse


def main(args):
    space  = [Real(10**-5, 10**-3, "log-uniform", name='initial_lr'),
              Real(10**-7,10**0, "log-uniform", name='entropy'),
              Categorical([0, 1, 2,3], name='size_nn'),
              Integer(3,20,  name='p_len_episode_buffer'),
              [args.cut_10],
              [args.max_no_improvement],
              [args.LT_s],
              [args.OrderFast],
              [args.OrderSlow],
              [args.cap_slow],
              [args.cap_fast],
              [args.C_f],
              [args.b]]

    log_path = 'BOLogs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    res_gp = gp_minimize(A3C.obj_bo, space, n_calls=int(args.iterations), random_state=0,n_jobs=-1)


    print("Best score=%f" % res_gp.fun)

    x_iters = res_gp.x_iters
    func_vals = res_gp.func_vals


    with open(log_path + '/BO_best.csv', 'w') as f:
        f.write(str(res_gp.fun))

    with open(log_path +'/BO_results.csv', 'w') as f:
        for  index,i in enumerate(x_iters):
            for item in i:
                f.write(str(item)+ ';')
            f.write(str(func_vals[index])+';')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-iterations', '--iterations', default=50, type=float,
                        help="Number of hyperparameter sets tested",
                        dest="iterations")
    parser.add_argument('--max_no_improvement', default=2500, type=float, help="max_no_improvement. Default = 5000",
                        dest="max_no_improvement")
    parser.add_argument('--LT_s', default=1, type=int, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=0, type=int, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--cap_slow', default=1, type=float,
                        help="cap_slow. Default = 1",
                        dest="cap_slow")
    parser.add_argument('--cap_fast', default=1, type=float,
                        help="cap_fast. Default = 1",
                        dest="cap_fast")
    parser.add_argument('--C_f', default=150, type=float,
                        help="C_f. Default = 150",
                        dest="C_f")
    parser.add_argument('--b', default=495, type=float,
                        help="b. Default = 495",
                        dest="b")
    parser.add_argument('--cut_10', default=2000, type=float,
                        help="cut_10. Default = 2000",
                        dest="cut_10")
    parser.add_argument('--OrderFast', default=5, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFast")
    parser.add_argument('--OrderSlow', default=5, type=int, help="OrderSlow. Default = 5", dest="OrderSlow")
    args = parser.parse_args()
    main(args)