import time
import torch
import configs
import tensor_utils as utils
from PSO_Algorithm.ChaoticPSO import ChaoticParticleSwarm
from MOPSO_Algorithm.Mopso import *
import numpy as np
import sys


class Logger(object):
    def __init__(self, filename='RCPSO-Pubmed-public-MO.Log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(args):
    torch.manual_seed(args.random_seed)
    sys.stdout = Logger(stream=sys.stdout)
    utils.makedirs(args.dataset)

    # pop = Population(args)
    # pop.evolve_net()
    x_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_max = np.array([6, 3, 7, 5, 6, 6, 3, 7, 5, 6, 6, 3, 3])
    max_vel = np.array([10, 4, 12, 8, 10, 10, 4, 12, 8, 10, 10, 4, 4])

    start = time.time()

    print('-----------------------------------------')
    print('-----------------------------------------')
    pso = ParticleSwarm(args, x_min, x_max, max_vel)
    # pso.update()
    # rcpso = ChaoticParticleSwarm(args, x_min, x_max, max_vel)
    # rcpso.update()
    mopso_ = Mopso(args=args, particles=10, w=0.8, c1=1.2, c2=1.2, max_=x_max, min_=x_min, thresh=20, mesh_div=10)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_=20)  # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n,迭代结束,over")

    end = time.time()
    stamp = end - start
    print("耗时:", stamp)
    print('-----------------------------------------')
    print('-----------------------------------------')


#     # run on single model
#     num_epochs = 200
#     actions = ['gcn', 'mean', 'softplus', 16, 8, 'gcn', 'max', 'tanh', 16, 6] 
#     pop.single_model_run(num_epochs, actions)


if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)
