import random
import math
from .Particle import Particle


class ParticleSwarm:
    def __init__(self, args, x_min, x_max, max_vel, trainer, logger):
        self.args = args
        self.x_min = x_min
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.trainer = trainer
        self.logger = logger
        self.best_fitness_value = float('-Inf')
        self.best_position = [0.0 for i in range(self.args.particle_dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.genGlobalBest = [[0] * self.args.particle_dim for i in range(self.args.iterations)]  # 每一轮迭代的全局最优位置
    

    def init_particle(self, x):
        # 对种群进行初始化
        self.Particle_list = [Particle(x, self.x_min, self.x_max, self.max_vel, self.args.particle_dim) for i in
                              range(self.args.particle_num)]
        for part in self.Particle_list:
            value = self.get_fitness(part.get_pos())
            part.set_fitness_value(value)


    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    def get_fitness(self, particle):  # 适应函数
        self.trainer.recover_old_parameters()
        stats = {'DIS_COSTS': []}
        stats_G = {'GEN_COSTS': []}
        sum_particle = particle[0] + particle[3] + particle[4]
        particle[0] = particle[0]/sum_particle
        particle[3] = particle[3]/sum_particle
        particle[4] = particle[4]/sum_particle
        for n_iter in range(0, self.args.pso_epoch_size, self.args.batch_size):
            # mapping training (discriminator fooling)
            self.trainer.mapping_step(stats_G, particle)
            # discriminator training
            for i in range(self.args.dis_steps):
                self.trainer.dis_step1(stats)
        val_acc = self.trainer.score()
        
        return val_acc

    # 更新速度
    def update_vel(self, part):
        for i in range(self.args.particle_dim):
            ww = self.args.W * part.get_vel()[i]
            vel_value = self.args.c1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) + self.args.c2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel[i]:
                vel_value = self.max_vel[i]
            elif vel_value < -self.max_vel[i]:
                vel_value = -self.max_vel[i]
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.args.particle_dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            # if (part.get_vel()[i] > (self.x_max[i] / 1000)) & (part.get_vel()[i] <= 1):
            #     pos_value = part.get_pos()[i] + 1
            # elif (part.get_vel()[i] >= -1) & (part.get_vel()[i] < (-self.x_max[i] / 1000)):
            #     pos_value = part.get_pos()[i] - 1
            # elif (part.get_vel()[i] > 1) & (part.get_vel()[i] <= self.max_vel[i]):
            #     pos_value = part.get_pos()[i] + math.ceil(part.get_vel()[i])
            # elif (part.get_vel()[i] < -1) & (part.get_vel()[i] >= -self.max_vel[i]):
            #     pos_value = part.get_pos()[i] - math.floor(part.get_vel()[i])
            # else:
            #     pos_value = part.get_pos()[i]

            if pos_value > self.x_max[i]:
                pos_value = self.x_max[i]
            elif pos_value < self.x_min[i]:
                pos_value = self.x_min[i]

            part.set_pos(i, pos_value)
        value = self.get_fitness(part.get_pos())
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.args.particle_dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value > self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.args.particle_dim):
                self.set_bestPosition(i, part.get_pos()[i])

    # def update(self):
    #     # self.load_training_data()
    #
    #     for i in range(self.args.iterations):
    #         for part in self.Particle_list:
    #             self.update_vel(part)  # 更新速度
    #             self.update_pos(part)  # 更新位置
    #         self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
    #         print('第' + str(i + 1) + '轮iteration的fitness:' + str(self.get_bestFitnessValue()))
    #         print('第' + str(i + 1) + '轮iteration的pos:' + str(self.get_bestPosition()))
    #         with open("pso_best.txt", "w") as f:
    #             f.write('第' + str(i + 1) + '轮iteration的fitness:' + str(self.get_bestFitnessValue()) + '\n')
    #             f.write('第' + str(i + 1) + '轮iteration的参数组合'
    #                                        ':' + str(self.get_bestPosition()) + '\n')
    #     return self.fitness_val_list, self.get_bestPosition()

    def update(self):
        # self.load_training_data()

        avgofdimension = [0.0 for i in range(self.args.particle_dim)]
        pm = 0.45
        exnon = 0.01
        x = 0.6
        pdavg = [0.0 for i in range(self.args.particle_dim)]
        variance = [0.0 for i in range(self.args.particle_dim)]

        for i in range(self.args.iterations):

            avgofdistance = 0.0  # 计算δ(pBest,gBest)  Eq.10
            for part in self.Particle_list:
                dist = 0
                for dim in range(self.args.particle_dim):
                    dist += math.pow((self.best_position[dim] - part.get_best_pos()[dim]), 2)
                avgofdistance += math.sqrt(dist)
            avgofdistance /= self.args.particle_num

            # for dim in range(self.args.particle_dim):  # 计算每一轮迭代的全局最优位置
            #     self.genGlobalBest[i][dim] = self.best_position[dim]

            if avgofdistance < 0.6 :# δd 设为 0.6

                # gBest 扰动维度距离计算
                for dim in range(self.args.particle_dim):  # 计算δ(gBest(particle/d)) Eq.12
                    avgofdimension[dim] = 0.0
                    for j in range(i + 1):
                        avgofdimension[dim] += math.pow((self.genGlobalBest[j][dim] - self.genGlobalBest[i][dim]), 2)
                    avgofdimension[dim] = math.sqrt(avgofdimension[dim] / (i + 1))

                # gBest 扰动判断和操作
                for dim in range(self.args.particle_dim):
                    if avgofdimension[dim] < 3:  # εδ(gbest)∈[0,10^-10]
                        x = 4 * x * (1 - x)
                        r6 = random.random()
                        r1 = 32767
                        if exnon < pm - r6:
                            timesequence = int(x * r1) % (self.x_max[dim])
                            self.best_position[dim] = (self.best_position[dim] + timesequence) % (self.x_max[dim])
                            self.logger.info('第' + str(dim) + '维进行gBest扰动操作')
                        elif r6 > pm:
                            self.best_position[dim] = self.best_position[dim] - random.randint(0, self.x_max[dim])
                            if self.best_position[dim] < self.x_min[dim]:
                                self.best_position[dim] = self.best_position[dim] + self.x_max[dim]
                            self.logger.info('第' + str(dim) + '维进行gBest扰动操作')

                # pBest 扰动维度距离计算

                # 计算每维位置的平均值
                for dim in range(self.args.particle_dim):
                    pdavg[dim] = 0.0
                    for part in self.Particle_list:
                        pdavg[dim] += part.get_best_pos()[dim]
                    pdavg[dim] /= self.args.particle_num

                # 计算每个位置的方差
                for dim in range(self.args.particle_dim):
                    variance[dim] = 0.0
                    for part in self.Particle_list:
                        variance[dim] += math.pow((part.get_best_pos()[dim] - pdavg[dim]), 2)
                    variance[dim] = math.sqrt(variance[dim] / self.args.particle_num)

                # pBest扰动
                for dim in range(self.args.particle_dim):
                    if variance[dim] < 3:  # εδ(pbest)∈[0,10^-10]
                        for part in self.Particle_list:
                            r3 = random.random()
                            if exnon < pm - r3:
                                part.set_best_pos(dim, (part.get_best_pos()[dim] + random.randint(0, self.x_max[dim])) % (self.x_max[dim]))
                                #print('粒子的第' + str(dim) + '维进行pBest扰动操作')
                            elif r3 > pm:
                                part.set_best_pos(dim, part.get_best_pos()[dim] + self.x_max[dim])
                                #print('粒子的第' + str(dim) + '维进行pBest扰动操作')

            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
                #print(str(part.get_pos()))
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表

            # sum_particle = self.best_position[0] + self.best_position[3] + self.best_position[4]
            # self.best_position[0] = self.best_position[0]/sum_particle
            # self.best_position[3] = self.best_position[3]/sum_particle
            # self.best_position[4] = self.best_position[4]/sum_particle

            self.logger.info('第' + str(i + 1) + '轮iteration的fitness:' + str(self.get_bestFitnessValue()))
            self.logger.info('第' + str(i + 1) + '轮iteration的pos:' + str(self.get_bestPosition()))

        return self.fitness_val_list, self.get_bestPosition()