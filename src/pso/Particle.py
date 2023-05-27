import random


class Particle:
    # 初始化
    def __init__(self, x, x_min, x_max, max_vel, dim):
        x_min = [-0.1,-0.5,-0.2,-0.1,-0.1]
        x_max = [0.1,0.5,0.2,0.1,0.1]
        
        self.__pos = [(random.uniform(x_min[i], x_max[i])+x[i]) for i in range(dim)]  # 粒子的位置
        self.__vel = [(random.uniform(-max_vel[i], max_vel[i])) for i in range(dim)]  # 粒子的速度
        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = 0.0  # 适应度函数值
        self.test_acc = 0

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue
