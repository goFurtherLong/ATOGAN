# -*- coding: utf-8 -*-

import argparse

def build_args(model_name):
    
    parser = argparse.ArgumentParser(description=model_name)
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser): 
    
    # general settings
    parser.add_argument('--random_seed', type=int, default=123)    
    parser.add_argument("--dataset", type=str, default="Citeseer", required=False,
                        help="The input dataset.")
    parser.add_argument("--split", type=str, default="public", required=False,
                        help="The type of dataset split.")
    # settings for the genetic algorithm
    parser.add_argument('--num_individuals', type=int, default=20,
                        help='the population size')
    parser.add_argument('--num_generations', type=int, default=100,
                        help='number of evolving generations')
    parser.add_argument('--num_parents', type=int, default=10,
                        help='number of parents for crossover')
    parser.add_argument('--num_offsprings', type=int, default=4,
                        help='number of offsprings')
    parser.add_argument('--mutate_prob', type=float, default=0.02,
                        help='mutation probability')
    
    # settings for the gnn model
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='number of the GNN layers')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")    
    parser.add_argument("--drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="cuda")

    # setting for the Particle swarm optimization
    parser.add_argument("--c1", type=float, default=1.8,
                        help='pso parameter c1 ')
    parser.add_argument("--c2", type=float, default=1.8,
                        help='pso parameter c1 ')
    parser.add_argument("--W", type=float, default=0.729,
                        help='pso parameter W ')
    parser.add_argument('--particle_dim', type=int, default=13,
                        help='the dimension of each particle')
    parser.add_argument('--particle_num', type=int, default=20,
                        help='the size of particle swarm')
    parser.add_argument('--iterations', type=int, default=50,
                        help='the maximum number of iterations')

#     parser.add_argument('--save_epoch', type=int, default=2)
#     parser.add_argument('--max_save_num', type=int, default=5)
#     parser.add_argument("--residual", action="store_false",
#                         help="use residual connection")
