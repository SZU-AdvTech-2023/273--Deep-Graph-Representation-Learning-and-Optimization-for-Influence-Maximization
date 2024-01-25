import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import time
import networkx as nx
import random
import pickle
from scipy.special import softmax
from scipy.sparse import csr_matrix
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

from main.utils import load_dataset, InverseProblemDataset, adj_process, diffusion_evaluation
from main.model.gat import GAT, SpGAT
from main.model.model import GNNModel, VAEModel, DiffusionPropagate, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

def diffusion_test(adj, seed, diffusion='LT'):
    infected_nodes = []
    G = nx.from_scipy_sparse_matrix(adj)
    if diffusion == 'LT':
        model = ep.ThresholdModel(G)
        config = mc.Configuration()
        # 为每个节点设置传播阈值
        for n in G.nodes():
            config.add_node_configuration("threshold", n, 0.5)
    elif diffusion == 'IC':
        model = ep.IndependentCascadesModel(G)
        config = mc.Configuration()
        # 为每条边设置传播概率
        for e in G.edges():
            config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
    elif diffusion == 'SIS':
        model = ep.SISModel(G)
        config = mc.Configuration()
        # 设置模型参数
        config.add_model_parameter('beta', 0.001)
        config.add_model_parameter('lambda', 0.001)
    else:
        raise ValueError('Only IC, LT and SIS are supported.')

        # 设置模型的初始状态为感染的种子节点
    config.add_model_initial_configuration("Infected", seed)

    # 将配置应用于模型
    model.set_initial_status(config)

    # 进行模拟迭代（这里是 100 次迭代）
    iterations = model.iteration_bunch(100)

    # 提取每次迭代的节点状态
    node_status = iterations[0]['status']

    # 在每次迭代中更新节点状态
    for j in range(1, len(iterations)):
        node_status.update(iterations[j]['status'])

    # 将节点状态转换为 0（未感染）和 1（感染）
    inf_vec = np.array(list(node_status.values()))
    inf_vec[inf_vec == 2] = 1

    # 将被感染的节点加入列表
    infected_nodes.extend(np.where(inf_vec == 1)[0])

    return np.unique(infected_nodes)

def my_diffusion_model(adj, seed_num, inverse_pairs_nums, diffusion='LT'):

    adj_vex_num = adj.shape[0]
    inverse_pairs = torch.zeros((inverse_pairs_nums, adj_vex_num, 2), dtype=torch.float32)
    for i in range(inverse_pairs_nums):
        seed = np.random.choice(np.arange(0, adj_vex_num), size=seed_num, replace=False)
        inverse_pairs[i, seed, 0] = 1
        infected_nodes = diffusion_test(adj, seed, diffusion)
        inverse_pairs[i, infected_nodes, 1] = 1

    return inverse_pairs

# file_name='Dolphins'
#
# with open(f'{file_name}' +'.PKL', 'rb') as f:
#     graph = pickle.load(f)
#
# adj = graph
#
# # 设置参数
# seed_num = 3
# inverse_pairs_nums = 1
# diffusion_model = 'LT'
#
# # 调用 my_diffusion_model 函数生成逆向对
# inverse_pairs = my_diffusion_model(adj, seed_num, inverse_pairs_nums, diffusion_model)
#
# # 打印生成的逆向对
# print("Generated Inverse Pairs:")
# print(inverse_pairs)
