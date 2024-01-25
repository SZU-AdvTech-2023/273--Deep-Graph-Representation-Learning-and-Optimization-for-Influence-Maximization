import os
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
import padding

from main.utils import load_dataset, InverseProblemDataset, adj_process, diffusion_evaluation
from main.model.gat import GAT, SpGAT
from main.model.model import GNNModel, VAEModel, DiffusionPropagate, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable

print('Is GPU available? {}\n'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1


def read_graph_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split())
        edges = [tuple(map(int, line.split())) for line in lines[1:]]
    return num_nodes, num_edges, edges


def create_adj_and_inverse_pairs(num_nodes, edges):
    G = nx.Graph()
    G.add_edges_from(edges)

    adj_matrix = nx.adjacency_matrix(G)
    # adj_sparse_tensor = torch.Tensor(adj_matrix.toarray()).to_sparse()

    # Initialize inverse_pairs with zeros

    # Populate inverse_pairs with a default weight (e.g., 1)

    return adj_matrix


# 请将 'your_graph_data.txt' 替换为你的实际文件路径
file_name = input("请输入文件名（不包括扩展名）：")

file_txt_path = f'my_data/{file_name}.txt'

if os.path.exists(file_txt_path):
    file_pkl_path = f'my_pkl/{file_name}.pkl'
    if os.path.exists(file_pkl_path):
        print(f"The file '{file_pkl_path}' exists.")
    else:
        num_nodes, num_edges, edges = read_graph_from_txt(file_txt_path)
        adj = create_adj_and_inverse_pairs(num_nodes, edges)
        data_to_save = adj
        with open(file_pkl_path, 'wb') as file:
            pickle.dump(data_to_save, file)
else:
    print(f"The file '{file_txt_path}' does not exist.")
    exit()


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sampling(inverse_pairs):
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1 * inverse_pairs.shape[0])).indices
    return top_k


n = int(input("请输入一个整数："))

random5 = bool(input("是否random5(0/1):"))

diffusion = input("请输入diffusion_model：")

with open(file_pkl_path, 'rb') as f:
    graph = pickle.load(f)

adj = graph

seednum = int(adj.shape[0] * 0.1)

inverse_pairs_num = 100


inverse_pairs = padding.my_diffusion_model(adj, seednum, inverse_pairs_num, diffusion)

for now_n in range(n):

    adj = graph

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.Tensor(adj.toarray()).to_sparse()

    # if random5:
    #     batch_size = 2
    #     hidden_dim = 4096
    #     latent_dim = 1024
    # else:
    batch_size = 16
    hidden_dim = 1024
    latent_dim = 512

    train_set, test_set = torch.utils.data.random_split(inverse_pairs,
                                                        [len(inverse_pairs) - batch_size,
                                                         batch_size])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    encoder = Encoder(input_dim=inverse_pairs.shape[1],
                      hidden_dim=hidden_dim,
                      latent_dim=latent_dim)

    decoder = Decoder(input_dim=latent_dim,
                      latent_dim=latent_dim,
                      hidden_dim=hidden_dim,
                      output_dim=inverse_pairs.shape[1])

    vae_model = VAEModel(Encoder=encoder, Decoder=decoder).to(device)

    forward_model = SpGAT(nfeat=1,
                          nhid=64,
                          nclass=1,
                          dropout=0.2,
                          nheads=4,
                          alpha=0.2)

    optimizer = Adam([{'params': vae_model.parameters()}, {'params': forward_model.parameters()}],
                     lr=1e-4)

    adj = adj.to(device)
    forward_model = forward_model.to(device)
    forward_model.train()


    def loss_all(x, x_hat, y, y_hat):
        reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        forward_loss = F.mse_loss(y_hat, y, reduction='sum')
        # forward_loss = F.binary_cross_entropy(y_hat, y, reduction='sum')
        return reproduction_loss + forward_loss, reproduction_loss, forward_loss


    for epoch in range(300):
        begin = time.time()
        total_overall = 0
        forward_loss = 0
        reproduction_loss = 0
        precision_for = 0
        recall_for = 0
        precision_re = 0
        recall_re = 0

        for batch_idx, data_pair in enumerate(train_loader):
            # input_pair = torch.cat((data_pair[:, :, 0], data_pair[:, :, 1]), 1).to(device)

            x = data_pair[:, :, 0].float().to(device)
            y = data_pair[:, :, 1].float().to(device)
            optimizer.zero_grad()

            y_true = y.cpu().detach().numpy()
            x_true = x.cpu().detach().numpy()

            loss = 0
            for i, x_i in enumerate(x):
                y_i = y[i]

                x_hat = vae_model(x_i.unsqueeze(0))
                y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)
                total, re, forw = loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))

                loss += total

                x_pred = x_hat.cpu().detach().numpy()
                x_pred[x_pred > 0.01] = 1
                x_pred[x_pred != 1] = 0

                precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
                recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)

            total_overall += loss.item()
            loss = loss / x.size(0)

            loss.backward()
            optimizer.step()
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

        end = time.time()
        print("Epoch: {}".format(epoch + 1),
              "\tTotal: {:.4f}".format(total_overall / len(train_set)),
              "\tReconstruction Precision: {:.4f}".format(precision_re / len(train_set)),
              "\tReconstruction Recall: {:.4f}".format(recall_re / len(train_set)),
              "\tTime: {:.4f}".format(end - begin)
              )

    for param in vae_model.parameters():
        param.requires_grad = False

    for param in forward_model.parameters():
        param.requires_grad = False

    encoder = vae_model.Encoder
    decoder = vae_model.Decoder


    def loss_inverse(y_true, y_hat, x_hat):
        y_true = y_true.transpose(0, 1)
        forward_loss = F.mse_loss(y_hat, y_true)
        L0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]
        return forward_loss + L0_loss, L0_loss


    with open(file_pkl_path, 'rb') as f:
        graph = pickle.load(f)

    adj = graph

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.Tensor(adj.toarray()).to_sparse().to(device)

    topk_seed = sampling(inverse_pairs)

    z_hat = 0
    for i in topk_seed:
        z_hat += encoder(inverse_pairs[i, :, 0].unsqueeze(0).to(device))

    z_hat = z_hat / len(topk_seed)
    seed_num = int(x_hat.sum().item())
    y_true = torch.ones(x_hat.shape).to(device)

    z_hat = z_hat.detach()
    z_hat.requires_grad = True
    z_optimizer = Adam([z_hat], lr=1e-4)

    for i in range(300):
        x_hat = decoder(z_hat)

        y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)

        y = torch.where(y_hat > 0.05, 1, 0)

        loss, L0 = loss_inverse(y_true, y_hat, x_hat)

        loss.backward()
        z_optimizer.step()

        print('Iteration: {}'.format(i + 1),
              '\t Total Loss:{:.5f}'.format(loss.item())
              )

    top_k = x_hat.topk(seed_num)
    print(top_k)

    seed = top_k.indices[0].cpu().detach().numpy()

    # Save top_k and seed to a file
    filename_topk = 'my_seed/' + f'{file_name}' + '_' + f'{diffusion}' + '_topk_output_' + f'{now_n}' + '.txt'
    filename_seed = 'my_seed/' + f'{file_name}' + '_' + f'{diffusion}' + '_seed_output_' + f'{now_n}' + '.txt'

    with open(filename_topk, 'w') as file_topk:
        for value in top_k.indices[0]:
            file_topk.write(str(value) + '\n')

    with open(filename_seed, 'w') as file_seed:
        for value in seed:
            file_seed.write(str(value) + '\n')

    print(f"Top_k and Seed values have been written to {filename_topk} and {filename_seed}")

    print(top_k)
    print(seed)

    # ... (remaining code)

    # 指定文件名
    filename = 'my_seed/' + f'{file_name}' + '_' + f'{diffusion}' + '_seed_output_'+f'{now_n}'+'.txt'

    # 写入到文本文件
    with open(filename, 'w') as file:
        for value in seed:
            file.write(str(value) + '\n')

    print(f"Seed values have been written to {filename}")

    print(seed)

    with open('my_pkl/' + f'{file_name}' + '.PKL', 'rb') as f:
        graph = pickle.load(f)

    adj = graph

    influence = diffusion_evaluation(adj, seed, diffusion)
    print('Diffusion count: {}'.format(influence))
