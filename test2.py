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
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable


# Function to load graph from pickle file
def load_graph(file_name):
    with open(file_name, 'rb') as f:
        graph = pickle.load(f)
    return graph


# Function to read seed nodes from a txt file
def read_seed(file_name):
    with open(file_name, 'r') as f:
        seeds = [int(line.strip()) for line in f]
    return seeds


def select_percentage_data(data, percentage):
    selected_count = int(len(data) * percentage / 100)
    if selected_count < 1:
        selected_count = 1
    selected_data = data[:selected_count]
    return selected_data


# Function to calculate influence and write to file for different percentages
def calculate_and_write_influence(file_prefix, n, diffusion, percentages):

    graph = load_graph(f'my_pkl/{file_prefix}.PKL')  # Adjust the file name accordingly
    adj = graph

    for i in range(1):
        seed_file = f'my_seed/{file_prefix}_{diffusion}_seed_output_{i}.txt'
        seeds = read_seed(seed_file)

        x = []
        y = []

        for percentage in percentages:
            selected_data = select_percentage_data(seeds, percentage)
            impact_range = diffusion_evaluation(adj, selected_data, diffusion)
            impact_range = format(impact_range)
            x.append(percentage)
            y.append(float(impact_range))

        with open(f'{file_prefix}_influence_{i}.txt', 'w') as output_file:
            output_file.write(f'{file_prefix}_Deep_{diffusion}_x_{i} = {x}\n')
            output_file.write(f'{file_prefix}_Deep_{diffusion}_y_{i} = {y}\n')


# Number of iterations (replace with the desired value)
n_iterations = 10

# Percentages of seeds to choose for each iteration
seed_percentages = [1, 5, 10, 20, 40]

# Run the calculation and write influence to file
calculate_and_write_influence('CA-GrQc', n_iterations, 'IC', seed_percentages)
#calculate_and_write_influence('Dolphins', n_iterations, 'IC', seed_percentages)
