import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch.nn as nn


import itertools
import os
from utils import *


os.environ["DGLBACKEND"] = "pytorch"

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch.nn.utils.prune as prune
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("../"))

import torch
import torch_pruning as tp




def train(model,dataset, optimizer,callbacks = None ):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(dataset[0]), dataset[0].y).backward()
    optimizer.step()
    if callbacks is not None:
        for callback in callbacks:
                callback()

def test(model, dataset):
    model.eval()
    logits, accs = model(dataset[0]), []
    for _, mask in dataset[0]('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(dataset[0].y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs                

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
def magnitude_pruning(sparsity):
       
    print(f'Training and evaluation before pruning ')
    model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(1, 100):
        train(model,dataset, optimizer,callbacks = None)
        train_acc, val_acc, test_acc = test(model, dataset)
        if epoch%20==0:
            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    best_checkpoint = dict()
    best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_checkpoint['state_dict'])
    recover_model = lambda: model.load_state_dict(best_checkpoint['state_dict'])

    t0=time.time()
    train_acc, val_acc,dense_model_accuracy=test(model, dataset)
    t1=time.time()
    t_dence_model=t1 - t0
    ###
    dense_model_size = get_model_size(model)
    num_parm_dense_model=get_num_parameters(model, count_nonzero_only=True)
    ###   
    print(f"dense model has accuracy on test set={dense_model_accuracy:.2f}%")
    print(f"dense model has size={dense_model_size/MiB:.2f} MiB")
    print(f"The time inference of dence model is ={t_dence_model}") 
    print(f"The number of parametrs of dence model is:{num_parm_dense_model}")   
    
    
    print('_______________________________________________________')
    print(f'Prune the Model and Re-Evaluate the Accuracy')
    recover_model()
    pruner = FineGrainedPruner(model, sparsity)
    pruner.apply(model)


    t0=time.time()
    train_acc, val_acc,pruned_model_accuracy=test(model, dataset)
    t1=time.time()
    t_pruned_model=t1 - t0
    ###
    pruned_model_size = get_model_size(model)
    num_parm_pruned_model=get_num_parameters(model, count_nonzero_only=True)
    ###   
    print(f"{sparsity*100}% sparse model has accuracy on test set={pruned_model_accuracy:.2f}%")
    print(f"{sparsity*100}% sparse model has size={pruned_model_size/MiB:.2f} MiB")
    print(f"The time inference of {sparsity*100}% sparse model is ={t_pruned_model}") 
    print(f"The number of parametrs of {sparsity*100}% sparse model is:{num_parm_pruned_model}")
    print(f"{sparsity*100}% sparse model has size={pruned_model_size/MiB:.2f} MiB, "
          f"which is {dense_model_size/pruned_model_size:.2f}X smaller than "
          f"the {dense_model_size/MiB:.2f} MiB dense model")
    
    best_sparse_checkpoint = dict()
    best_sparse_accuracy = 0
    num_finetune_epochs=100
    
    print('_______________________________________________________')
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask
        #    to keep the model sparse during the training
        train(model,dataset, optimizer,callbacks=[lambda: pruner.apply(model)])
        train_acc, val_acc, accuracy = test(model, dataset)
        
        is_best = accuracy > best_sparse_accuracy
        if is_best:
            best_sparse_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_sparse_accuracy = accuracy

        if epoch % 20 == 0:
             print(f'Epoch {epoch} Sparse Accuracy {accuracy:.2f}% / Best Sparse Accuracy: {best_sparse_accuracy:.2f}%')

    model.load_state_dict(best_sparse_checkpoint['state_dict'])


    t0=time.time()
    train_acc, val_acc,pruned_finetune_model_accuracy=test(model, dataset)
    t1=time.time()
    t_pruned_finetune_model=t1 - t0
    ###
    pruned_finetune_model_size = get_model_size(model)
    num_parm_pruned_finetune_model=get_num_parameters(model, count_nonzero_only=True)
    ###   
    print(f"{sparsity*100}% sparse model has accuracy on test set={pruned_finetune_model_accuracy:.2f}%")
    print(f"{sparsity*100}% sparse model has size={pruned_finetune_model_size/MiB:.2f} MiB")
    print(f"The time inference of {sparsity*100}% sparse model is ={t_pruned_finetune_model}") 
    print(f"The number of parametrs of {sparsity*100}% sparse model is:{num_parm_pruned_finetune_model}")
    print(f"{sparsity*100}% sparse model has size={pruned_finetune_model_size/MiB:.2f} MiB, "
          f"which is {dense_model_size/pruned_finetune_model_size:.2f}X smaller than "
      f"the {dense_model_size/MiB:.2f} MiB dense model")
    
    base_model=[dense_model_accuracy,dense_model_size,t_dence_model,num_parm_dense_model]
    pruned_model=[pruned_model_accuracy,pruned_model_size,t_pruned_model,num_parm_pruned_model]
    pruned_finetune_model=[pruned_finetune_model_accuracy,pruned_finetune_model_size,t_pruned_finetune_model,num_parm_pruned_finetune_model]

    return  base_model, pruned_model, pruned_finetune_model
    




if __name__ == '__main__':
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    sparsity=0.1
    base_model, pruned_model, pruned_finetune_model=magnitude_pruning(sparsity)

    
