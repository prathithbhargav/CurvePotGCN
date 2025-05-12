
from sklearn.model_selection import KFold
import random
import pandas as pd
import numpy as np
import os
import difflib
from scipy.spatial import distance
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
import sys

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from scipy.spatial import distance
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
from torcheval.metrics.functional import binary_confusion_matrix as bcm
import torcheval.metrics.functional as tmf
# from lin3Gconv2Lin3 import *
from lin3Gconv2Lin4 import *
from humans import get_humans_data
import time
#print('choose gpu number in range 0 to 3')
#gpu_choose = sys.argv[1]
#gpu_choose = int(gpu_choose)
#assert isinstance(gpu_choose,int)


epoch = 250

j = lin3_GCNet_2conv_4linear(1024)

h = j.__class__.__name__

kf = KFold(n_splits=5, shuffle=True)

if torch.cuda.is_available():
    device = f'cuda:0'
else :
    device = 'cpu'


print('data loading')
a = get_humans_data(owner='uddipan', in_device = device)
print('data loaded')
a_cv = a[:-750]    

criterion = torch.nn.CrossEntropyLoss()




def reset_weights(model):

  for layer in model.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

    

def train_once(model, optimizer, loader,dropout):
    model.train()
    collective_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch,dropout)
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()
        collective_loss+=loss
    return collective_loss
    
def test(model, loader,dropout):
    model.eval()
    correct = 0
    print(len(loader.dataset))
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch,dropout)
            pred = out.argmax(dim = 1)
            correct += (pred == data.y).sum()

    return correct / len(loader.dataset)


       
def test_new(model, loader,dropout):
    model.eval()
    correct = 0
    l1 = []
    l2 = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch,dropout)
            pred = out.argmax(dim = 1)        
            correct += (pred == data.y).sum()
            l1.append(pred.reshape(-1,1))
            l2.append(data.y.reshape(-1,1))
        # print(f'correct are {correct/len(loader.dataset)} ')
        return (l1,l2)
        # train.report({"accuracy": test_acc})




def calc_everything(device):
    for ids, (train_ids, test_ids) in enumerate(kf.split(a_cv)):
        with open(f'humans_{ids}_{h}.txt', 'a') as f:
            #define model here
            model = j.to(device)
            model.apply(reset_weights)
        
        
            train_data = [a_cv[i] for i in train_ids]
            test_data = [a_cv[i] for i in test_ids]
            
            train_loader = DataLoader(train_data, batch_size=64)
            test_loader = DataLoader(test_data, batch_size=64)
        
            dropout = 0
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            print('training ...')
            for _ in tqdm(range(epoch)):
                loss = train_once(model, optimizer, train_loader, dropout)
                train_acc = test(model, train_loader, dropout)
                test_acc = test(model,  test_loader, dropout)
                f.write(f'train loss { loss:20.5f}| train acc { train_acc:20.5f} | test_acc { test_acc:20.5f} \n')
            f.close()
    
            with open(f'humans_{ids}_metrics_{h}_test.txt','a') as f:
                t1,t2 = test_new(j, test_loader, dropout=0)
                
                
                tensor_initial = t1[0]
                for i in range(1,len(t1)):
                    tensor_initial = torch.vstack((tensor_initial,t1[i]))
                
                data_initial = torch.tensor([[15]]).to(device)
                
                for data in test_loader:
                    data_initial = torch.vstack((data_initial,data.y.reshape(-1,1)))
                data_initial = data_initial[1:,:]
                
                
                
                aaa = bcm(tensor_initial.reshape(-1),data_initial.reshape(-1))
        
                auroc = tmf.binary_auroc( tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                recall = tmf.binary_recall( tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                f1_score = tmf.binary_f1_score(tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                precision = tmf.binary_precision( tensor_initial.reshape(-1),data_initial.reshape(-1))
    
                f.write(f'auroc {auroc:20.5f} | recall { recall:20.5f} | f1_score {f1_score:20.5f} | precision {precision:20.5f}  ')
                np.savetxt(f'humans_{ids}_{h}_confusion_mat_test.txt',aaa.to('cpu'))
                f.close()
                
            with open(f'humans_{ids}_metrics_{h}_train.txt','a') as f:
                t1,t2 = test_new(j, train_loader, dropout=0)
                
                
                tensor_initial = t1[0]
                for i in range(1,len(t1)):
                    tensor_initial = torch.vstack((tensor_initial,t1[i]))
                
                data_initial = torch.tensor([[15]]).to(device)
                
                for data in train_loader:
                    data_initial = torch.vstack((data_initial,data.y.reshape(-1,1)))
                data_initial = data_initial[1:,:]
                
                
                
                aaa = bcm(tensor_initial.reshape(-1),data_initial.reshape(-1))
        
                auroc = tmf.binary_auroc( tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                recall = tmf.binary_recall( tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                f1_score = tmf.binary_f1_score(tensor_initial.reshape(-1),data_initial.reshape(-1))
                
                precision = tmf.binary_precision( tensor_initial.reshape(-1),data_initial.reshape(-1))
    
                f.write(f'auroc {auroc:20.5f} | recall { recall:20.5f} | f1_score {f1_score:20.5f} | precision {precision:20.5f}  ')
                np.savetxt(f'humans_{ids}_{h}_confusion_mat_train.txt',aaa.to('cpu'))
                f.close()
    
    
    
    
    pathname = os.path.dirname(os.path.realpath(__file__))
    new_path = os.path.join(pathname,'humans_' +str(h)+'_'+ str(time.time()))
    os.mkdir(new_path)
    os.chdir(new_path)
    
    
    with open(f'humans_{h}.txt', 'a') as f:
        #define model here
        model = j.to(device)
        model.apply(reset_weights)
    
    
        train_data = a[:6000]
        test_data = a[6000:-750]
        val_data = a[-750:]
        
        
        train_loader = DataLoader(train_data, batch_size=64)
        test_loader = DataLoader(test_data, batch_size=64)
        val_loader = DataLoader(val_data, batch_size=64)
    
        criterion = torch.nn.CrossEntropyLoss()
        dropout = 0
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        print('training ...')
        for _ in tqdm(range(epoch)):
            loss = train_once(model, optimizer, train_loader, dropout)
            train_acc = test(model, train_loader, dropout)
            test_acc = test(model,  test_loader, dropout)
            val_acc = test(model,  val_loader, dropout)
            f.write(f'train loss { loss:20.5f}| train acc { train_acc:20.5f} | test_acc { test_acc:20.5f} | val_acc { val_acc:20.5f} \n')
            if epoch % 5 == 0:
                string = new_path + '/'+str(test_acc)
                torch.save(model.state_dict(),string) 
        f.close()
    
        with open(f'humans_metrics_{h}_test.txt','a') as f:
            t1,t2 = test_new(j, test_loader, dropout=0)
            
            
            tensor_initial = t1[0]
            for i in range(1,len(t1)):
                tensor_initial = torch.vstack((tensor_initial,t1[i]))
            
            data_initial = torch.tensor([[15]])
            
            for data in test_loader:
                data_initial = torch.vstack((data_initial,data.y.reshape(-1,1)))
            data_initial = data_initial[1:,:]
            
            
            
            aaa = bcm(tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            auroc = tmf.binary_auroc( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            recall = tmf.binary_recall( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            f1_score = tmf.binary_f1_score(tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            precision = tmf.binary_precision( tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            f.write(f'auroc {auroc:20.5f} | recall { recall:20.5f} | f1_score {f1_score:20.5f} | precision {precision:20.5f}  ')
            np.savetxt(f'humans_{h}_confusion_mat_test.txt',aaa.to('cpu'))
            f.close()
            
        with open(f'humans_metrics_{h}_train.txt','a') as f:
            t1,t2 = test_new(j, train_loader, dropout=0)
            
            
            tensor_initial = t1[0]
            for i in range(1,len(t1)):
                tensor_initial = torch.vstack((tensor_initial,t1[i]))
            
            data_initial = torch.tensor([[15]])
            
            for data in train_loader:
                data_initial = torch.vstack((data_initial,data.y.reshape(-1,1)))
            data_initial = data_initial[1:,:].to(device)
            
            
            
            aaa = bcm(tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            auroc = tmf.binary_auroc( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            recall = tmf.binary_recall( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            f1_score = tmf.binary_f1_score(tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            precision = tmf.binary_precision( tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            f.write(f'auroc {auroc:20.5f} | recall { recall:20.5f} | f1_score {f1_score:20.5f} | precision {precision:20.5f}  ')
            np.savetxt(f'humans_{h}_confusion_mat_train.txt',aaa.to('cpu'))
            f.close()
        with open(f'humans_metrics_{h}_val.txt','a') as f:
            t1,t2 = test_new(j, val_loader, dropout=0)
            
            
            tensor_initial = t1[0]
            for i in range(1,len(t1)):
                tensor_initial = torch.vstack((tensor_initial,t1[i]))
            
            data_initial = torch.tensor([[15]])
            
            for data in val_loader:
                data_initial = torch.vstack((data_initial,data.y.reshape(-1,1)))
            data_initial = data_initial[1:,:].to(device)
            
            
            
            aaa = bcm(tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            auroc = tmf.binary_auroc( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            recall = tmf.binary_recall( tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            f1_score = tmf.binary_f1_score(tensor_initial.reshape(-1),data_initial.reshape(-1))
            
            precision = tmf.binary_precision( tensor_initial.reshape(-1),data_initial.reshape(-1))
    
            f.write(f'auroc {auroc:20.5f} | recall { recall:20.5f} | f1_score {f1_score:20.5f} | precision {precision:20.5f}  ')
            np.savetxt(f'humans_{h}_confusion_mat_val.txt',aaa.to('cpu'))
            f.close() 
print(device)
calc_everything(device )
    
