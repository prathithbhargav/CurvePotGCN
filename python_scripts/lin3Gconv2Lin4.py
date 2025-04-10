
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



class lin3_GCNet_2conv_4linear(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.first = Linear(2,64)
        self.second = Linear(64,128)
        self.third  = Linear(128, 512)
        self.conv1 = GCNConv(512, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 256)

        self.lin = Linear(256, 128)
        self.lin2 = Linear(128,64)
        self.lin3 = Linear(64, 28)
        self.lin4 = Linear(28, 2)
    def forward(self, x, edge_index, batch,dropout):
        # 1. Obtain node embeddings 
        x = F.relu(self.first(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.second(x))
        x = F.dropout(x, p=dropout)
        x = F.relu(self.third(x))
        x = F.dropout(x, p=dropout)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()


        # 2. Readout layer
        x = gmp(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=dropout)
        x = F.relu(self.lin(x))
        x = F.dropout(x,p=dropout)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=dropout)
        x = F.relu( self.lin3(x) )
        x = self.lin4(x)
        
        
        return x
