
from torch import nn 
import torch_geometric.nn as gnn 
import torch.nn.functional as F 

import math 
from typing import Optional 

import torch 
from torch_geometric.utils import softmax
from torch_geometric.utils import degree


class GCNConv(gnn.MessagePassing): # ogbg-code2, PATTERN, CLUSTER 
    def __init__(self, emb_dim, edge_input_dim): 
        super().__init__(aggr='add')

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

        self.edge_encoder = nn.Linear(edge_input_dim, emb_dim) 
    
    def reset_parameters(self): 
        self.linear.reset_parameters() 
        self.root_emb.reset_parameters() 
        self.edge_encoder.reset_parameters() 

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr) 

        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype) + 1 
        deg_inv_sqrt = deg.pow(-0.5) 
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1) 

    def message(self, x_j, edge_attr, norm): 
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



class GINConv(gnn.MessagePassing): # ZINC 
    def __init__(self, emb_dim): 

        super().__init__(aggr = "add")

        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 

            nn.Linear(2*emb_dim, emb_dim) 
        ) 
        self.edge_linear = nn.Linear(emb_dim, emb_dim) 

    def reset_parameters(self): 
        self.edge_linear.reset_parameters() 
        for layer in self.mlp:
            if isinstance(layer, nn.Linear): 
                layer.reset_parameters() 

    def forward(self, x, edge_index, edge_attr): 
        edge_embedding = self.edge_linear(edge_attr) 
        out = self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)) 

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr) 

    def update(self, aggr_out):
        return aggr_out


class EELA(gnn.MessagePassing): # ogbg-molpcba 
    def __init__(self, hidden_dim: int, num_heads: int,
                local_attn_dropout_ratio: float = 0.0, 
                local_ffn_dropout_ratio: float = 0.0): 
        
        super().__init__(aggr='add', node_dim=0) 

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads 
        self.local_attn_dropout_ratio = local_attn_dropout_ratio 

        self.linear_dst = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_src_edge = nn.Linear(2 * hidden_dim, hidden_dim) 

        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(local_ffn_dropout_ratio), 
        ) 
    
    def reset_parameters(self): 
        self.linear_dst.reset_parameters() 
        self.linear_src_edge.reset_parameters() 
        for layer in self.ffn: 
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm): 
                layer.reset_parameters() 

    def forward(self, x, edge_index, edge_attr): 
        local_out = self.propagate(edge_index, x=x, edge_attr=edge_attr) 
        local_out = local_out.view(-1, self.hidden_dim) 
        x = self.ffn(local_out) 

        return x 


    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i: Optional[int]): 
        H, C = self.num_heads, self.hidden_dim // self.num_heads 

        x_dst = self.linear_dst(x_i).view(-1, H, C) 
        m_src = self.linear_src_edge(torch.cat([x_j, edge_attr], dim=-1)).view(-1, H, C) 

        alpha = (x_dst * m_src).sum(dim=-1) / math.sqrt(C) 

        alpha = F.leaky_relu(alpha, 0.2) 
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i) 
        alpha = F.dropout(alpha, p=self.local_attn_dropout_ratio, training=self.training) 

        return m_src * alpha.unsqueeze(-1) 

