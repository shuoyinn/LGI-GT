

import math 
from typing import Optional, Tuple 

import torch 
from torch import Tensor 

from torch import nn 
from torch_geometric.utils import to_dense_batch 


class MultiHeadAttention(nn.Module): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.0): 

        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.dropout = nn.Dropout(dropout_ratio) 

        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 

        self.linear_attn_out = nn.Linear(hidden_dim, hidden_dim) 

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters() 
        self.linear_attn_out.reset_parameters()

    def forward(
        self,
        x_q: Tensor,
        x_k: Tensor, 
        x_v: Tensor, 
        mask: Tensor 
    ) -> Tensor: # B x N x F -> B x N x F 

        Q = self.linear_q(x_q) 
        K = self.linear_k(x_k) 
        V = self.linear_v(x_v) 

        dim_split = self.hidden_dim // self.num_heads
        Q_heads = torch.cat(Q.split(dim_split, 2), dim=0)
        K_heads = torch.cat(K.split(dim_split, 2), dim=0)
        V_heads = torch.cat(V.split(dim_split, 2), dim=0)
        
        attention_score = Q_heads.bmm(K_heads.transpose(1, 2)) 
        attention_score = attention_score / math.sqrt(self.hidden_dim // self.num_heads) 

        inf_mask = (~mask).unsqueeze(1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.num_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 
        
        A = self.dropout(A) 
        out = torch.cat((A.bmm(V_heads)).split(Q.size(0), 0), 2) 
        out = self.linear_attn_out(out) 
        
        return out


class NodeSelfAttention(MultiHeadAttention): 

    def __init__(self, hidden_dim: int, num_heads: int, 
            dropout_ratio: float = 0.):
        super().__init__(hidden_dim, num_heads, 
                            dropout_ratio) 

    def forward(
        self,
        x: Tensor, 
        mask: Tensor 
    ) -> Tensor: 

        return super().forward(x, x, x, mask) 
        

class GraphTransformerEncoderLayer(nn.Module): 
    
    def __init__(self, hidden_dim, num_heads: int, 
            attn_dropout_ratio: float = 0.0, 
            dropout_ratio: float = 0.0, 
            ffn_hidden_times: int = 2, 
            norm: str = 'ln'): 
        
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 

        if norm == 'ln':
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 

        self.dropout1 = nn.Dropout(dropout_ratio) 
        self.dropout2 = nn.Dropout(dropout_ratio) 
        self.dropout3 = nn.Dropout(dropout_ratio) 
        
        self.node_self_attention = NodeSelfAttention( 
                                    hidden_dim, num_heads, 
                                    attn_dropout_ratio) 

        self.linear_out1 = nn.Linear(hidden_dim, ffn_hidden_times * hidden_dim) 
        self.linear_out2 = nn.Linear(ffn_hidden_times * hidden_dim, hidden_dim) 

        self.norm0 = norm_class(hidden_dim) 
        self.norm1 = norm_class(hidden_dim) 

    def reset_parameters(self): 
        self.node_self_attention.reset_parameters() 
        self.linear_out1.reset_parameters() 
        self.linear_out2.reset_parameters() 
        self.norm0.reset_parameters()
        self.norm1.reset_parameters() 
        
    def forward(self, graph: Optional[Tuple[Tensor, Tensor, Tensor]]): 

        # Sparse -> Dense 
        x, batch, batch_CLS = graph 
        x_dense, mask = to_dense_batch(x, batch) 
        attention_mask = mask 

        if batch_CLS is not None: 
            x_dense = torch.cat([x_dense, batch_CLS], dim=1) 
            CLS_mask = torch.full((batch_CLS.shape[0], 1), True, device=batch_CLS.device) 
            attention_mask = torch.cat([mask, CLS_mask], dim=1) 
        
        # Dense given to Node Self Attention 
        attention_out = self.node_self_attention(x_dense, attention_mask) 
        attention_out = self.dropout1(attention_out) 
        attention_out = attention_out + x_dense 

        # Dense -> Sparse 
        # for BatchNorm, excluding padding nodes 
        if batch_CLS is not None: 
            batch_CLS = attention_out[:, -1, :] 
            attention_out = attention_out[:, :-1, :] 
            attention_out = attention_out[mask] 
            attention_out = torch.cat((attention_out, batch_CLS), dim=0) 
        else: 
            attention_out = attention_out[mask] 

        attention_out = self.norm0(attention_out) 
        
        out = self.linear_out1(attention_out) 
        out = torch.relu(out) 
        out = self.dropout2(out) 
        out = self.linear_out2(out) 
        out = self.dropout3(out) 

        out = out + attention_out 

        out = self.norm1(out) 

        return out 
