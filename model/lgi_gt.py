

from typing import Union 
import os 

import math 
import pandas as pd 

import torch 
from torch import Tensor 
from torch import nn 
from torch.nn import init 

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool 
from torch_geometric.data import Batch, Data 

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder 

from .gconv import GINConv, GCNConv, EELA 
from .tlayer import GraphTransformerEncoderLayer 
from .rwse import RW_StructuralEncoder 
from .ast_node_encoder import ASTNodeEncoder 


class LGI_GT(nn.Module): 

    def __init__(
        self, 
        
        gconv_dim: int, 
        tlayer_dim: int, 
        dataset_name: str, 

        dataset_root = None, 
        num_vocab = None, 
        max_seq_len = None, 
        segment_pooling = None, 

        in_dim = None, 
        out_dim = None, 

        node_num_types = None, 
        edge_num_types = None, 
        dim_pe = None, 
        num_rw_steps = None, 
        
        gconv_attn_dropout: float = 0., 
        gconv_ffn_dropout: float = 0., 
        tlayer_attn_dropout: float = 0., 
        tlayer_ffn_dropout: float = 0., 
        tlayer_ffn_hidden_times: int = 1, 
        gconv_type: str = 'gin', 
        num_layers: int = 4, 
        num_heads: int = 4, 
        middle_layer_type: str = 'none', 
        skip_connection: str = 'none', 
        readout: str = 'mean', 
        norm: str = 'ln', 

        out_layer: int = 1, 
        out_hidden_times: int = 1 
    ): 
        super().__init__() 

        self.gconv_dim = gconv_dim 
        self.tlayer_dim = tlayer_dim 
        self.dataset_name = dataset_name 

        self.dataset_root = dataset_root 
        self.max_seq_len = max_seq_len 

        self.in_dim = in_dim 

        self.node_num_types = node_num_types 
        self.edge_num_types = edge_num_types 
        self.dim_pe = dim_pe 

        self.num_layers = num_layers 
        self.skip_connection = skip_connection 
        self.middle_layer_type = middle_layer_type 
        self.readout = readout 

        if segment_pooling == 'mean': 
            self.segment_pooling_fn = global_mean_pool 
        elif segment_pooling == 'max': 
            self.segment_pooling_fn = global_max_pool 
        elif segment_pooling == 'sum': 
            self.segment_pooling_fn = global_add_pool 
        else: 
            self.segment_pooling_fn = None 

        if readout == 'mean': 
            self.readout_fn = global_mean_pool 
        elif readout == 'add': 
            self.readout_fn = global_add_pool 
        elif readout == 'cls': 
            self.CLS = nn.Parameter(torch.randn(1, tlayer_dim)) 
        else: 
            pass 

        self.init_node_edge_encoders() 

        self.se_encoder = None 
        if num_rw_steps: 
            self.se_encoder = RW_StructuralEncoder(dim_pe, num_rw_steps, 
                                model_type='linear', 
                                n_layers=2, 
                                norm_type='bn') 

        self.graph_pred_linear_list = None 
        self.predict_head = None 
        if dataset_name == 'ogbg-code2': 
            self.graph_pred_linear_list = nn.ModuleList() 
            for i in range(max_seq_len): 
                self.graph_pred_linear_list.append(nn.Linear(tlayer_dim, num_vocab)) 
        else: 
            predict_head_modules = [] 
            last_layer_dim = tlayer_dim 
            for i in range(out_layer-1): 
                predict_head_modules.append(nn.Linear(last_layer_dim, out_hidden_times * tlayer_dim)) 
                predict_head_modules.append(nn.ReLU()) 
                last_layer_dim = out_hidden_times * tlayer_dim 
            predict_head_modules.append(nn.Linear(last_layer_dim, out_dim)) 
            self.predict_head = nn.Sequential(*predict_head_modules) 

        self.gconvs = nn.ModuleList() 
        self.tlayers = nn.ModuleList() 
        self.middle_layers = nn.ModuleList() 

        for i in range(num_layers): 
            if gconv_type == 'gin': 
                self.gconvs.append(GINConv(gconv_dim)) 
            elif gconv_type == 'gcn': 
                if self.dataset_name == 'ogbg-code2': 
                    self.gconvs.append(GCNConv(gconv_dim, 128)) 
                else: 
                    self.gconvs.append(GCNConv(gconv_dim, gconv_dim)) 
            elif gconv_type == 'eela': 
                self.gconvs.append(EELA(gconv_dim, num_heads, 
                                local_attn_dropout_ratio=gconv_attn_dropout, 
                                local_ffn_dropout_ratio=gconv_ffn_dropout)) 
            
            if middle_layer_type == 'residual': 
                self.middle_layers.append(nn.BatchNorm1d(gconv_dim)) 
            elif middle_layer_type == 'mlp': 
                self.middle_layers.append(nn.Sequential(
                    nn.Dropout(gconv_ffn_dropout), 
                    nn.Linear(gconv_dim, gconv_dim), 
                    nn.LayerNorm(gconv_dim), 
                    nn.ReLU(), 
                    nn.Dropout(gconv_ffn_dropout) 
                )) 
            
            self.tlayers.append( 
                GraphTransformerEncoderLayer( 
                    tlayer_dim, 
                    num_heads, 
                    attn_dropout_ratio=tlayer_attn_dropout, 
                    dropout_ratio=tlayer_ffn_dropout, 
                    ffn_hidden_times=tlayer_ffn_hidden_times, 
                    norm=norm)) 

    def init_node_edge_encoders(self): 
        if self.dataset_name == 'ogbg-code2': 
            nodetypes_mapping = pd.read_csv(os.path.join(self.dataset_root, 'mapping', 'typeidx2type.csv.gz')) 
            nodeattributes_mapping = pd.read_csv(os.path.join(self.dataset_root, 'mapping', 'attridx2attr.csv.gz')) 
            self.node_encoder = ASTNodeEncoder( 
                    self.gconv_dim, 
                    num_nodetypes = len(nodetypes_mapping['type']), 
                    num_nodeattributes = len(nodeattributes_mapping['attr']), 
                    max_depth = 20 
                ) 
            self.edge_encoder = nn.Linear(2, 128) 
        elif self.dataset_name == 'ogbg-molpcba': 
            self.node_encoder = AtomEncoder(self.gconv_dim) 
            self.edge_encoder = BondEncoder(self.gconv_dim) 
        elif self.dataset_name == 'ZINC': 
            self.node_encoder = nn.Embedding(self.node_num_types, self.gconv_dim - self.dim_pe) 
            self.edge_encoder = nn.Embedding(self.edge_num_types, self.gconv_dim) 
        elif self.dataset_name == 'PATTERN' or self.dataset_name == 'CLUSTER': 
            self.node_encoder = nn.Linear(self.in_dim, self.gconv_dim - self.dim_pe) 
            self.edge_encoder = nn.Embedding(1, self.gconv_dim) 

    def reset_parameters(self): 
        if self.readout == 'cls': 
            init.kaiming_uniform_(self.CLS, a=math.sqrt(5)) 

        if self.dataset_name == 'ogbg-molpcba': 
            for emb in self.node_encoder.atom_embedding_list: 
                emb.reset_parameters() 
            for emb in self.edge_encoder.bond_embedding_list: 
                emb.reset_parameters() 
        else: 
            self.node_encoder.reset_parameters() 
            self.edge_encoder.reset_parameters() 
        
        if self.se_encoder: 
            self.se_encoder.reset_parameters() 
        
        if self.graph_pred_linear_list:
            for layer in self.graph_pred_linear_list: 
                layer.reset_parameters() 
        elif self.predict_head: 
            for layer in self.predict_head: 
                if isinstance(layer, nn.Linear): 
                    layer.reset_parameters() 
            
        for layer in self.gconvs: 
            layer.reset_parameters() 
        for layer in self.tlayers: 
            layer.reset_parameters() 
        for layer in self.middle_layers: 
            if isinstance(layer, nn.BatchNorm1d): 
                layer.reset_parameters() 
            else: 
                for l in layer: 
                    if isinstance(l, nn.Linear) or isinstance(l, nn.LayerNorm):
                        l.reset_parameters() 

    def forward(self, data: Union[Data, Batch]) -> Tensor: 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_attr = data.edge_attr 
        num_graphs = data.num_graphs 

        if self.dataset_name == 'ogbg-code2': 
            """
            segment graph into subgraphs (if node number > 1000)
            CLS from different subgraphs (same graph) will be global pooled 
            """ 
            node_depth = data.node_depth 
            # use node_segment_batch as batch 
            node_segment_batch = data.node_segment_batch # every 1000 nodes come into 1 segment as 1 subgraph
            subgraphs_to_graph_batch = data.subgraphs_to_graph_batch # which subgraph belongs to which graph, used for cls readout
            num_segments = subgraphs_to_graph_batch.shape[0] # how many subgraphs after segmenting 
            num_graphs = num_segments 
            x = self.node_encoder(x, node_depth.view(-1,)) 
        elif self.dataset_name == 'ogbg-molpcba': 
            x = self.node_encoder(x) 
        else: 
            rw = data.pestat_RWSE 
            x = x.squeeze(-1) 
            x = self.node_encoder(x) 
            rwse = self.se_encoder(rw) 
            x = torch.cat((x, rwse), dim=1) 

        edge_attr = self.edge_encoder(edge_attr) 

        if self.readout == 'cls': 
            batch_CLS = self.CLS.expand(num_graphs, 1, -1) 
        else: 
            batch_CLS = None 

        if self.middle_layer_type == 'residual': 
            out = x 
        else: 
            out = 0 

        for i in range(self.num_layers): 

            x = self.gconvs[i](x, edge_index, edge_attr) 

            if self.middle_layer_type == 'residual': 
                x = out + x 
                x = self.middle_layers[i](x) 
                out = x 
            elif self.middle_layer_type == 'mlp': 
                x = self.middle_layers[i](x) 

            if self.dataset_name == 'ogbg-code2': 
                graph = (x, node_segment_batch, batch_CLS) 
            else: 
                graph = (x, batch, batch_CLS) 
            x = self.tlayers[i](graph) 

            if self.readout == 'cls': 
                batch_CLS = x[-num_graphs:].unsqueeze(1) 
                x = x[:-num_graphs] 

            if self.skip_connection == 'none': 
                out = x 
            elif self.skip_connection == 'long': 
                out = out + x 
            elif self.skip_connection == 'short': 
                out = out + x 
                x = out 

        if self.readout == 'cls': 
            out = batch_CLS.squeeze(1) 
            if self.segment_pooling_fn: 
                out = self.segment_pooling_fn(out, subgraphs_to_graph_batch) 
        elif self.readout: 
            out = self.readout_fn(out, batch) 

        if self.graph_pred_linear_list: 
            pred_list = [] 
            for i in range(self.max_seq_len): 
                pred_list.append(self.graph_pred_linear_list[i](out)) 
            out = pred_list 
        elif self.predict_head:
            out = self.predict_head(out) 

        return out 
