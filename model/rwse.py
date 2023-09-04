
import torch 
from torch import nn 

class RW_StructuralEncoder(nn.Module): 

    def __init__(self, dim_pe, num_rw_steps, model_type='linear', n_layers=1, norm_type='bn'):
        super().__init__()

        if norm_type == 'bn': 
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                # layers.append(nn.ReLU()) 
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def reset_parameters(self): 
        if self.raw_norm: 
            self.raw_norm.reset_parameters() 
        if isinstance(self.pe_encoder, nn.Linear): 
            self.pe_encoder.reset_parameters() 
        elif isinstance(self.pe_encoder, nn.Sequential):
            for layer in self.pe_encoder: 
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters() 

    def forward(self, rw):
        if self.raw_norm:
            rw = self.raw_norm(rw)
        rw = self.pe_encoder(rw)  # (Num nodes) x dim_pe

        return rw 