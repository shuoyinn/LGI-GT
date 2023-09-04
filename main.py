

import configargparse 

import os 
import random 
import numpy as np 

import torch 

from torch_geometric.transforms import Compose 

from train_evaluate import hold_out_code2, hold_out_pcba, hold_out_sbm, hold_out_zinc 

from ogb.graphproppred import PygGraphPropPredDataset 
from torch_geometric.datasets import ZINC 
from torch_geometric.datasets import GNNBenchmarkDataset 


from utils import get_vocab_mapping 
### for data transform
from utils import augment_edge, encode_y_to_arr 

# for data pre-processing 
from utils import segment 
from utils import compute_posenc_stats 
from functools import partial 

from utils import add_zeros 

from model.lgi_gt import LGI_GT 




def set_random_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 


def main(): 

    parser = configargparse.ArgumentParser(description="LGI-GT model training and evaluating") 

    parser.add_argument('--configs', required=False, is_config_file=True) 

    # dataset settings 
    parser.add_argument('--dataset_name', type=str, required=True, choices=["ogbg-code2", "ogbg-molpcba", "PATTERN", "CLUSTER", "ZINC"]) 

    parser.add_argument('--max_seq_len', type=int, default=None, 
                            help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=None,
                            help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument("--max_input_len", type=int, default=None, help="The max input length of transformer input") 
    parser.add_argument('--segment_pooling', type=str, default=None, choices=['mean', 'max', 'sum'])

    parser.add_argument('--num_rw_steps', type=int, default=None) 
    parser.add_argument('--dim_pe', type=int, default=None) 
    parser.add_argument('--in_dim', type=int, default=None) 
    parser.add_argument('--out_dim', type=int, default=None) 
    parser.add_argument('--node_num_types', type=int, default=None) 
    parser.add_argument('--edge_num_types', type=int, default=None) 

    # training settings 
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--scheduler', type=str, default='none', choices=['linear', 'cosine', 'none']) 
    parser.add_argument('--warmup', type=int, default=5) 
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--weight_decay', type=float, default=1e-6) 

    # model settings 
    parser.add_argument('--gconv_dim', type=int, default=128) 
    parser.add_argument('--tlayer_dim', type=int, default=128) 

    parser.add_argument('--gconv_attn_dropout', type=float, default=None) 
    parser.add_argument('--gconv_ffn_dropout', type=float, default=0) 
    parser.add_argument('--tlayer_attn_dropout', type=float, default=0) 
    parser.add_argument('--tlayer_ffn_dropout', type=float, default=0) 
    parser.add_argument('--tlayer_ffn_hidden_times', type=int, default=1) 

    parser.add_argument('--gconv_type', type=str, default='gin', choices=['gin', 'gcn', 'eela']) 
    parser.add_argument('--num_layers', type=int, default=4) 
    parser.add_argument('--num_heads', type=int, default=4) 
    parser.add_argument('--middle_layer_type', type=str, default='none', choices=['none', 'mlp', 'residual']) 
    parser.add_argument('--skip_connection', type=str, default='none', choices=['none', 'long', 'short']) 
    parser.add_argument('--readout', type=str, default=None, choices=['mean', 'add', 'cls']) 
    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn']) 

    parser.add_argument('--out_layer', type=int, default=1) 
    parser.add_argument('--out_hidden_times', type=int, default=1) 

    # other settings 
    parser.add_argument('--save_state', action='store_true') 
    parser.add_argument('--seeds', type=int, default=0) 

    args = parser.parse_args() 


    set_random_seed(args.seeds) 

    dataset_root = None 
    dataset = None 
    train_dataset = None 
    val_dataset = None 
    test_dataset = None 
    num_vocab = None 
    if args.dataset_name == 'ogbg-code2': 
        dataset = PygGraphPropPredDataset(name='ogbg-code2', root='dataset') 
        # pre-processing 
        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list))) 

        split_idx = dataset.get_idx_split() 
        vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)
        dataset.transform = Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len),  segment]) 
        # pre-processing done 
        dataset_root = os.path.abspath(dataset.root) 
        num_vocab=len(vocab2idx)
    elif args.dataset_name == 'ogbg-molpcba': 
        dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root='dataset') 
    elif args.dataset_name == 'PATTERN' or args.dataset_name == 'CLUSTER':
        train_dataset = GNNBenchmarkDataset(root='dataset', name=args.dataset_name, split='train', 
                            pre_transform=Compose([ 
                                            add_zeros, 
                                            partial( 
                                                compute_posenc_stats, 
                                                pe_types=['RWSE'], 
                                                times=list(range(1, args.num_rw_steps + 1))
                                                )
                                            ]) ) 
        val_dataset = GNNBenchmarkDataset(root='dataset', name=args.dataset_name, split='val', 
                            pre_transform=Compose([ 
                                            add_zeros, 
                                            partial( 
                                                compute_posenc_stats, 
                                                pe_types=['RWSE'], 
                                                times=list(range(1, args.num_rw_steps + 1))
                                                )
                                            ]) ) 
        test_dataset = GNNBenchmarkDataset(root='dataset', name=args.dataset_name, split='test', 
                            pre_transform=Compose([ 
                                            add_zeros, 
                                            partial( 
                                                compute_posenc_stats, 
                                                pe_types=['RWSE'], 
                                                times=list(range(1, args.num_rw_steps + 1))
                                                )
                                            ]) ) 
    elif args.dataset_name == 'ZINC': 
        train_dataset = ZINC(root='dataset/ZINC', subset=True, split='train', 
                            pre_transform=partial(
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1))) )
        val_dataset = ZINC(root='dataset/ZINC', subset=True, split='val', 
                            pre_transform=partial(
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1))) ) 
        test_dataset = ZINC(root='dataset/ZINC', subset=True, split='test', 
                            pre_transform=partial(
                                        compute_posenc_stats, 
                                        pe_types=['RWSE'], 
                                        times=list(range(1, args.num_rw_steps + 1))) ) 

    model = LGI_GT( 
                gconv_dim=args.gconv_dim, 
                tlayer_dim=args.tlayer_dim, 
                dataset_name=args.dataset_name, 

                dataset_root=dataset_root, 
                num_vocab=num_vocab, 
                max_seq_len=args.max_seq_len, 
                segment_pooling=args.segment_pooling, 

                in_dim=args.in_dim, 
                out_dim=args.out_dim, 

                node_num_types=args.node_num_types, 
                edge_num_types=args.edge_num_types, 
                dim_pe=args.dim_pe, 
                num_rw_steps=args.num_rw_steps, 
                
                gconv_attn_dropout=args.gconv_attn_dropout, 
                gconv_ffn_dropout=args.gconv_ffn_dropout, 
                tlayer_attn_dropout=args.tlayer_attn_dropout, 
                tlayer_ffn_dropout=args.tlayer_ffn_dropout, 
                tlayer_ffn_hidden_times=args.tlayer_ffn_hidden_times, 
                gconv_type=args.gconv_type, 
                num_layers=args.num_layers, 
                num_heads=args.num_heads, 
                middle_layer_type=args.middle_layer_type, 
                skip_connection=args.skip_connection, 
                readout=args.readout, 
                norm=args.norm, 

                out_layer=args.out_layer, 
                out_hidden_times=args.out_hidden_times) 

    if args.dataset_name == 'ogbg-code2': 
        hold_out_code2(model, dataset, args, idx2vocab) 
    elif args.dataset_name == 'ogbg-molpcba': 
        hold_out_pcba(model, dataset, args) 
    elif args.dataset_name == 'PATTERN' or args.dataset_name == 'CLUSTER': 
        hold_out_sbm(model, train_dataset, val_dataset, test_dataset, args) 
    elif args.dataset_name == 'ZINC': 
        hold_out_zinc(model, train_dataset, val_dataset, test_dataset, args) 


if __name__ == "__main__":
    main() 
    
