

from collections import Counter 
import numpy as np
import torch 

import torch_geometric.utils as pyg_utils 

from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_scatter import scatter_add 

import torch.nn.functional as F 
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_dense_adj)


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab

def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim = 0)
    edge_attr_ast_inverse = torch.cat([torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim = 1)


    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1,) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)


    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))


    data.edge_index = torch.cat([edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim = 1)
    data.edge_attr = torch.cat([edge_attr_ast,   edge_attr_ast_inverse, edge_attr_nextoken,  edge_attr_nextoken_inverse], dim = 0)

    return data

def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data 
    '''

    # PyG >= 1.5.0
    seq = data.y
    
    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data

def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''


    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1, as_tuple=False) # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)] # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def test():
    seq_list = [['a', 'b'], ['a', 'b', 'c', 'df', 'f', '2edea', 'a'], ['eraea', 'a', 'c'], ['d'], ['4rq4f','f','a','a', 'g']]
    vocab2idx, idx2vocab = get_vocab_mapping(seq_list, 4)
    print(vocab2idx)
    print(idx2vocab)
    print()
    assert(len(vocab2idx) == len(idx2vocab))

    for vocab, idx in vocab2idx.items():
        assert(idx2vocab[idx] == vocab)


    for seq in seq_list:
        print(seq)
        arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len = 4)[0]
        # Test the effect of predicting __EOS__
        # arr[2] = vocab2idx['__EOS__']
        print(arr)
        seq_dec = decode_arr_to_seq(arr, idx2vocab)

        print(arr)
        print(seq_dec)
        print('')


################################## 
# data pre-processing 
################################## 

def segment(data): 
    '''
    Input:
        data: PyG data object
    Output:
        data (segmented) 
    ''' 

    num_nodes = data.num_nodes 
    node_segment_batch = torch.zeros(num_nodes, dtype=torch.int64) 
    num_segments = (num_nodes // 1000 + 1) if num_nodes % 1000 != 0 else (num_nodes // 1000) 
    for i in range(num_segments): 
        if i < num_segments - 1: 
            node_segment_batch[i*1000: (i+1)*1000] = i 
        else: 
            node_segment_batch[i*1000: num_nodes] = i 
    subgraphs_to_graph_batch = torch.zeros(num_segments, dtype=torch.int64) 

    data.node_segment_batch = node_segment_batch 
    data.subgraphs_to_graph_batch = subgraphs_to_graph_batch 

    return data 

def clip_wrapper(mode): 
    '''
    Input:
        data: PyG data object
    Output:
        data (clipped) 
    ''' 

    def clip(data): 
        num_nodes = data.num_nodes 
        if num_nodes <= 1000: 
            return data 
        
        row, col = data.edge_index 
        if mode == 'head': 
            x = data.x[:1000] 
            node_is_attributed = data.node_is_attributed[:1000] 
            node_dfs_order = data.node_dfs_order[:1000] 
            node_depth = data.node_depth[:1000] 
            row = row < 1000 
            col = col < 1000 
        elif mode == 'tail': 
            x = data.x[-1000:] 
            node_is_attributed = data.node_is_attributed[-1000:] 
            node_dfs_order = data.node_dfs_order[-1000:] 
            node_depth = data.node_depth[-1000:] 
            row = (row >= num_nodes - 1000 )
            col = (col >= num_nodes - 1000 )
        elif mode == 'head+tail': 
            x = torch.cat([data.x[:500], data.x[-500:]], dim=0) 
            node_is_attributed = torch.cat([data.node_is_attributed[:500], data.node_is_attributed[-500:]], dim=0) 
            node_dfs_order = torch.cat([data.node_dfs_order[:500], data.node_dfs_order[-500:]], dim=0) 
            node_depth = torch.cat([data.node_depth[:500], data.node_depth[-500:]], dim=0) 
            row = (row < 500) | (row >= num_nodes - 500) 
            col = (col < 500) | (col >= num_nodes - 500) 

        pick_index = (row & col )
        edge_index = data.edge_index[:, pick_index] 
        if mode == 'tail': 
            edge_index = edge_index - (num_nodes - 1000) 
        elif mode == 'head+tail': 
            edge_index_tmp = edge_index - (num_nodes - 1000) 
            edge_index = torch.where(edge_index >= 500, edge_index_tmp, edge_index) # >= 500 as the tail, id - (num_nodes - 500) + 500 == id - (num_nodes - 1000) 

        edge_attr = data.edge_attr[pick_index] 

        data.num_nodes = 1000 
        data.x = x 
        data.node_is_attributed = node_is_attributed 
        data.node_dfs_order = node_dfs_order 
        data.node_depth = node_depth 
        data.edge_index = edge_index 
        data.edge_attr = edge_attr 

        return data 
    
    return clip 

# SAT used degree to do scaling in tlayer 
def compute_degree(data): 
    deg = 1. / torch.sqrt(1. + pyg_utils.degree(data.edge_index[0], data.num_nodes)) 
    data.degree = deg 

    return data 


def compute_posenc_stats(data, pe_types, times, max_freqs=None, eigvec_norm=None):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    
    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=None, 
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray()) 
        
        # if 'LapPE' in pe_types:
        #     max_freqs=cfg.posenc_LapPE.eigen.max_freqs
        #     eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        # elif 'EquivStableLapPE' in pe_types:  
        #     max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
        #     eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types: 
        rw_landing = get_rw_landing_probs(times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing 

    return data

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        
        P = P.type(torch.float) 
        Pk = P.clone().detach().matrix_power(min(ksteps)) 
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing




def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def add_zeros(data): 
    z = data.edge_index.new_zeros(data.edge_index.shape[1]) 
    data.edge_attr = z 
    return data 


if __name__ == '__main__':
    test()