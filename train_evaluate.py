    

import torch 
import numpy as np 
import os 
import time 

from sklearn.metrics import confusion_matrix 

import torch.nn.functional as F 
from torch.optim import Adam, AdamW 

# from torch_geometric.data.dataloader import DataLoader # for pyg == 1.7.0 
from torch_geometric.loader import DataLoader # for pyg == 2.0.4 

from transformers.optimization import get_cosine_schedule_with_warmup,\
    get_linear_schedule_with_warmup 

from ogb.graphproppred import Evaluator 

from utils import decode_arr_to_seq 


def record_basic_info(fp, current_time, args): 
    fp.write("\n\n===============================================\n") 
    fp.write(current_time) 
    fp.write("\n") 
    for key, value in args.__dict__.items(): 
        fp.write("\n"+key+": "+str(value)) 
    fp.write("\n\n") 
    fp.flush() 


def hold_out_sbm(model, train_dataset, val_dataset, test_dataset, args): 
    epochs: int = args.epochs 
    lr: float = args.lr 
    warmup: int = args.warmup 
    weight_decay: float = args.weight_decay 
    device = torch.device('cuda') 

    num_classes = args.out_dim 

    exp_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '.', 'exp_result') 
    if not os.path.exists(exp_result_path): 
        os.makedirs(exp_result_path) 
    
    log_path = os.path.join(exp_result_path, 'log') 
    if not os.path.exists(log_path): 
        os.makedirs(log_path) 
    
    result_file = os.path.join(exp_result_path, train_dataset.name + '.txt') 
    current_time = time.localtime() 

    current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 
    current_time_filename =  time.strftime("%Y-%m-%d_%H%M%S", current_time) 
    log_file = os.path.join(log_path, train_dataset.name + '_' + current_time_filename + '.log') 

    if args.save_state: 
        state_path = os.path.join(exp_result_path, 'state', train_dataset.name) 
        if not os.path.exists(state_path): 
            os.makedirs(state_path) 
        state_file = os.path.join(state_path, current_time_filename + '.pt') 

    result_fp = open(result_file, 'a') 
    log_fp = open(log_file, 'w') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    model.to(device) 
    print(f'#Params: {sum(p.numel() for p in model.parameters())}') 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if args.scheduler == 'cosine': 
        scheduler = get_cosine_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'linear': 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'none': 
        scheduler = None 

    # p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    train_curve = [] 
    val_curve = [] 
    test_curve = [] 
    best_val = 0 

    # for epoch in p_bar: 
    for epoch in range(0, epochs): 
        train_loss = train_sbm(model, device, train_loader, optimizer, scheduler, num_classes) 

        train_perf, _ = eval_sbm(model, device, train_loader, num_classes, desc="Eval Train ")
        val_perf, _ = eval_sbm(model, device, val_loader, num_classes, desc="Eval   Val ")
        test_perf, _ = eval_sbm(model, device, test_loader, num_classes, desc="Eval  Test ") 

        train_curve.append(train_perf) 
        val_curve.append(val_perf) 
        test_curve.append(test_perf) 

        if args.save_state and val_curve[-1] > best_val: 
            torch.save(model.state_dict(), state_file) 
            best_val = val_curve[-1] 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]: 6.4f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]: 6.4f} " 
        test_log = f"| Test e: {test_curve[-1]: 6.4f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        # bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        # p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 
        print(file_log) 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve)) 
    # best_val_epoch = np.argmin(np.array(val_curve)) 
    best_val = val_curve[best_val_epoch]

    test_score = test_curve[best_val_epoch] 

    result_log = (f"\nRun time: {run_time}\n"  
                 f"Best Epoch: {best_val_epoch}\n" 
                 f"Val: {best_val:6.4f}\n"  
                 f"Test Score: {test_score:6.4f}\n\n" ) 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 


def train_sbm(model, device, loader, optimizer, scheduler, num_classes): 
    model.train() 
    loss_accum = 0 

    for step, batch in enumerate(loader): 
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch) 
            optimizer.zero_grad() 
            loss, pred_score = weighted_cross_entropy(pred, batch.y) 
            loss.backward() 
            optimizer.step() 

            loss_accum += loss.item() 
            # p_bar.set_description(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
            # if step % 1000 == 0: # for log print into file instead of terminal while using nohup 
            #     print(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
        
        if scheduler: 
            scheduler.step() 

    return loss_accum / (step + 1) 


def accuracy_SBM(pred_int, targets):
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc 


def _get_pred_int(pred_score):
    # if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
    #     return (pred_score > cfg.model.thresh).long()
    # else:
    #     return pred_score.max(dim=1)[1]
    return pred_score.max(dim=1)[1] 


@torch.no_grad() 
def eval_sbm(model, device, loader, num_classes, desc): 
    model.eval() 
    loss_accum = 0.0 
    acc = 0 
    preds = [] 
    trues = [] 

    for step, batch in enumerate(loader): 
        batch = batch.to(device) 
        pred = model(batch) 
        loss, pred_score = weighted_cross_entropy(pred, batch.y) # mean over all steps, so no multiplied by len of each batch.y 
        _true = batch.y.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss_accum += loss.item() 

        preds.append(_pred) 
        trues.append(_true) 
    
    true = torch.cat(trues) 
    pred = torch.cat(preds) 
    pred = _get_pred_int(pred) 

    return accuracy_SBM(pred, true), loss_accum / (step + 1) 


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        return loss, torch.sigmoid(pred)




def hold_out_code2(model, dataset, args, idx2vocab): 
    epochs: int = args.epochs 
    lr: float = args.lr 
    warmup: int = args.warmup 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device('cuda') 

    exp_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '.', 'exp_result') 
    if not os.path.exists(exp_result_path): 
        os.makedirs(exp_result_path) 
    
    log_path = os.path.join(exp_result_path, 'log') 
    if not os.path.exists(log_path): 
        os.makedirs(log_path) 
    
    result_file = os.path.join(exp_result_path, dataset.name + '.txt') 
    current_time = time.localtime() 

    current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 
    current_time_filename =  time.strftime("%Y-%m-%d_%H%M%S", current_time) 
    log_file = os.path.join(log_path, dataset.name + '_' + current_time_filename + '.log') 

    if args.save_state: 
        state_path = os.path.join(exp_result_path, 'state', dataset.name) 
        if not os.path.exists(state_path): 
            os.makedirs(state_path) 
        state_file = os.path.join(state_path, current_time_filename + '.pt') 

    result_fp = open(result_file, 'a') 
    log_fp = open(log_file, 'w') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    model.to(device) 
    print(f'#Params: {sum(p.numel() for p in model.parameters())}') 

    evaluator = Evaluator(dataset.name) 

    split_idx = dataset.get_idx_split() 

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False) 
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False) 

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if args.scheduler == 'cosine': 
        scheduler = get_cosine_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'linear': 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler is None: 
        scheduler = None 

    # p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    train_curve = [] 
    val_curve = [] 
    test_curve = [] 
    best_val = 0 

    # for epoch in p_bar: 
    for epoch in range(0, epochs): 
        train_loss = train_code2(model, device, train_loader, optimizer, scheduler) 

        train_perf = eval_code2(model, device, train_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval Train ")
        val_perf = eval_code2(model, device, val_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval   Val ")
        test_perf = eval_code2(model, device, test_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab), desc="Eval  Test ") 

        train_curve.append(train_perf[dataset.eval_metric]) 
        val_curve.append(val_perf[dataset.eval_metric]) 
        test_curve.append(test_perf[dataset.eval_metric]) 

        if args.save_state and val_curve[-1] > best_val: 
            torch.save(model.state_dict(), state_file) 
            best_val = val_curve[-1] 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]: 6.4f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]: 6.4f} " 
        test_log = f"| Test e: {test_curve[-1]: 6.4f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        # bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        # p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 
        print(file_log) 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve)) 
    best_val = val_curve[best_val_epoch]

    test_score = test_curve[best_val_epoch] 

    result_log = (f"\nRun time: {run_time}\n"  
                 f"Best Epoch: {best_val_epoch}\n" 
                 f"Val: {best_val:6.4f}\n"  
                 f"Test Score: {test_score:6.4f}\n\n" ) 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 


def train_code2(model, device, loader, optimizer, scheduler): 
    model.train() 
    # p_bar = tqdm(loader, desc="Train Iteration") 
    loss_accum = 0
    # for step, batch in enumerate(p_bar):
    for step, batch in enumerate(loader): # for log print into file instead of terminal while using nohup 
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                # print(batch.y_arr[:,i]) 
                loss += F.cross_entropy(pred_list[i].to(torch.float32), batch.y_arr[:,i])

            loss = loss / len(pred_list)
            
            loss.backward()
            optimizer.step()

            loss_accum += loss.item() 
            # p_bar.set_description(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
            if step % 100 == 0: # for log print into file instead of terminal while using nohup 
                print(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 

        if scheduler:
            scheduler.step() 

    # print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1) 

def eval_code2(model, device, loader, evaluator, arr_to_seq, desc):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []
    print(desc+"...") # for log print into file instead of terminal while using nohup
    # p_bar = tqdm(loader, desc=desc+"| Iteration") 
    # for step, batch in enumerate(p_bar): 
    for step, batch in enumerate(loader): # for log print into file instead of terminal while using nohup 
        batch = batch.to(device) 

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat] 
            
            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict) 




def hold_out_pcba(model, dataset, args): 
    epochs: int = args.epochs 
    lr: float = args.lr 
    warmup: int = args.warmup 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device('cuda') 

    exp_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '.', 'exp_result') 
    if not os.path.exists(exp_result_path): 
        os.makedirs(exp_result_path) 
    
    log_path = os.path.join(exp_result_path, 'log') 
    if not os.path.exists(log_path): 
        os.makedirs(log_path) 
    
    result_file = os.path.join(exp_result_path, dataset.name + '.txt') 
    current_time = time.localtime() 

    current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 
    current_time_filename =  time.strftime("%Y-%m-%d_%H%M%S", current_time) 
    log_file = os.path.join(log_path, dataset.name + '_' + current_time_filename + '.log') 

    if args.save_state: 
        state_path = os.path.join(exp_result_path, 'state', dataset.name) 
        if not os.path.exists(state_path): 
            os.makedirs(state_path) 
        state_file = os.path.join(state_path, current_time_filename + '.pt') 

    result_fp = open(result_file, 'a') 
    log_fp = open(log_file, 'w') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    model.to(device) 
    print(f'#Params: {sum(p.numel() for p in model.parameters())}') 

    evaluator = Evaluator(dataset.name) 

    split_idx = dataset.get_idx_split() 

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False) 
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False) 

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if args.scheduler == 'cosine': 
        scheduler = get_cosine_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'linear': 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'none': 
        scheduler = None 

    # p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    train_curve = [] 
    val_curve = [] 
    test_curve = [] 
    best_val = 0 

    # for epoch in p_bar: 
    for epoch in range(0, epochs): 
        train_loss = train_pcba(model, device, train_loader, optimizer, scheduler) 

        train_perf = eval_pcba(model, device, train_loader, evaluator, desc="Eval Train ")
        val_perf = eval_pcba(model, device, val_loader, evaluator, desc="Eval   Val ")
        test_perf = eval_pcba(model, device, test_loader, evaluator, desc="Eval  Test ") 

        train_curve.append(train_perf[dataset.eval_metric]) 
        val_curve.append(val_perf[dataset.eval_metric]) 
        test_curve.append(test_perf[dataset.eval_metric]) 

        if args.save_state and val_curve[-1] > best_val: 
            torch.save(model.state_dict(), state_file) 
            best_val = val_curve[-1] 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]: 6.4f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]: 6.4f} " 
        test_log = f"| Test e: {test_curve[-1]: 6.4f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        # bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        # p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 
        print(file_log) 

        # scheduler.step() 

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    best_val_epoch = np.argmax(np.array(val_curve)) 
    best_val = val_curve[best_val_epoch]

    test_score = test_curve[best_val_epoch] 

    result_log = (f"\nRun time: {run_time}\n"  
                 f"Best Epoch: {best_val_epoch}\n" 
                 f"Val: {best_val:6.4f}\n"  
                 f"Test Score: {test_score:6.4f}\n\n" ) 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 


def train_pcba(model, device, loader, optimizer, scheduler): 
    model.train() 
    loss_accum = 0 

    for step, batch in enumerate(loader): 
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = F.binary_cross_entropy_with_logits(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step() 

            loss_accum += loss.item() 
            # p_bar.set_description(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
            # if step % 1000 == 0: # for log print into file instead of terminal while using nohup 
            #     print(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
        
        if scheduler: 
            scheduler.step() 

    return loss_accum / (step + 1) 


def eval_pcba(model, device, loader, evaluator, desc):
    model.eval()
    y_true = []
    y_pred = []

    # print(desc+"...") # for log print into file instead of terminal while using nohup

    for step, batch in enumerate(loader): 
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict) 




def hold_out_zinc(model, train_dataset, val_dataset, test_dataset, args): 
    epochs: int = args.epochs 
    lr: float = args.lr 
    warmup: int = args.warmup 
    weight_decay: float = args.weight_decay 
    # device: str = "cuda:" + str(args.device) # "cuda:0" 
    device = torch.device('cuda') 

    exp_result_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '.', 'exp_result') 
    if not os.path.exists(exp_result_path): 
        os.makedirs(exp_result_path) 
    
    log_path = os.path.join(exp_result_path, 'log') 
    if not os.path.exists(log_path): 
        os.makedirs(log_path) 
    
    result_file = os.path.join(exp_result_path, 'ZINC' + '.txt') 
    current_time = time.localtime() 

    current_time_log = time.strftime("%Y-%m-%d %H:%M:%S", current_time) 
    current_time_filename =  time.strftime("%Y-%m-%d_%H%M%S", current_time) 
    log_file = os.path.join(log_path, 'ZINC' + '_' + current_time_filename + '.log') 

    if args.save_state: 
        state_path = os.path.join(exp_result_path, 'state', 'ZINC') 
        if not os.path.exists(state_path): 
            os.makedirs(state_path) 
        state_file = os.path.join(state_path, current_time_filename + '.pt') 

    result_fp = open(result_file, 'a') 
    log_fp = open(log_file, 'w') 

    record_basic_info(log_fp, current_time_log, args) 
    log_fp.write("\n\n") 

    model.to(device) 
    print(f'#Params: {sum(p.numel() for p in model.parameters())}') 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if args.scheduler == 'cosine': 
        scheduler = get_cosine_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'linear': 
        scheduler = get_linear_schedule_with_warmup( 
            optimizer, warmup*len(train_loader), epochs*len(train_loader)) 
    elif args.scheduler == 'none': 
        scheduler = None 

    # p_bar = tqdm(range(0, epochs), bar_format='{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') 

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter() 

    train_curve = [] 
    val_curve = [] 
    test_curve = [] 
    best_val = 99999999 

    # for epoch in p_bar: 
    for epoch in range(0, epochs): 
        train_loss = train_zinc(model, device, train_loader, optimizer, scheduler) 

        train_perf = eval_zinc(model, device, train_loader, desc="Eval Train ")
        val_perf = eval_zinc(model, device, val_loader, desc="Eval   Val ")
        test_perf = eval_zinc(model, device, test_loader, desc="Eval  Test ") 

        train_curve.append(train_perf) 
        val_curve.append(val_perf) 
        test_curve.append(test_perf) 

        if args.save_state and val_curve[-1] < best_val: 
            torch.save(model.state_dict(), state_file) 
            best_val = val_curve[-1] 

        epoch_log = f"| Epoch{epoch: 5d} " 
        train_log = f"| Train e: {train_curve[-1]: 6.4f}, T l: {train_loss: 6.4f} " 
        val_log = f"| Val e: {val_curve[-1]: 6.4f} " 
        test_log = f"| Test e: {test_curve[-1]: 6.4f} " 
        file_log = epoch_log + train_log + val_log + test_log 
        # bar_log = train_log + val_log + test_log 
        log_fp.write(file_log + "\n") 
        log_fp.flush() 
        # p_bar.set_description(bar_log) # tqdm cannot set multi-line description, so only bar_log 
        print(file_log) 

    if torch.cuda.is_available(): 
            torch.cuda.synchronize() 
    end_time = time.perf_counter() 
    run_time = end_time - start_time 

    # best_val_epoch = np.argmax(np.array(val_curve)) 
    best_val_epoch = np.argmin(np.array(val_curve)) 
    best_val = val_curve[best_val_epoch]

    test_score = test_curve[best_val_epoch] 

    result_log = (f"\nRun time: {run_time}\n"  
                 f"Best Epoch: {best_val_epoch}\n" 
                 f"Val: {best_val:6.4f}\n"  
                 f"Test Score: {test_score:6.4f}\n\n" ) 

    record_basic_info(result_fp, current_time_log, args) 
    result_fp.write(result_log) 
    result_fp.flush() 

    print(result_log) 


def train_zinc(model, device, loader, optimizer, scheduler): 
    model.train() 
    loss_accum = 0 

    for step, batch in enumerate(loader): 
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch) 
            optimizer.zero_grad() 
            loss = F.l1_loss(pred, batch.y.unsqueeze(-1)) 
            loss.backward() 

            optimizer.step() 

            loss_accum += loss.item() 
            # p_bar.set_description(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
            # if step % 1000 == 0: # for log print into file instead of terminal while using nohup 
            #     print(f"Train (loss = {loss.item():.4f}, smoothed = {loss_accum / (step + 1):.4f})") 
        
        if scheduler: 
            scheduler.step() 

    return loss_accum / (step + 1) 


@torch.no_grad() 
def eval_zinc(model, device, loader, desc): 
    model.eval() 
    mae_loss = 0.0 

    for batch in loader: 
        batch = batch.to(device) 

        output = model(batch) 
        mae_loss += F.l1_loss(output, batch.y.unsqueeze(-1)).item() * len(batch.y) 

    n_sample = len(loader.dataset) 
    epoch_mae = mae_loss / n_sample 
    return epoch_mae 
