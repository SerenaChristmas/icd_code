import torch
import sys, random, os
import numpy as np
import _pickle as pickle
import heapq
from collections import defaultdict
from sklearn.preprocessing import scale
import scipy.sparse as sp
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def seed_torch(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def load_num(file_path):
    dic = pickle.load(open(file_path, 'rb'))
    patient_num = dic["patient_num"]
    visit_num = dic["visit_num"]
    code_num = dic["group_code_num"]
    return patient_num, visit_num, code_num

def load_pid(file_path):
    st = pickle.load(open(file_path, 'rb'))
    train_pid = st[0]
    valid_pid = st[1]
    test_pid = st[2]
    all_pid = train_pid | valid_pid | test_pid
    return train_pid, valid_pid, test_pid, all_pid

def load_data(file_path, time_scaling=True):
    start_timestamp = None
    dic = pickle.load(open(file_path, 'rb'))
    patient_seq = dic['patient_id']
    codes_seq = dic['cids_idx']
    timestamp_seq = dic['timestamp']
        
    patient_seq = np.array(patient_seq)
    codes_seq = np.array(codes_seq)#dim = 2
    timestamp_seq = np.array(timestamp_seq)

    #print("Getting code and patient time difference seq")
    codes_timedifference_seq = []#当前visit中codes距离上次出现的时间差
    codes_current_timestamp = defaultdict(float)
    patient_timedifference_seq = []#当前visit中patient距离上次出现的时间差
    patient_current_timestamp = defaultdict(float)
    for cnt in range(len(codes_seq)):
        timestamp = timestamp_seq[cnt]
        patient = patient_seq[cnt]
        patient_timedifference_seq.append(timestamp - patient_current_timestamp[patient])
        patient_current_timestamp[patient] = timestamp
        temp = []
        for code in codes_seq[cnt]:
            temp.append(timestamp - codes_current_timestamp[code])
            codes_current_timestamp[code] = timestamp
        codes_timedifference_seq.append(temp)

    if time_scaling:
        patient_timedifference_seq = scale(np.array(patient_timedifference_seq) + 1)
        temp = []
        for i in range(len(codes_timedifference_seq)):
            temp.extend(codes_timedifference_seq[i])
        temp = scale(np.array(temp)+1)
        idx = 0
        for i in range(len(codes_timedifference_seq)):
            length = len(codes_timedifference_seq[i])
            codes_timedifference_seq[i] = temp[idx: idx+length]
            idx = idx + length

    return patient_seq, patient_timedifference_seq, codes_seq, codes_timedifference_seq


def load_data_1(file_path, n_group_code, time_scaling=True):
    start_timestamp = None
    dic = pickle.load(open(file_path, 'rb'))
    patient_seq_init = dic['patient_id']
    codes_seq_init = dic['cids_idx']
    timestamp_seq_init = dic['timestamp'] 
    
    patient_seq = []
    codes_seq = []
    codes_timedifference_seq = []#当前visit中codes距离上次出现的时间差
    codes_current_timestamp = defaultdict(float)
    patient_timedifference_seq = []#当前visit中patient距离上次出现的时间差
    patient_current_timestamp = defaultdict(float)
    patient_latest_visitid = defaultdict(int)
    previous_visit_seq = []#同一个patient的当前visit前的visitid，如果该visit是第一次，则存储该visitid
    for cnt in range(len(patient_seq_init)):
        patient = patient_seq_init[cnt]
        patient_seq.append(patient)
        codes_seq.append(codes_seq_init[cnt])
        timestamp = timestamp_seq_init[cnt]
        patient_timedifference_seq.append(timestamp - patient_current_timestamp[patient])
        patient_current_timestamp[patient] = timestamp
        
        if patient in patient_latest_visitid:
            previous_visit_seq.append(patient_latest_visitid[patient])
        else:
            previous_visit_seq.append(cnt)
        patient_latest_visitid[patient] = cnt
        
        temp = []
        for code in codes_seq_init[cnt]:
            temp.append(timestamp - codes_current_timestamp[code])
            codes_current_timestamp[code] = timestamp
        codes_timedifference_seq.append(temp)

    if time_scaling:
        patient_timedifference_seq = scale(np.array(patient_timedifference_seq) + 1)
        temp = []
        for i in range(len(codes_timedifference_seq)):
            temp.extend(codes_timedifference_seq[i])
        temp = scale(np.array(temp)+1)
        idx = 0
        for i in range(len(codes_timedifference_seq)):
            length = len(codes_timedifference_seq[i])
            codes_timedifference_seq[i] = temp[idx: idx+length]
            idx = idx + length
    
    labels = encode_onehot(codes_seq, n_group_code)
    #feature_seq.append(list(map(float,ls[4:])))
    feature_seq = np.zeros((len(patient_seq), 2))
    
    return patient_seq, patient_timedifference_seq, codes_seq, codes_timedifference_seq, feature_seq, labels, previous_visit_seq


def evaluate(outputs, targets, k_list):
    seq_len= len(outputs) #[test_visit_num, n_medical_code]
    outputs = np.array(outputs)
    targets = np.array(targets)
    
    metrics = {}
    for k in k_list:
        macro_precision, micro_precision = get_precision(outputs, targets, k)
        macro_recall, micro_recall = get_recall(outputs, targets, k)
        macro_f1 = get_f1(macro_precision, macro_recall)
        micro_f1 = get_f1(micro_precision, micro_recall)
        macro_map = get_map(outputs, targets, k)
        
        metrics['macro_precision@%d' % k] = macro_precision
        metrics['micro_precision@%d' % k] = micro_precision
        metrics['macro_recall@%d' % k] = macro_recall
        metrics['micro_recall@%d' % k] = micro_recall
        metrics['macro_f1@%d' % k] = macro_f1
        metrics['micro_f1@%d' % k] = micro_f1
        metrics['macro_map@%d' % k] = macro_map
        
    macro_auc, micro_auc = get_auc(outputs, targets)
    metrics['macro_auc'] = macro_auc
    metrics['micro_auc'] = micro_auc
    return metrics 
    
def get_map(yhat_raw, y, k):
    maps = []
    seq_len, dict_size = yhat_raw.shape
    for i in range(seq_len):
        predicts = yhat_raw[i] #[n_medical_code]
        labels = y[i] #[n_medical_code]
        yNum = float(np.sum(labels))
        topK = heapq.nlargest(k, range(dict_size), predicts.take)
        
        hit = 0
        Map = 0
        for r in range(k):
            if labels[topK[r]] == 1:
                hit += 1
                Map += float(hit) / float(r + 1)
        Map /= yNum
        maps.append(Map)
    macro_map = np.mean(maps)
    return macro_map
    
def get_precision(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    precisions = []
    numerators = []
    denominators = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        yNum = float(len(tk))
        precisions.append(num_true_in_top_k / yNum)
        numerators.append(num_true_in_top_k)
        denominators.append(yNum)

    macro_precision = np.mean(precisions)
    micro_precision = np.sum(numerators) / np.sum(denominators)
    #macro_precision[np.isnan(macro_precision)] = 0.
    #micro_precision[np.isnan(micro_precision)] = 0.
    return macro_precision, micro_precision

def get_recall(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    recalls = []
    numerators = []
    denominators = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        yNum = float(y[i,:].sum())
        recalls.append(num_true_in_top_k / yNum)
        numerators.append(num_true_in_top_k)
        denominators.append(yNum)

    macro_recall = np.mean(recalls)
    micro_recall = np.sum(numerators) / np.sum(denominators)
    #macro_recall[np.isnan(macro_recall)] = 0.
    #micro_recall[np.isnan(micro_recall)] = 0.
    return macro_recall, micro_recall
    
def get_auc(yhat_raw, y):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score): 
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    auc_macro = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    ymic = y.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic) 
    auc_micro = auc(fpr["micro"], tpr["micro"])

    return auc_macro, auc_micro

def get_f1(prec, rec):
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1



# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, epoch, patient_dynamic_embedding, code_dynamic_embedding, filename):
    state = {
            'patient_dynamic_embedding': patient_dynamic_embedding.data.cpu().numpy(),
            'code_dynamic_embedding': code_dynamic_embedding.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, filename)

def save_model_1(model, optimizer, epoch, filename):
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, filename)    

# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w+") #如果是追加，用a+

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_ancestor(codeid, newfather):
    ancestors = []
    for code in codeid:
        if code in newfather: #code 264 265没有father
            ancestors.append(newfather[code])
    return ancestors    

def build_tree(corMat, n_group_code):
    infoDict = defaultdict(list)
    for i in range(n_group_code):
        tmpAns = list(np.nonzero(corMat[i])[0])
        ansNum = len(tmpAns)
        tmpLea = [i] * ansNum
        infoDict[ansNum].append((tmpLea, tmpAns, i))
    lens = sorted(list(infoDict.keys()))
    leaveList = []
    ancestorList = []
    cur = 0
    mapInfo = [0] * n_group_code
    for k in lens:#ancestor为数目相同的leaves放在一组里
        leaves = []
        ancestors = []
        for meta in infoDict[k]:
            leaves.append(meta[0])
            ancestors.append(meta[1])
            mapInfo[meta[2]] = cur
            cur += 1
        leaves = torch.LongTensor(leaves).cuda()
        ancestors = torch.LongTensor(ancestors).cuda()
        leaveList.append(leaves)
        ancestorList.append(ancestors)
    return leaveList, leave_ancestors_list, mapInfo #leavesList:[leave]*ancestor_num; mapInfo: leaveid到在该list中idx

def build_tree_1(corMat, n_group_code, n_total_medical_code):
    infoDict = defaultdict(list)
    for i in range(n_group_code):
        tmpAns = list(np.nonzero(corMat[i])[0]) #按照行访问
        ansNum = len(tmpAns)
        tmpLea = [i] * ansNum
        infoDict[ansNum].append((tmpLea, tmpAns, i))
    lens = sorted(list(infoDict.keys()))
    leafList = []
    leaf_ancestor_list = []
    cur = 0
    mapInfo = [0] * n_total_medical_code
    for k in lens:#ancestor为数目相同的leaves放在一组里
        leaves = []
        ancestors = []
        for meta in infoDict[k]:
            leaves.append(meta[0])
            ancestors.append(meta[1])
            mapInfo[meta[2]] = cur
            cur += 1
        leaves = torch.LongTensor(leaves).cuda()
        ancestors = torch.LongTensor(ancestors).cuda()
        leafList.append(leaves)
        leaf_ancestor_list.append(ancestors)
        
    infoDict = defaultdict(list)
    for i in range(n_group_code, n_total_medical_code):
        tmpAns = list(np.nonzero(corMat[:, i])[0]) #按照列访问ancestor
        sonNum = len(tmpAns)
        tmpLea = [i] * sonNum
        infoDict[sonNum].append((tmpLea, tmpAns, i))
    lens = sorted(list(infoDict.keys()))
    
    ancestorList = []
    sonList = []

    for k in lens:#son为数目相同的ancestor放在一组里
        ancestors = []
        sons = []
        for meta in infoDict[k]:
            ancestors.append(meta[0])
            sons.append(meta[1])
            mapInfo[meta[2]] = cur
            cur += 1
        ancestors = torch.LongTensor(ancestors).cuda()
        sons = torch.LongTensor(sons).cuda()
        ancestorList.append(ancestors)   
        sonList.append(sons)
        
    return leafList, leaf_ancestor_list, ancestorList, sonList, mapInfo #leave_ancestor_list记录的是leave的 ancestors；sonList记录的是ancestor的sons; mapInfo: nodeid到在leafList+ancestorList中idx

def build_tree_2(corMat, n_group_code, n_total_medical_code):
    node_list = []
    related_list = []
    cur = 0
    node_seq = []
    
    infoDict = defaultdict(list)
    for i in range(n_group_code):
        tmpAns = list(np.nonzero(corMat[i])[0].astype(int)) #按照行访问
        ansNum = len(tmpAns)
        tmpLea = [i] * ansNum
        infoDict[ansNum].append((torch.LongTensor(tmpLea).cuda(), torch.LongTensor(tmpAns).cuda(), i))
    lens = sorted(list(infoDict.keys()))
    for k in lens: #ancestor为数目相同的leaves放在一组里
        leaves = []
        ancestors = []
        for meta in infoDict[k]:
            node_list.append(meta[0])
            related_list.append(meta[1])
            node_seq.append(meta[2])
            cur += 1
        
    infoDict = defaultdict(list)
    for i in range(n_group_code, n_total_medical_code):
        tmpAns = list(np.nonzero(corMat[:, i])[0].astype(int)) #按照列访问ancestor
        sonNum = len(tmpAns)
        tmpLea = [i] * sonNum
        infoDict[sonNum].append((torch.LongTensor(tmpLea).cuda(), torch.LongTensor(tmpAns).cuda(), i))
    lens = sorted(list(infoDict.keys()))
    for k in lens: #son为数目相同的ancestor放在一组里
        ancestors = []
        sons = []
        for meta in infoDict[k]:
            node_list.append(meta[0])
            related_list.append(meta[1])
            node_seq.append(meta[2])
            cur += 1
    #将node_seq数组进行排序并获取排序后的索引
    node_seq_sorted_index = np.argsort(np.array(node_seq)).tolist()
    node_list = [node_list[i] for i in node_seq_sorted_index]
    related_list = [related_list[i] for i in node_seq_sorted_index]

    return node_list, related_list #node_list前128个记录的是leave node，related_list前128个记录的是leave对应的ancestor；node_list 128之后记录的是ancestor node，related_list 128之后记录的是ancestor对应的descendant

'''
def change_to_onehot(labels, code_num):
    return np.array([int(i in labels) for i in range(n_group_code)])
'''

def encode_onehot(labels, n_group_code):
    labels_onehot = []
    for label in labels:
        labels_onehot.append([int(i in label) for i in range(n_group_code)])
    labels_onehot = np.array(labels_onehot, dtype=np.int32)
    return labels_onehot

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)    

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

    

'''
def coo2torchsparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    #sparse_mx = sparse_mx.tocoo().astype(np.float32) #tocoo([copy])：返回稀疏矩阵的coo_matrix形式, astype()转换数组的数据类型
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # vstack()将两个数组按垂直方向堆叠成一个新数组
	# torch.from_numpy()是numpy中的ndarray转化成pytorch中的tensor
	# Coo的索引
    values = torch.from_numpy(sparse_mx.data) # Coo的值
    shape = torch.Size(sparse_mx.shape) # Coo的形状大小
    return torch.sparse.FloatTensor(indices, values, shape) # sparse.FloatTensor()构造构造稀疏张量

def coo2torchsparse(A):
    """Convert scipy.sparse.coo_matrix to torch.sparse.FloatTensor"""
    if not sp.issparse(A):
        raise TypeError('input matrix should be scipy.sparse')
    if not sp.isspmatrix_coo(A):
        A = A.tocoo()

    v = torch.FloatTensor(A.data)
    i = torch.LongTensor([A.row, A.col])
    shape = torch.Size(A.shape)

    return torch.sparse.FloatTensor(i,v,shape)

def pad_matrix(code_seq, ancestor_seq, code_timediffs_seq):
    lengths_code = [len(seq) for seq in code_seq]  # number of codes for each seq
    maxlen_code = max(lengths_code)
    lengths_ancestor = [len(seq) for seq in code_seq]  # number of codes for each seq
    maxlen_ancestor = max(lengths_ancestor)
    
    batch_code = []
    batch_ancestor = []
    batch_code_timediffs = []
    for item in code_seq:
        item = item + [-1]*(maxlen_code - len(item))
        batch_code.append(item)
    for item in ancestor_seq:
        item = item + [-1]*(maxlen_ancestor - len(item))
        batch_ancestor.append(item)

    for item in code_timediffs_seq:
        item = item.tolist()
        item = item + [-1]*(maxlen_code - len(item))
        batch_code_timediffs.append(item)
    return batch_code, batch_ancestor, batch_code_timediffs, lengths_code
'''

    
    
    