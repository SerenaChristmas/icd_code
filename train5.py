#!/usr/bin/env python
# coding: utf-8

import datetime
from model import *
import argparse
from utils import *

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default="dataset/mimic3/data_by_time.pkl")
parser.add_argument('--numpath', default="dataset/mimic3/types.pkl")
parser.add_argument('--filtered_pid_path', default="dataset/mimic3/mimic.filtered_pid")
parser.add_argument('--model_name', default="ModelTree2")
parser.add_argument('--save_model_path', default="save_model/")
parser.add_argument('--hierarchy_file', default='dataset/mimic3/mimic.forgram')
#If set to -1 (default), the GPU with most free memory will be chosen.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--train_proportion', default=0.8, type=float)
parser.add_argument('--dynamic_size', default=128, type=int)
parser.add_argument('--static_size', default=128, type=int)
parser.add_argument('--atten_size', default=48, type=int)
parser.add_argument('--feature_size', default=2, type=int) #interaction feature
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--n_layer', default=2, type=int) #layer of RNN
parser.add_argument('--n_directions', default=1, type=int) #单向RNN

parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for the leaky_relu.')
args = parser.parse_args()

#固定seed
seed_torch(args.seed)

############### initializing ###############
now = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
save_model_path = args.save_model_path + args.model_name + '_' + now + '/'
if os.path.isdir(save_model_path):
    pass
else:
    os.mkdir(save_model_path)
sys.stdout = Logger("log/" + args.model_name + "_" + now + ".txt")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

############### Loading data ###############
print("**************** loading dataset ****************")
train_pid, valid_pid, test_pid, all_pid = load_pid(args.filtered_pid_path)
n_patient, n_visit, n_group_code = load_num(args.numpath) #n_group_code:272
patient_sequence, patient_timediffs_sequence, code_sequence, code_timediffs_sequence, feature_sequence, labels, patient_previous_visit_seq = load_data_1(args.datapath, n_group_code)
print("%d patients\n%d codes\n%d interactions" % (n_patient, n_group_code, n_visit))

############### set boundaries ###############
train_end_idx = validation_start_idx = int(n_visit * args.train_proportion) 
test_start_idx = int(n_visit * (args.train_proportion+0.1))
test_end_idx = int(n_visit * (args.train_proportion+0.2))

############### get last visit ###############
valid_last_visit = defaultdict(int) #记录的是validset中每个patient最后一次visit的idx集合
for idx in range(train_end_idx, test_start_idx):
    patient = patient_sequence[idx]
    if patient in valid_pid:
        valid_last_visit[patient] = idx
valid_last_visit = valid_last_visit.values()
test_last_visit = defaultdict(int)
for idx in range(test_start_idx, test_end_idx):
    patient = patient_sequence[idx]
    if patient in test_pid:
        test_last_visit[patient] = idx
test_last_visit = test_last_visit.values()
print("%d patients in valid set" % (len(valid_last_visit)))
print("%d patients in test set\n" % (len(test_last_visit)))

############### process ICD tree ###############
node_to_idx, newfather, corMat, ini_embds = pickle.load(open(args.hierarchy_file, 'rb'))
#adj邻接矩阵(1代表father-son关系)，corMat中1代表的是ancestor和descendant
n_total_medical_code = corMat.shape[0] #code and ancestor num:332
row = np.array(list(newfather.keys()))
col = np.array(list(newfather.values()))
data = np.ones(len(newfather))
#生成领接矩阵，生成的矩阵为稀疏矩阵，对应的行和列坐标分别为边的两个点，该步骤之后得到的是一个有向图
adj = sp.coo_matrix((data, (row, col)),
                    shape=(n_total_medical_code, n_total_medical_code),
                        dtype=np.float32)
#得到无向图的领接矩阵，对称邻接矩阵
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#进行归一化，实现的是A^=(D~)^-1 A~；A^=I+A
adj = normalize(adj + sp.eye(adj.shape[0])) #(332, 332)
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj = adj.cuda()

############### INITIALIZE MODEL AND PARAMETERS ###############
print('**************** Initializing ' + args.model_name + ' ****************')
model_file = eval(args.model_name)
model = model_file(args, n_patient, n_group_code, n_total_medical_code).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model_eval = model_file(args, n_patient, n_group_code, n_total_medical_code).cuda()
optimizer_eval = optim.Adam(model_eval.parameters(), lr=args.lr, weight_decay=args.weight_decay)

patient_static = Variable(torch.eye(n_patient).cuda()) # one-hot vectors for static embeddings
code_static = Variable(torch.eye(n_total_medical_code).cuda()) # one-hot vectors for static embeddings
idx_group_code = [i for i in range(n_group_code)]

for epoch in range(args.epochs):
    print('\n========== Epoch  %d ==========='%epoch)
    epoch_start_time = datetime.datetime.now()
    
    #initialize dynamic embedding
    patient_dynamic = model.initial_patient_dynamic.cuda().repeat(n_patient, 1) #[n_patient, dynamic_dim]
    code_dynamic = model.initial_code_dynamic.cuda().repeat(n_total_medical_code, 1) #[n_total_medical_code, dynamic_dim]
    init_code_dynamic = code_dynamic.clone().detach()
    ############### train process ###############
    train_losses = []
    model.train()

    for j in range(train_end_idx):
        patientid = patient_sequence[j]
        if patientid not in train_pid:
            continue
        
        patientid_tensor = torch.LongTensor([patientid]).cuda()
        codeid_tensor = torch.LongTensor(code_sequence[j]).cuda()
        ancestorid_tensor = torch.LongTensor(get_ancestor(code_sequence[j], newfather)).cuda()
        target = torch.FloatTensor(labels[j]).cuda().unsqueeze(0)#[1, n_group_code]
        patient_timediff_tensor=Variable(torch.Tensor([patient_timediffs_sequence[j]]).cuda()).unsqueeze(1) #[1,1]
        code_timediff_tensor = Variable(torch.Tensor(code_timediffs_sequence[j]).cuda()).unsqueeze(1) #[codeid_num,1]
        feature_tensor = Variable(torch.Tensor(feature_sequence[j]).unsqueeze(0).cuda()) #[1, feature_dim]
        
        # PROJECT patient EMBEDDING TO CURRENT TIME
        patient_projected = model.forward(patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor, select='project') #[1, dynamic_dim]
        
        # CALCULATE PREDICTION LOSS
        train_loss = 0
        output = model.predict_labels(patient_projected, patient_static[patientid].unsqueeze(0), code_dynamic[idx_group_code], code_static[idx_group_code]) #[1, n_total_medical_code]
        train_loss += bce_loss(output, target)

        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
        patient_embedding_output = model.forward(patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor,  select='patient_update') #[1, dynamic_dim]
        
        code_embedding_output = model.forward(patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor.repeat(len(codeid_tensor), 1), timediffs=code_timediff_tensor,  select='code_update') #[codeid_num, dynamic_dim]
        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        train_loss += mse_loss(code_embedding_output, code_dynamic[codeid_tensor].detach())
        train_loss += mse_loss(patient_embedding_output, patient_dynamic[patientid_tensor].detach())
        
        #update dynamic embedding
        patient_dynamic[patientid_tensor] = patient_embedding_output
        code_dynamic[codeid_tensor] = code_embedding_output
        
        #update ancestor dynamic embeddings
        ancestor_embedding_output = model.forward(patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor,  select='ancestor_update') #[ancestorid_num, dynamic_dim]
        #calculate loss to maintain temporal smoothness
        train_loss += mse_loss(ancestor_embedding_output, code_dynamic[ancestorid_tensor].detach())
        code_dynamic[ancestorid_tensor] = ancestor_embedding_output
        
        # BACKPROPAGATE ERROR AFTER END OF a batch
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.cpu().item())

        # RESET FOR NEXT T-BATCH
        patient_dynamic.detach_() # Detachment is needed to prevent double propagation of gradient
        code_dynamic.detach_()
        
    print("train_loss: ", round(np.average(train_losses), 5))
    param_norm = 0
    grad_norm = 0
    for param_tensor in model.parameters():
        param_norm += torch.norm(param_tensor)
        grad_tensor = param_tensor.grad
        grad_norm += torch.norm(grad_tensor).data
    print('param_norm: %.3f' % param_norm.data)
    print('grad_norm: %.3f' % grad_norm.data)
    ############### end of train process ###############

    ############### save and load model  ###############
    filename = save_model_path + str(epoch)+".model"
    save_model_1(model, optimizer, epoch, filename)
    model_eval, optimizer_eval = load_model(model_eval, optimizer_eval, filename)    
    patient_dynamic_eval = patient_dynamic.clone().detach()
    code_dynamic_eval = code_dynamic.clone().detach()

    ############### eval process  ###############
    outputs_valid = []
    targets_valid = []
    outputs_test = [] 
    targets_test = []
    
    for j in range(train_end_idx, test_end_idx):
        patientid = patient_sequence[j]
        if patientid not in (valid_pid | test_pid):
            continue
        
        patientid_tensor = torch.LongTensor([patientid]).cuda()
        codeid_tensor = torch.LongTensor(code_sequence[j]).cuda()
        ancestorid_tensor = torch.LongTensor(get_ancestor(code_sequence[j], newfather)).cuda()
        target = torch.FloatTensor(labels[j]).cuda().unsqueeze(0)#[1, n_group_code] one-hot
        patient_timediff_tensor=Variable(torch.Tensor([patient_timediffs_sequence[j]]).cuda()).unsqueeze(1) #[1,1]
        code_timediff_tensor = Variable(torch.Tensor(code_timediffs_sequence[j]).cuda()).unsqueeze(1) #[codeid_num,1]
        feature_tensor = Variable(torch.Tensor(feature_sequence[j]).unsqueeze(0).cuda()) #[1, feature_dim]

        # PROJECT patient EMBEDDING TO CURRENT TIME
        patient_projected = model_eval.forward(patient_dynamic_eval, code_dynamic_eval, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor, select='project') #[1, dynamic_dim]patient_projecied_embeds_final = torch.cat([patient_projected, patient_static[patientid].unsqueeze(0)],dim=1)#[1,dynamic_dim + static_dim]
        
        # CALCULATE PREDICTION LOSS
        eval_loss = 0
        output = model_eval.predict_labels(patient_projected, patient_static[patientid].unsqueeze(0), code_dynamic_eval[idx_group_code], code_static[idx_group_code]) #[1, n_total_medical_code]
        eval_loss += bce_loss(output, target)
        
        if j in valid_last_visit:
            targets_valid.append(target[0].data.cpu().tolist())
            outputs_valid.append(output[0].data.cpu().tolist())
        if j in test_last_visit:
            targets_test.append(target[0].data.cpu().tolist())
            outputs_test.append(output[0].data.cpu().tolist())
            '''
            if flag[patientid] == 1:
                targets_test.append(target[0].data.cpu().tolist())
                outputs_test.append(output[0].data.cpu().tolist())
            else:
                flag[patientid] = 1
            '''
        # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
        patient_embedding_output = model_eval.forward(patient_dynamic_eval, code_dynamic_eval, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor,  select='patient_update') #[1, dynamic_dim]
        code_embedding_output = model_eval.forward(patient_dynamic_eval, code_dynamic_eval, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor.repeat(len(codeid_tensor), 1), timediffs=code_timediff_tensor,  select='code_update') #[codeid_num, dynamic_dim]

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        eval_loss += mse_loss(code_embedding_output, code_dynamic_eval[codeid_tensor].detach())
        eval_loss += mse_loss(patient_embedding_output, patient_dynamic_eval[patientid_tensor].detach())
        #update dynamic embedding
        patient_dynamic_eval[patientid_tensor] = patient_embedding_output
        code_dynamic_eval[codeid_tensor] = code_embedding_output
        
        #update ancestor dynamic embeddings
        ancestor_embedding_output = model_eval.forward(patient_dynamic_eval, code_dynamic_eval, init_code_dynamic, adj, patientid_tensor, codeid_tensor, ancestorid_tensor, feature_tensor, timediffs=patient_timediff_tensor,  select='ancestor_update') #[ancestorid_num, dynamic_dim]
        #calculate loss to maintain temporal smoothness
        eval_loss += mse_loss(ancestor_embedding_output, code_dynamic_eval[ancestorid_tensor].detach())
        code_dynamic_eval[ancestorid_tensor] = ancestor_embedding_output        
        # BACKPROPAGATE ERROR AFTER END OF a batch
        optimizer_eval.zero_grad()
        eval_loss.backward()
        optimizer_eval.step()
            
        patient_dynamic_eval.detach_() # Detachment is needed to prevent double propagation of gradient
        code_dynamic_eval.detach_()
    
    ############### end of eval process  ###############
    print("time of this epoch: ", datetime.datetime.now() - epoch_start_time)
    
    ############### calculating metrics  ###############
    k_list = [5, 10, 15]
    valid_metrics = evaluate(outputs_valid, targets_valid, k_list)
    test_metrics = evaluate(outputs_test, targets_test, k_list)
    print('\n')
    for key in valid_metrics:
        print('valid: %s: %.3f' % (key, valid_metrics[key]))
    print('\n')
    for key in test_metrics:
        print('test: %s: %.3f' % (key, test_metrics[key]))        
    ############### end of epoch  ###############
    