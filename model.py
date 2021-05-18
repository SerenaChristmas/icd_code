import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math
import numpy as np

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)
'''            
class Model(nn.Module):
    def __init__(self, args, n_patient, n_group_code, num_visit, n_total_medical_code, leafList, leaf_ancestor_list, ancestorList, sonList, mapInfo):
        #only used: args, n_patient, n_group_code, num_visit
        super(Model,self).__init__()

        args.dynamic_size = args.dynamic_size
        self.static_size = args.static_size
        args.feature_size = args.feature_size
        self.n_group_code = n_group_code
        self.idx_group_code = torch.LongTensor([i for i in range(self.n_group_code)]).cuda()
        
        #print("Initializing patient and code dynamic embeds")
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))
        self.patient_static = nn.embed(n_patient, self.static_size)
        self.code_static = nn.embed(n_group_code, self.static_size)

        #print("Initializing patient and code encoder")
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        #print("Initializing linear layers")
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + self.static_size*2, 200), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
    def forward(self, patient_dynamic, code_dynamic, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            code_embed = code_dynamic[codeid]#[codeid_num, dynamic_dim]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]

        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed
    
    def predict_labels(self, patient_projected_embed, patientid, code_embed_dynamic):
        patient_projecied_embeds_final = torch.cat([patient_projected_embed, self.patient_static(patientid)],dim=1)#[1,dynamic_dim + static_dim]
        patient_embed_final = patient_projecied_embeds_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_embed_final = torch.cat([code_embed_dynamic[self.idx_group_code], self.code_static(self.idx_group_code)], dim=1) #[n_group_code, dynamic_dim + static_dim]
        cat = torch.cat([patient_embed_final, code_embed_final], dim=1) #[n_group_code, dynamic_dim*2 + static_num*2]
        out = self.linear_layer1(cat) #[n_group_code, 200]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out


class ModelTreeInit(nn.Module):
    def __init__(self, args, n_patient, n_group_code, num_visit, n_total_medical_code, leafList, leaf_ancestor_list, ancestorList, sonList, mapInfo):
        #only used: args, n_patient, n_group_code, num_visit, n_total_medical_code, leafList, leaf_ancestor_list, mapInfo
        super(ModelTreeInit,self).__init__()

        args.dynamic_size = args.dynamic_size
        self.static_size = args.static_size
        args.feature_size = args.feature_size
        args.atten_size = args.atten_size
        self.n_group_code = n_group_code
        self.leafList = leafList
        self.leaf_ancestor_list = leaf_ancestor_list
        self.mapInfo = mapInfo
        self.idx_group_code = torch.LongTensor([i for i in range(self.n_group_code)]).cuda()
        #self.embedMat = torch.Tensor(np.zeros()).cuda()
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))
        #self.initial_patient_dynamic = nn.Parameter(torch.rand(n_patient, args.dynamic_size))
        #self.initial_code_dynamic = nn.Parameter(torch.rand(n_total_medical_code, args.dynamic_size))
        
        self.patient_static = nn.embed(n_patient, self.static_size)
        self.code_static = nn.embed(n_total_medical_code, self.static_size)

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.Wa = nn.Linear(2 * args.dynamic_size, args.atten_size)
        self.Ua = nn.Linear(args.atten_size, 1, bias=False)
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + self.static_size*2, 200), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
    def forward(self, patient_dynamic, code_dynamic, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            
            embedList = []
            for leaves, ancestors in zip(self.leafList, self.leaf_ancestor_list):
                sampleNum, _ = leaves.size()
                leavesEmbd = code_dynamic[leaves.view(1, -1).squeeze(dim=0)].view(
                    sampleNum, -1, args.dynamic_size)
                ancestorsEmbd = code_dynamic[ancestors.view(
                    1, -1).squeeze(dim=0)].view(sampleNum, -1, args.dynamic_size)
                concated = torch.cat([leavesEmbd, ancestorsEmbd], dim=2)  # codeNum * len* 2 dynamic_size
                weights = F.softmax(self.Ua(torch.tanh(self.Wa(concated))), dim=1).transpose(1, 2)
                embedList.append(weights.bmm(code_dynamic[ancestors]).squeeze(dim=1))
            embedMat = torch.cat(embedList, dim=0)#[n_group_code, 128]
            embedMat = embedMat[self.mapInfo[0:self.n_group_code]]#[n_group_code, 128]
            
            code_embed = embedMat[codeid]#[codeid_num, dynamic_dim]
            
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed
    
    def predict_labels(self, patient_projected_embed, patientid, code_embed_dynamic):
        patient_projecied_embeds_final = torch.cat([patient_projected_embed, self.patient_static(patientid)],dim=1)#[1,dynamic_dim + static_dim]
        patient_embed_final = patient_projecied_embeds_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_embed_final = torch.cat([code_embed_dynamic[self.idx_group_code], self.code_static(self.idx_group_code)], dim=1) #[n_group_code, dynamic_dim + static_dim]
        cat = torch.cat([patient_embed_final, code_embed_final], dim=1) #[n_group_code, dynamic_dim*2 + static_num*2]
        out = self.linear_layer1(cat) #[n_group_code, 200]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
    
class ModelTree(nn.Module):
    def __init__(self, args, n_patient, n_group_code, num_visit, n_total_medical_code, leafList, leaf_ancestor_list, ancestorList, sonList, mapInfo):
        super(ModelTree,self).__init__()

        args.dynamic_size = args.dynamic_size
        self.static_size = args.static_size
        args.feature_size = args.feature_size
        args.atten_size = args.atten_size
        self.n_group_code = n_group_code
        self.leafList = leafList
        self.leaf_ancestor_list = leaf_ancestor_list
        self.ancestorList = ancestorList
        self.sonList = sonList
        self.mapInfo = mapInfo
        self.idx_group_code = torch.LongTensor([i for i in range(self.n_group_code)]).cuda()
        #self.embedMat = torch.Tensor(np.zeros()).cuda()
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))
        #self.initial_patient_dynamic = nn.Parameter(torch.rand(n_patient, args.dynamic_size))
        #self.initial_code_dynamic = nn.Parameter(torch.rand(n_total_medical_code, args.dynamic_size))
        
        self.patient_static = nn.embed(n_patient, self.static_size)
        self.code_static = nn.embed(n_total_medical_code, self.static_size)

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.Wa = nn.Linear(2 * args.dynamic_size, args.atten_size)
        self.Ua = nn.Linear(args.atten_size, 1, bias=False)
        self.Wa_1 = nn.Linear(2 * args.dynamic_size, args.atten_size)
        self.Ua_1 = nn.Linear(args.atten_size, 1, bias=False)
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + self.static_size*2, 200), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
    def forward(self, patient_dynamic, code_dynamic, patientid, codeid,  ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            
            embedList = []
            for leaves, ancestors in zip(self.leafList, self.leaf_ancestor_list):
                sampleNum, _ = leaves.size()
                leavesEmbd = code_dynamic[leaves.view(1, -1).squeeze(dim=0)].view(
                    sampleNum, -1, args.dynamic_size)
                ancestorsEmbd = code_dynamic[ancestors.view(
                    1, -1).squeeze(dim=0)].view(sampleNum, -1, args.dynamic_size)
                concated = torch.cat([leavesEmbd, ancestorsEmbd], dim=2)  # codeNum * len* 2 dynamic_size
                weights = F.softmax(self.Ua(torch.tanh(self.Wa(concated))), dim=1).transpose(1, 2)
                embedList.append(weights.bmm(code_dynamic[ancestors]).squeeze(dim=1))
            embedMat = torch.cat(embedList, dim=0)#[n_group_code, 128]
            embedMat = embedMat[self.mapInfo[0:self.n_group_code]]#[n_group_code, 128]
            
            code_embed = embedMat[codeid]#[codeid_num, dynamic_dim]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            embedList_1 = []
            for ancestors, sons in zip(self.ancestorList, self.sonList):
                sampleNum_1, _ = ancestors.size()
                ancestorsEmbd = code_dynamic[ancestors.view(1, -1).squeeze(dim=0)].view(sampleNum_1, -1, args.dynamic_size)
                sonsEmbd = code_dynamic[sons.view(1, -1).squeeze(dim=0)].view(sampleNum_1, -1, args.dynamic_size)
                concated_1 = torch.cat([ancestorsEmbd, sonsEmbd], dim=2)  # ancestorNum * len * 2 dynamic_size
                weights_1 = F.softmax(self.Ua(torch.tanh(self.Wa(concated_1))), dim=1).transpose(1, 2)
                embedList_1.append(weights_1.bmm(code_dynamic[sons]).squeeze(dim=1))
            tmplist = np.zeros((self.n_group_code, args.dynamic_size))
            tmpTeonsor = torch.Tensor(tmplist).cuda() #[n_group_code, 128]
            embedMat_1 = torch.cat(embedList_1, dim=0)#[ancestorNum, 128]
            embedMat_1 = torch.cat([tmpTeonsor, embedMat_1], dim=0)#[n_total_medical_code, 128]，前n_group_code个无效
            embedMat_1 = embedMat_1[self.mapInfo]#[n_total_medical_code, 128]
            ancestor_embed_output = embedMat_1[ancestorid]#[ancestor_num, dynamic_dim]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed
    
    def predict_labels(self, patient_projected_embed, patientid, code_embed_dynamic):
        patient_projecied_embeds_final = torch.cat([patient_projected_embed, self.patient_static(patientid)],dim=1)#[1,dynamic_dim + static_dim]
        patient_embed_final = patient_projecied_embeds_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_embed_final = torch.cat([code_embed_dynamic[self.idx_group_code], self.code_static(self.idx_group_code)], dim=1) #[n_group_code, dynamic_dim + static_dim]
        cat = torch.cat([patient_embed_final, code_embed_final], dim=1) #[n_group_code, dynamic_dim*2 + static_num*2]
        out = self.linear_layer1(cat) #[n_group_code, 200]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out    
    
class ModelTree1(nn.Module):
    def __init__(self, args, n_patient, n_group_code, num_visit, n_total_medical_code, node_list, related_list):
        super(ModelTree1,self).__init__()

        args.dynamic_size = args.dynamic_size
        self.static_size = args.static_size
        args.feature_size = args.feature_size
        args.atten_size = args.atten_size
        self.n_group_code = n_group_code
        self.node_list = node_list
        self.related_list = related_list
        self.idx_group_code = torch.LongTensor([i for i in range(self.n_group_code)]).cuda()
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))
        self.patient_static = nn.embed(n_patient, self.static_size)
        self.code_static = nn.embed(n_total_medical_code, self.static_size)

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.Wa = nn.Linear(2 * args.dynamic_size, args.atten_size)
        self.Ua = nn.Linear(args.atten_size, 1, bias=False)
        self.Wa_1 = nn.Linear(2 * args.dynamic_size, args.atten_size)
        self.Ua_1 = nn.Linear(args.atten_size, 1, bias=False)
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + self.static_size*2, 200), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
    def forward(self, patient_dynamic, code_dynamic, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            embedList = []
            for idx in codeid:
                idx = int(idx)
                leaves = self.node_list[idx]
                ancestors = self.related_list[idx]
                leavesEmbd = code_dynamic[leaves] #[ancestorNum, dynamic_size]
                ancestorsEmbd = code_dynamic[ancestors] #[ancestorNum, dynamic_size]
                concated = torch.cat([leavesEmbd, ancestorsEmbd], dim=1)  #[ancestorNum, dynamic_size*2]
                weights = F.softmax(self.Ua(torch.tanh(self.Wa(concated))), dim=1).transpose(0, 1) #[1, ancestorNum]
                embedList.append(weights.matmul(code_dynamic[ancestors]))
            code_embed = torch.cat(embedList, dim=0) #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            embedList = []
            for idx in ancestorid:
                idx = int(idx)
                ancestors = self.node_list[idx]
                descendants = self.related_list[idx]
                ancestorsEmbd = code_dynamic[ancestors] #[descendantsNum, dynamic_size]
                descendantsEmbd = code_dynamic[descendants] #[descendantsNum, dynamic_size]
                concated = torch.cat([ancestorsEmbd, descendantsEmbd], dim=1)  #[descendantsNum, dynamic_size*2]
                weights = F.softmax(self.Ua(torch.tanh(self.Wa(concated))), dim=1).transpose(0, 1) #[1, descendantsNum]
                embedList.append(weights.matmul(code_dynamic[ancestors]))
                
            ancestor_embed_output = torch.cat(embedList, dim=0) #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed
    
    def predict_labels(self, patient_projected_embed, patientid, code_embed_dynamic):
        patient_projecied_embeds_final = torch.cat([patient_projected_embed, self.patient_static(patientid)],dim=1)#[1,dynamic_dim + static_dim]
        patient_embed_final = patient_projecied_embeds_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_embed_final = torch.cat([code_embed_dynamic[self.idx_group_code], self.code_static(self.idx_group_code)], dim=1) #[n_group_code, dynamic_dim + static_dim]
        cat = torch.cat([patient_embed_final, code_embed_final], dim=1) #[n_group_code, dynamic_dim*2 + static_num*2]
        out = self.linear_layer1(cat) #[n_group_code, 200]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
'''
    
class GCN(nn.Module):
    def __init__(self, in_features, out_features, act = nn.ReLU(), bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.act = act #激活函数
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output =  output + self.bias
        if self.act is not None:
            return self.act(output) 
        return output
        
    
class ModelTree2(nn.Module):
    '''
    使用GCN: AXW更新leaf code和ancestor
    '''
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree2,self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        #self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        #self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Softmax())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN(args.dynamic_size, int(args.dynamic_size))
        self.gc2 = GCN(int(args.dynamic_size), int(args.dynamic_size))
        self.dropout = args.dropout
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            x = self.gc1(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj) #(332, 128)
            code_embed = x[codeid] #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            x = code_dynamic.clone().detach() #(332, 128)
            x = self.gc1(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj) #(332, 128)
            ancestor_embed_output = x[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        '''
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        out = out.t() #[1, n_group_code]
        return out
        '''
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        out = torch.mm(patient_final, code_final.t()) #[1, n_group_code]
        out = F.softmax(out, dim=1)
        #out = F.sigmoid(out)
        return out
        
        
    
class GCN_1(nn.Module):
    def __init__(self, in_features, out_features, alpha, act = nn.ReLU(), bias=False):
        super(GCN_1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.act = act #激活函数
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj, init_input):
        
        #print(input.shape) [332,128]
        #print(adj.shape) [332, 332]
        #print(init_input.shape) [332,128]
        #print(self.weight.shape) #[128, 128]
        support = torch.mm(input, self.weight)
        #print(support.shape) [332,128]
        output = torch.spmm(adj, support)
        output = torch.mul(output, self.alpha) + torch.mul(init_input, (1-self.alpha))

        if self.bias is not None:
            output =  output + self.bias
        if self.act is not None:
            return self.act(output) 
        return output

class ModelTree2_1(nn.Module):
    '''
    GCN_1: AXW => alpha * AX^{k}W + (1-alpha) X^{0}
    '''
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree2_1,self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        #self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        #self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Softmax())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN_1(args.dynamic_size, int(args.dynamic_size), args.alpha)
        self.gc2 = GCN_1(int(args.dynamic_size), int(args.dynamic_size), args.alpha)
        self.dropout = args.dropout
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            x = self.gc1(x, adj, init_code_dynamic)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj, init_code_dynamic) #(332, 128)
            
            code_embed = x[codeid] #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            x = code_dynamic.clone().detach() #(332, 128)
            x = self.gc1(x, adj, init_code_dynamic)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj, init_code_dynamic) #(332, 128)
            ancestor_embed_output = x[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]
    
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        '''
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        out = out.t() #[1, n_group_code]
        out = F.softmax(out, dim=1)
        return out
        '''
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        out = torch.mm(patient_final, code_final.t()) #[1, n_group_code]
        out = F.softmax(out, dim=1)
        #out = F.sigmoid(out)
        return out
        
    
class ModelTree3(nn.Module):
    '''
    GCN_1: AXW => alpha * AX^{k}W + (1-alpha) X^{0}
    code update: 𝐶(𝑡'')=GCN(𝐶(𝑡𝑘-))
    feature vector：x(n) = GRU( h(n-1) + x(n))
    '''
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN_1(args.dynamic_size, int(args.dynamic_size), args.alpha)
        self.gc2 = GCN_1(int(args.dynamic_size), int(args.dynamic_size), args.alpha)
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj, init_code_dynamic)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj, init_code_dynamic) #(332, 128)
            code_embed = x_3[codeid] #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj, init_code_dynamic)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj, init_code_dynamic) #(332, 128)
            ancestor_embed_output = y_3[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out

    
class ModelTree3_0(nn.Module):
    '''
    GCN_1: AXW => AX^{k}W +  X^{0}
    code update: 𝐶(𝑡'')=GCN_1(𝐶(𝑡𝑘-))
    feature vector：x(n) = GRU( h(n-1) + x(n))
    '''
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3_0, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN_2(args.dynamic_size, int(args.dynamic_size), args.alpha)
        self.gc2 = GCN_2(int(args.dynamic_size), int(args.dynamic_size), args.alpha)
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj, init_code_dynamic)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj, init_code_dynamic) #(332, 128)
            code_embed = x_3[codeid] #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj, init_code_dynamic)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj, init_code_dynamic) #(332, 128)
            ancestor_embed_output = y_3[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
    
class ModelTree3_1(nn.Module):
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3_1, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN(args.dynamic_size, int(args.dynamic_size))
        self.gc2 = GCN(int(args.dynamic_size), int(args.dynamic_size))
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj) #(332, 128)
            code_embed = x_3[codeid] #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj) #(332, 128)
            ancestor_embed_output = y_3[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
    
class ModelTree3_2(nn.Module):
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3_2, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        self.alpha = args.alpha
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN(args.dynamic_size, int(args.dynamic_size))
        self.gc2 = GCN(int(args.dynamic_size), int(args.dynamic_size))
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj) #(332, 128)
            code_embed = torch.mul(x_3[codeid], self.alpha) + torch.mul(init_code_dynamic[codeid], (1-self.alpha)) #[codeid_num, 128]
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj) #(332, 128)
            ancestor_embed_output = torch.mul(y_3[ancestorid], self.alpha) + torch.mul(init_code_dynamic[ancestorid], (1-self.alpha)) #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
    
    
class ModelTree3_3(nn.Module):
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3_3, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        self.alpha = args.alpha
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN(args.dynamic_size, int(args.dynamic_size))
        self.gc2 = GCN(int(args.dynamic_size), int(args.dynamic_size))
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj) #(332, 128)
            code_embed = torch.mul(x_3[codeid], self.alpha) + torch.mul(x[codeid], (1-self.alpha)) #[codeid_num, 128]
            
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj) #(332, 128)
            ancestor_embed_output = torch.mul(y_3[ancestorid], self.alpha) + torch.mul(y[ancestorid], (1-self.alpha)) #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out
    
    
class ModelTree3_4(nn.Module):
    def __init__(self, args, n_patient, n_group_code, n_total_medical_code):
        super(ModelTree3_4, self).__init__()
        self.n_patient = n_patient
        self.n_total_medical_code = n_total_medical_code
        self.n_group_code = n_group_code
        self.alpha = args.alpha
        
        ############### Initializing patient and code dynamic embeds ###############
        self.initial_patient_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size), dim=0))
        self.initial_code_dynamic = nn.Parameter(F.normalize(torch.rand(args.dynamic_size).cuda(), dim=0))

        ############### Initializing patient and code encoder ###############
        update_input_size_code = update_input_size_patients = args.dynamic_size + 1 + args.feature_size
        self.code_update = nn.RNNCell(update_input_size_patients, args.dynamic_size)
        self.patient_update = nn.RNNCell(update_input_size_code, args.dynamic_size)
        
        ############### Initializing linear layers ###############
        self.linear_layer1 = nn.Sequential(nn.Linear(args.dynamic_size*2 + args.static_size*2, 256), nn.ReLU())
        self.linear_layer2 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.linear_layer3 = nn.Sequential(nn.Linear(args.dynamic_size, args.dynamic_size), nn.Sigmoid())
        self.patient_static_embed_layer = nn.Sequential(nn.Linear(self.n_patient, args.static_size), nn.ReLU())
        self.code_static_embed_layer = nn.Sequential(nn.Linear(self.n_total_medical_code, args.static_size), nn.ReLU())
        self.embed_layer = NormalLinear(1, args.dynamic_size)
        
        ############### Initializing gcn layers ###############
        self.gc1 = GCN(args.dynamic_size, int(args.dynamic_size))
        self.gc2 = GCN(int(args.dynamic_size), int(args.dynamic_size))
        self.dropout = args.dropout
        
        ############### Initializing GRU layers for features ###############
        self.feature_emb_layer = nn.Sequential(nn.Linear(args.feature_size, args.feature_size), nn.ReLU())
        self.rnn = nn.GRU(args.feature_size, args.hidden_size,  args.n_layer * args.n_directions)
        
    def forward(self, patient_dynamic, code_dynamic, init_code_dynamic, adj, patientid, codeid, ancestorid, features=None, timediffs=None, select=None):
        if select == 'code_update':
            x = code_dynamic.clone().detach() #(332, 128)
            
            x_1 = self.gc1(x, adj)
            x_2 = F.dropout(x_1, self.dropout, training=self.training)
            x_3 = self.gc2(x_2, adj) #(332, 128)
            g = self.linear_layer3(x) #[332, 128]
            res = torch.mul(g, x_3) + torch.mul(1 - g, x) #对应位相乘，[332, 128]
            code_embed = res[codeid] #[codeid_num, 128]
            
            patient_embed = patient_dynamic[patientid].repeat(len(codeid), 1)
            input1 = torch.cat([patient_embed, timediffs, features], dim=1) #[codeid_num, dynamic_dim + 1 + feature_dim]
            code_embed_output = self.code_update(input1, code_embed) #[codeid_num, dynamic_dim]
            
            return F.normalize(code_embed_output) #[codeid_num, dynamic_dim]
        
        elif select == 'ancestor_update':
            y = code_dynamic.clone().detach() #(332, 128)
            y_1 = self.gc1(y, adj)
            y_2 = F.dropout(y_1, self.dropout, training=self.training)
            y_3 = self.gc2(y_2, adj) #(332, 128)
            h = self.linear_layer3(y) #[332, 128]
            res = torch.mul(h, y_3) + torch.mul(1 - h, y) #对应位相乘，[332, 128]
            ancestor_embed_output = res[ancestorid] #[ancestorid_num, 128]
            return F.normalize(ancestor_embed_output) #[ancestorid_num, dynamic_dim]
            
        elif select == 'patient_update':
            patient_embed = patient_dynamic[patientid]
            code_embed = torch.mean(code_dynamic[codeid], keepdim = True,dim=0)
            input2 = torch.cat([code_embed, timediffs, features], dim=1) #[1, dynamic_dim + 1 + feature_dim]
            patient_embed_output = self.patient_update(input2, patient_embed) #[1, dynamic_dim]
            return F.normalize(patient_embed_output)

        elif select == 'project':
            patient_embed = patient_dynamic[patientid] #[1, dynamic_dim]
            patient_projected_embed = patient_embed * (1 + self.embed_layer(timediffs))
            return patient_projected_embed #[1, dynamic_dim]

    def feature_update(self, features, rnn_hidden):
        v = self.feature_emb_layer(features)
        #print(features.shape) #[1, feature_size] [1, 128]
        #print(v.shape) #[1, feature_size]
        #print(rnn_hidden.shape) #[2, 1, hidden_size]
        # v.unsqueeze(0) [1, 1, 128] [seq_len, batch_sz, input_sz]
        output, hidden = self.rnn(v.unsqueeze(0), rnn_hidden)
        return hidden[-1] #[1, hidden_size]
        
    def predict_labels(self, patient_projected, patient_static_onehot, code_dynamic, code_static_onehot):
        patient_static_embed = self.patient_static_embed_layer(patient_static_onehot) #[1, static_dim]
        patient_projected_final = torch.cat([patient_projected, patient_static_embed], dim=1)#[1, dynamic_dim + static_dim]
        patient_final = patient_projected_final.repeat(self.n_group_code,1) #[n_group_code, dynamic_dim + static_dim]
        code_static_embed = self.code_static_embed_layer(code_static_onehot) #[n_group_code, static_dim]
        code_final = torch.cat([code_dynamic, code_static_embed], dim=1) #[n_group_code, dynamic_dim + static_dim]
        patient_code_embed = torch.cat([patient_final, code_final], dim=1) #[n_group_code, dynamic_dim*2 + static_dim * 2]
        out = self.linear_layer1(patient_code_embed) #[n_group_code, 256]
        out = self.linear_layer2(out) #[n_group_code, 1]
        return out.t() #[1, n_group_code]