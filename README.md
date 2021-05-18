# icd_code

#### configuration

python3.7.9
numpy==1.19.2
pytorch==1.4.0



#### Run顺序

​    1、process_mimic.py
​        处理mimic3获得processed_init.pkl文件
​        {"subject_id":subid_seq, "hadm_id":hadmid_seq, "admittime":admittime_seq, "icds":icds_seq}
​        按照admittime从小到大排序
​    2、process_init.py
​        处理processed_init.pkl文件，获得data_by_time.pkl和types.pkl
​        data_by_time.pkl：{"patient_id":patientid_seq, "visit_id":visitid_seq, "timestamp":timestamp_seq, "icds_idx": icds_idx_seq, "cids_idx":cids_idx_seq}
​            timestamp：和第一个时间点相差的天数
​        types.pkl：{"patient_num": len(subjectid_to_patientid), "visit_num": len(hadmid_to_visitid), "icd_code_num": len(icd_to_index), "group_code_num": len(cid_to_index), 
​             "subjectid_to_patientid":subjectid_to_patientid, "hadmid_to_visitid":hadmid_to_visitid, "icd_to_index":icd_to_index, "cid_to_index":cid_to_index}
​    3、get_hierarchy.py：处理ccs_single.txt和ccs_multi.txt，获得mimic.forgram
​    4、train.py



#### dataset

mimic3：PATIENT.csv、DIAGNOSES_ICD.csv、ADMISSIONS.csv
mimic.forgram文件：(node to index, newfather, cormat and pre-trained embeddings)
            node_to_idx：前面是cid to idx int:int,后面是father node to idx str:int
            father_idx = newfather[son_idx]
            cormat：用矩阵方式记录ICD tree；shape：(len(types), len(types))，每一行代表一个node，数值为1表示对应的列是其father or ancestor；每一列数为1代表其son
            embeddings dim：128
            利用CAMP模型获得的



#### model介绍

'''
Model && train1.py

ModelTreeInit && train1.py
    dynamic code initial embedding：随机初始化
    添加了ICD tree，利用graph attention来更新leaf node，ancestor未更新
    
ModelTree && train2.py
    添加了ICD tree，使用更新后的leaf node和graph attention更新ancestor，矩阵方式更新
    根据leaf到root的路径获得ancestor集合，更新ancestor时使用该ancestor所有的descentant来进行更新
    一个epoch：24min

ModelTree1 && train3.py
    添加了ICD tree
    只更新相关的leaf or ancestor，而非先全部更新，再局部更新，使用graph attention更新
'''

ModelTree2 && train5.py
    使用GCN: AXW更新leaf code和ancestor
    static embedding从model中拿到了train中
        
ModelTree2_1 && train5.py
    GCN_1: AXW => alpha * AX^{k}W + (1-alpha) X^{0}

ModelTree3 && train6.py
    GCN_1: AXW => alpha * AX^{k}W + (1-alpha) X^{0}
    code update: 𝐶(𝑡'')=GCN(𝐶(𝑡𝑘-))
    feature vector：x(n) = GRU( h(n-1) + x(n))

ModelTree3_0 && train6.py
    GCN_1: AXW => AX^{k}W +  X^{0}
    code update: 𝐶(𝑡'')=GCN_1(𝐶(𝑡𝑘-))
    feature vector：x(n) = GRU( h(n-1) + x(n))

ModelTree3_1 && train6.py
    GCN: AXW
    code update: 𝐶(𝑡'')=GCN(𝐶(𝑡𝑘-))
    feature vector update：x(n) = GRU( h(n-1) + x(n))，n是一个patient第n次就诊
            
ModelTree3_2 && train6.py
    GCN: AXW
    code update: 𝐶(𝑡'') = alpha*GCN(𝐶(𝑡𝑘-)) + (1-alpha)*𝐶(𝑡0)
    feature vector update：x(n) = GRU( h(n-1) + x(n))，n是一个patient第n次就诊

ModelTree3_3 && train6.py
    GCN: AXW
    code update: 𝐶(𝑡'') = alpha*GCN(𝐶(𝑡𝑘-)) + (1-alpha)*𝐶(𝑡k-)
    feature vector update：x(n) = GRU( h(n-1) + x(n))，n是一个patient第n次就诊
    
ModelTree3_4 && train6.py
    GCN: AXW
    code update: 𝐶(𝑡'') = g(x)*GCN(𝐶(𝑡𝑘-)) + (1-g(x))*𝐶(𝑡k-)
        g(x)=sigmoid(W.𝐶(𝑡𝑘-)) #linear layer
        y=f(x)+x = > y=g(x) o f(x) + (1-g(x)) o x; g(x)=sigmoid(Wx)
    feature vector update：x(n) = GRU( h(n-1) + x(n))，n是一个patient第n次就诊
    
