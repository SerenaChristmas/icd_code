import datetime
import csv
import pandas as pd
import json
import _pickle as pickle
import warnings
warnings.filterwarnings('ignore')

'''
处理processed_init.pkl文件，
获得data_by_time.pkl：{"patient_id":patientid_seq, "visit_id":visitid_seq, "timestamp":timestamp_seq, "icds_idx": icds_idx_seq, "cids_idx":cids_idx_seq}；timestamp：和第一个时间点相差的天数
获得types.pkl：{"patient_num", "visit_num", "icd_code_num", "group_code_num", "subjectid_to_patientid", "hadmid_to_visitid", "icd_to_index" "cid_to_index"}
'''

#获得cid(ccs_single.txt的category)
def icdMapper():
    #icd9: 15073; cid: 283
    singleFile = open('dataset/ccs/ccs_single.txt', 'r')
    cnt = 0
    icd2Cid = {}
    cidSet = set()
    for l in singleFile:
        if l[0].isdigit():
            line = l.strip().split()
            curCid = int(line[0])
            cidSet.add(curCid)
        else:
            line = l.strip().split()
            for icd in line:
                icd2Cid[icd] = curCid
    singleFile.close()
    #print(len(icd2Cid))#15072
    return icd2Cid

if __name__ == '__main__':
    '''
    all_data = pd.read_csv("processed_init.csv")
    sub_seq = all_data.subject_id
    hadm_seq = all_data.hadm_id
    admittime_seq = all_data.admittime
    icds_seq = all_data.icds
    sub_unique = sub_seq.unique()
    '''
    path = "dataset/mimic3_demo/"
    all_data = pickle.load(open(path + "process_init.pkl", 'rb'))
    sub_seq = all_data["subject_id"]
    hadm_seq = all_data["hadm_id"]
    admittime_seq = all_data["admittime"]
    icds_seq = all_data["icds"]
    sub_unique = set(sub_seq)
    
    subjectid_to_patientid = {}
    cnt = 0
    for subid in sub_unique:
        subjectid_to_patientid[subid] = cnt
        cnt = cnt + 1
    patientid_seq = [subjectid_to_patientid[subid] for subid in sub_seq]
    
    hadmid_to_visitid = {}
    cnt = 0
    for hadm_id in hadm_seq:
        hadmid_to_visitid[hadm_id] = cnt
        cnt = cnt + 1
    visitid_seq = [hadmid_to_visitid[hadm_id] for hadm_id in hadm_seq]
    
    start_time = datetime.datetime.strptime(admittime_seq[0], "%Y-%m-%d")
    timestamp_seq = [(datetime.datetime.strptime(time, "%Y-%m-%d") - start_time).days for time in admittime_seq]
    
    #获得cids_seq，icd_to_index
    print("build ics grouper with css single layer file")
    icd2Cid = icdMapper()
    cids_seq = []
    icd_to_index = {}
    for icds in icds_seq:
        for icd in icds:
            if icd not in icd_to_index:
                icd_to_index[icd] = len(icd_to_index)
        cids_seq.append(set([icd2Cid[icd] for icd in icds]))#去掉重复的
    
    #获得cid_to_index
    cid_to_index = {}
    for cids in cids_seq:
        for cid in cids:
            if cid not in cid_to_index:
                cid_to_index[cid] = len(cid_to_index)
    
    #把icd_seq和cid_seq都转化为index_seq
    icds_idx_seq = []
    for icds in icds_seq:
        icds_idx_seq.append([icd_to_index[icd] for icd in icds])

    cids_idx_seq = []
    for cids in cids_seq:
        cids_idx_seq.append([cid_to_index[cid] for cid in cids])
    print("patient_num: ", len(subjectid_to_patientid))#100
    print("visit_num: ", len(hadmid_to_visitid))#129
    print("icd_code_num: ", len(icd_to_index))#581
    print("group_code_num: ", len(cid_to_index))#168
    
    '''
    frame = pd.DataFrame({"patient_id":patientid_seq, "visit_id":visitid_seq, "timestamp":timestamp_seq, "icds_idx": icds_idx_seq, "cids_idx":cids_idx_seq})
    frame.to_csv("processed_idx.csv" ,index=None)
    
    #获得num.json文件，存储的是dataset中出现的patient num, visit num, icd num, cid num
    dic = {"patient_num": len(subjectid_to_patientid), "visit_num": len(hadmid_to_visitid), "icd_code_num": len(icd_to_index), "group_code_num": len(cid_to_index)}
    f = open('types.json', mode='w', encoding='utf-8')
    f.write(json.dumps(dic)+'\n')
    f.write(json.dumps(subjectid_to_patientid)+'\n')
    f.write(json.dumps(hadmid_to_visitid)+'\n')
    f.write(json.dumps(icd_to_index)+'\n')
    f.write(json.dumps(cid_to_index))
    '''
    
    dic0 = {"patient_id":patientid_seq, "visit_id":visitid_seq, "timestamp":timestamp_seq, "icds_idx": icds_idx_seq, "cids_idx":cids_idx_seq}
    pickle.dump(dic0, open(path+'data_by_time.pkl', 'wb'), -1)
    
    dic = {"patient_num": len(subjectid_to_patientid), "visit_num": len(hadmid_to_visitid), "icd_code_num": len(icd_to_index), "group_code_num": len(cid_to_index), "subjectid_to_patientid":subjectid_to_patientid, "hadmid_to_visitid":hadmid_to_visitid, "icd_to_index":icd_to_index, "cid_to_index":cid_to_index}
    pickle.dump(dic, open(path+'types.pkl', 'wb'), -1)
    

    
    