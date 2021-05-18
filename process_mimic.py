#coding:utf-8
import datetime
import csv
import _pickle as pickle
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')

'''
处理mimic获得processed_init.pkl文件：{"subject_id":subid_seq, "hadm_id":hadmid_seq, "admittime":admittime_seq, "icds":icds_seq}；按照admittime从小到大排序
'''

if __name__ == '__main__':
    
    dataset_path = 'dataset/mimic3_demo/'
    
    file_adm = dataset_path + 'ADMISSIONS.csv'
    file_icd = dataset_path + 'DIAGNOSES_ICD.csv'

    m_adm = pd.read_csv(file_adm, dtype={'HOSPITAL_EXPIRE_FLAG': object})
    m_icd = pd.read_csv(file_icd, dtype={'ICD9_CODE': object})
    
    subid_seq = []
    hadmid_seq = []
    admittime_seq = []
    icds_seq = []
    
    # get total unique patients
    unique_pats = m_icd.SUBJECT_ID.unique()
    for sub_id in unique_pats:
        pat_icd = m_icd[m_icd.SUBJECT_ID == sub_id]  # get a specific patient's all data in icd file
        uni_hadm = pat_icd.HADM_ID.unique()  # get all unique admissions
        grouped = pat_icd.groupby(['HADM_ID'])
        if len(uni_hadm) < 2:#过滤掉visit少于2的patient
            continue
        for hadm in uni_hadm:
            visit = dict()
            adm = m_adm[(m_adm.SUBJECT_ID == sub_id) & (m_adm.HADM_ID == hadm)]
            admittime = datetime.datetime.strptime(adm.ADMITTIME.values[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

            codes = grouped.get_group(hadm)  # get all diagnosis codes in the adm
            icds = []
            for index, row in codes.iterrows():
                dx = row['ICD9_CODE']
                # if dx is not NaN
                if dx == dx and dx != '':
                    icds.append(dx)
            if icds:#不为空
                subid_seq.append(sub_id)
                hadmid_seq.append(hadm)
                admittime_seq.append(admittime)
                icds_seq.append(icds)
    '''
    frame = pd.DataFrame({"subject_id":subid_seq, "hadm_id":hadmid_seq, "admittime":admittime_seq, "icds":icds_seq})
    frame.to_csv("processed_init.csv" ,index=None)
    
    df = pd.read_csv("processed_init.csv")
    all_data = df.sort_values(by="admittime" , ascending=True)
    all_data.to_csv(dataset_path+"processed_init.csv" ,index=None)
    
    '''
    #=============== sorting visit according to timestamps
    arr = np.array(admittime_seq)
    sorted_idx = np.argsort(arr)
    subid_seq = [subid_seq[i] for i in sorted_idx]
    hadmid_seq = [hadmid_seq[i] for i in sorted_idx]
    admittime_seq = [admittime_seq[i] for i in sorted_idx]
    icds_seq = [icds_seq[i] for i in sorted_idx]
    
    dic = {"subject_id":subid_seq, "hadm_id":hadmid_seq, "admittime":admittime_seq, "icds":icds_seq}
    pickle.dump(dic, open(dataset_path+'process_init.pkl', 'wb'), -1)
    
