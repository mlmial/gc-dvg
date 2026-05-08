import pickle
import os

dcts_path = 'data/picklejar/time2id_dcts'

if not os.path.exists(f'{dcts_path}/tr_id2timepoint.pkl'):
    print('pickling time2id dictionaries')
    data_path = 'data'
    datasets_path = 'data/datasets'

    time2id_dcts_path = f'data/picklejar/time2id_dcts'
    os.makedirs(time2id_dcts_path, exist_ok=True)

    with open(f'{datasets_path}/pelz_2021_long/long_pelz_timepoint_dct_data.tsv','r') as f:
        lines = f.readlines()
        tmp_long_pelz_id2timepoint = {k:f'PN{k.split("_")[-1]}'.replace('II','31.1') for k in lines[0].strip().split('\t')}
        tmp_long_pelz_id2timepoint['VB3_Saat_CGGACAAC-TCCGGATT_24'] = 'PN6'
        long_pelz_id2timepoint = {}
        for k in sorted(tmp_long_pelz_id2timepoint.keys(), 
                        key=lambda x: float(tmp_long_pelz_id2timepoint[x].split('N')[-1])):
            long_pelz_id2timepoint[k] = tmp_long_pelz_id2timepoint[k]

    with open(f'{time2id_dcts_path}/long_pelz_id2timepoint.pkl','wb') as f:
        pickle.dump(long_pelz_id2timepoint, f)
        
long_pelz_id2timepoint = pickle.load(open(f'{dcts_path}/long_pelz_id2timepoint.pkl','rb'))
