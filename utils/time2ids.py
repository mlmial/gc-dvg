data_path = '/Users/mia.le/Downloads/dip_analysis/data'
datasets_path = '/Users/mia.le/Downloads/dip_analysis/data/datasets'

with open(f'{datasets_path}/pelz_2021/pelz_timepoint_dct_data.tsv','r') as f:
    lines = f.readlines()
    pelz_id2timepoint = {k:f'PN{k.split("-")[-1]}'.replace('Saat','6') for k in lines[0].strip().split('\t')}

with open(f'{datasets_path}/pelz_2021/long_pelz_timepoint_dct_data.tsv','r') as f:
    lines = f.readlines()
    long_pelz_id2timepoint = {k:f'PN{k.split("_")[-1]}'.replace('II','31.1') for k in lines[0].strip().split('\t')}
    long_pelz_id2timepoint['VB3_Saat_CGGACAAC-TCCGGATT_24'] = 'PN6'
fu_id2timepoint =  {'A1':'pi0',
                    'A2':'pi9',
                    'A3':'pi16',
                    'A4':'pi29',
                    'A44':'pi29.1', \
                    'A5':'pi41'}

tr_id2timepoint = {'C1':'pi1',
                'C2':'pi21',
                'C22':'pi21.1',
                'C3':'pi38',
                'B1':'pi1',
                'B2':'pi23',
                'B22':'pi23.1',
                'B3':'pi39'}

md_id2timepoint = {k:v for k,v in zip(['VB1_Saat','VB1_10','VB1_18','VB1_25','VB1_27','VB1_36','VB1_43','VB1_48', 'VB1_25_II'],
                                      ['PN6',     'PN10',  'PN18',  'PN25',  'PN27',  'PN36',  'PN43',  'PN48',   'PN25.1'])}
