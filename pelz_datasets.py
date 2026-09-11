import utils.dip_utils as du
from time2id_dictionaries import *
import pandas as pd

datasets_dir = 'data/datasets'

# cultivation values
def pelz_cultivation_values(xlsx_filepath='data/datasets/pelz_2021/SC3 SC_ECACC_summary.xlsx',
                            sheet_name='General Parameters',
                            h=10,
                            cultivation_value_columns=['#','Time p.i.1','HA','HA titer', 'Plaque Assay','TCID50' ,'vRNA'],
                            cultivation_value_names = ['timepoint', 'dpi', 'log_ha', 'ha_titer', 'plaque_assay', 'tcid50', 'vrna']):
    pelz_cultivation_info, units = du.load_pelz_datasheet(xlsx_filepath,
                                                            sheet_name=sheet_name,
                                                            h=h)
    cultivation_df = pelz_cultivation_info.filter(cultivation_value_columns)
    cultivation_units = {}
    new_cols = cultivation_value_names
    for col, orig_col in zip(new_cols, cultivation_df.columns):
            cultivation_units[col] = units[orig_col]
    cultivation_df.columns = new_cols
    cultivation_df['ha_titer'] = calculate_ha_titer(cultivation_df, 'log_ha')
        
    #drop empty columns
    cultivation_df = du.drop_row_if_col_isna(cultivation_df, 'timepoint')
    cultivation_df = du.drop_row_if_col_isna(cultivation_df, 'log_ha')
    cultivation_df = cultivation_df.fillna(0)

    # flip dataframe
    cultivation_df_T = cultivation_df.transpose()
    cultivation_df_T.columns = list(cultivation_df_T.loc['timepoint'])
    cultivation_df_T['key'] = list(cultivation_df_T.index)
    cultivation_df_T['unit'] = cultivation_df_T['key'].apply(lambda x: cultivation_units[x])
    cultivation_df_T = cultivation_df_T.drop(index=['timepoint'])
    cultivation_values = cultivation_df_T[['key', 'unit'] + cultivation_df_T.columns[:-2].tolist()]
    return cultivation_values  

def calculate_ha_titer(dataframe, ha_col):
    return [10 ** (ha) * 2 * (10 ** 7) if ha > 0 else None for ha in dataframe[ha_col]]

def split_df_by_segment(df, segment_col='segment'):
    segments = list(df[segment_col].unique())
    segment_dfs = {}
    for segment in segments:
        segment_dfs[segment] = df[df[segment_col] == segment]
    return segment_dfs

# Long Pelz readcounts

def long_readcounts(cutoff=0, frac=False):
    long_readcounts = du.load_pelz_tsv_file(f'{datasets_dir}/pelz_2021/compMatrix_exp2_edited_raw.tsv',
                                            long_pelz_id2timepoint,
                                            cutoff=cutoff)
    long_readcounts.drop(columns=['PN31.1'], inplace=True)
    long_readcounts = long_readcounts[long_readcounts[long_readcounts.columns[4:]].max(axis=1) >= cutoff]
    # rename column
    long_readcounts = long_readcounts.rename(columns={'Unique Keys':'key', 'Segment':'segment',
                                                      'Start':'start', 'End':'end'})
    long_readcounts.segment = long_readcounts['key'].apply(lambda x: x.split('_')[0])
    long_readcounts = long_readcounts.fillna(0)
    if frac:
        long_readcounts = du.calculate_fractional_readcounts(long_readcounts, 4)
    return long_readcounts

def short_readcounts(cutoff=0, frac=False):
    long_readcounts = du.load_pelz_tsv_file(f'{datasets_dir}/pelz_2021/compMatrix_exp2_edited_raw.tsv',
                                            long_pelz_id2timepoint,
                                            cutoff=cutoff)
    long_readcounts.drop(columns=['PN31.1'], inplace=True)
    long_readcounts = long_readcounts[long_readcounts[long_readcounts.columns[4:]].max(axis=1) >= cutoff]
    # rename column
    long_readcounts = long_readcounts.rename(columns={'Unique Keys':'key', 'Segment':'segment',
                                                      'Start':'start', 'End':'end'})
    long_readcounts.segment = long_readcounts['key'].apply(lambda x: x.split('_')[0])
    long_readcounts = long_readcounts.fillna(0)
    long_readcounts['del_len'] = long_readcounts['end'] - long_readcounts['start']
    short_readcounts = long_readcounts[long_readcounts['del_len'] >= 351]
    short_readcounts.drop(columns=['del_len'], inplace=True)
    if frac:
        short_readcounts = du.calculate_fractional_readcounts(short_readcounts, 4)
    return short_readcounts

# transform time_ids to days post infection (dpi)

def timeid2dpi_dct():
    cultivation_values = pelz_cultivation_values()
    timeid2dpi_dct = {k:v for k,v in zip(cultivation_values.columns[2:],cultivation_values.loc['dpi'][2:])}
    return timeid2dpi_dct