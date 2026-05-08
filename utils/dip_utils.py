import os
import pandas as pd
from Bio import SeqIO

def load_pelz_dataset(xlsx_filepath, sheet_name=None, h = 0):
    data_dict = pd.read_excel(io=xlsx_filepath,
                              sheet_name=sheet_name,
                              header=h,
                              na_values=["", "None"],
                              keep_default_na=False)
    return data_dict

def drop_row_if_col_isna(df,column):
    """removes a row of a dataframe if a specified column is None

    Args:
        df (pandas DataFrame): the dataframe that should be edited
        column (_type_): the column that shall not be none

    Returns:
        pandas DataFrame: the original dataframe with dropped columns
    """
    for idx,row in df.iterrows():
        if pd.isna(row[column]):
            df = df.drop(index=idx)
    return df

def load_pelz_datasheet(xlsx_filepath, sheet_name, h=0, unit_given=True):
    """return a dataframe from a pelz dataset result sheet in an xlsl file

    Args:
        xlsx_filename (str): path to xlsl file
        sheet_name (str): sheet name that should be loaded
        h (int, optional): row index of the header. Defaults to 0.
        unit_given (bool, optional): states if the unit column is right below the header. Defaults to True.

    Returns:
        _type_: _description_
    """
    df = load_pelz_dataset(xlsx_filepath,sheet_name,h)
    if unit_given:
        unit_column = {k:v for k,v in zip(df.columns, df.iloc[0])}
        df = df.drop(index=0)
    else:
        unit_column = None
    return df, unit_column

def load_pelz_tsv_file(tsv_filepath, id2timepoint_dct, cutoff=0):
    df = pd.read_csv(tsv_filepath,sep='\t')
    # rename ids to timepoints
    df.columns = [id2timepoint_dct[col] if col in id2timepoint_dct else col for col in df.columns]
    new_df = pd.DataFrame()
    tp_start_index = 0
    for col in df.columns:
        if col not in id2timepoint_dct.values():
            new_df[col] = df[col]
            tp_start_index += 1
    for tp in id2timepoint_dct.values():
        new_df[tp] = df[tp]
        
    if cutoff > 0:
        drop_rows = []
        for i,row in new_df.iterrows():
            readcounts = row[tp_start_index:]
            if not any(x >= cutoff for x in readcounts):
                drop_rows.append(i)
        new_df = new_df.drop(drop_rows)
    return new_df

def calculate_fractional_readcounts(df, timepoint_start=4):
    """calculates the fractional values of a pelz dataset at each timepoint

    Args:
        df (pandas DataFrame): the dataframe that should be edited
        timepoint_start (int, optional): the timepoint from which on the fractional data should be calculated. Defaults to 5.

    Returns:
        pandas DataFrame: the original dataframe with fractional data
    """
    new_df = df.copy()
    for col in new_df.columns[timepoint_start:]:
        new_df[col] = new_df[col]/new_df[col].sum()
    return new_df

def values_at_timepoint(dataframe,columnname,time_columnname):
    """create a dictionary of the form {timepoint:value} that returns which value was present at a certain timepoint of the pelz experiments
        timepoints will be rounded to 2 decimals

    Args:
        dataframe (DataFrame): A data frame that contains timepoints assigned to certain measured values
        columnname (str): _description_
        time_columnname (str): _description_

    Returns:
        _type_: a dictionary with timepoints as keys and measured values as values
    """
    values_at_timepoint = dict(zip(dataframe[time_columnname],dataframe[columnname]))
    values_at_timepoint = {abs(round(k,2)):v for k,v in values_at_timepoint.items() if k>=0}
    return values_at_timepoint


def load_boussier_dataset(xlsx_filename, h = 0):
    file_path = os.path.join(xlsx_filename, "boussier_2020", xlsx_filename)
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=h,
                              na_values=["", "None"],
                              keep_default_na=False)
    return data_dict

def get_table_from_xsls(loaded_dataset,table_name):
    df = loaded_dataset[table_name]
    return df

def get_PR8_ref_sequences_dct(data_path='data'):
    reference_sequence = {}
    for f in [f'{data_path}/reference_genomes/PR8_34_MS_H1N1_HA.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_M.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_NA.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_NP.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_NS.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_PA.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_PB1.fasta',
    f'{data_path}/reference_genomes/PR8_34_MS_H1N1_PB2.fasta']:
        fasta_sequences = SeqIO.parse(open(f),'fasta')
        for fseq in fasta_sequences:
            reference_sequence[f.rsplit('.', 1)[0].rsplit('_',1)[1]] = str(fseq.seq)
    return reference_sequence

def get_deleted_sequence(dip_id):
    """return the sequence of the deleted part of a dip
       uses the PR8 reference sequence

    Args:
        dip_id (str): the id of the dip

    Returns:
        str: the sequence of the deleted part of the dip
    """
    pr8_refseq_dct = get_PR8_ref_sequences_dct()
    seg, start, end = dip_id.split('_')
    return pr8_refseq_dct[seg][int(start):int(end)]

def get_PR8_dip_sequence(dip_id, with_gap=True):
    """return the sequence of a dip with or without * for the deleted part
       uses the PR8 reference sequence
    Args:
        dip_id (str): the id of the dip of the form seg_start_end
        with_gap (bool, optional): if True, the deleted part will be replaced by *. Defaults to True.
    
    Returns:
        str: the sequence of the dip"""
    pr8_refseq_dct = get_PR8_ref_sequences_dct()
    seg, start, end = dip_id.split('_')
    fl_seq = pr8_refseq_dct[seg]
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end):]
    del_length = int(end)-int(start) + 1
    if with_gap:
        return seq_head + '*'*del_length + seq_foot
    else:
        return seq_head + seq_foot
    
def split_dip_id(dip_id):
    """split a dip id into its components

    Args:
        dip_id (str): the id of the dip of the form seg_start_end

    Returns:
        tuple: a tuple of the form (seg,start,end)
    """
    seg, start, end = dip_id.split('_')
    return seg, start, end

def rename_segments(df,key_col='Unique Keys',seg_col='Segment'):
    df[seg_col] = [k.split('_')[0] for k in df[key_col]]
    return df

def split_dataframe_by_col(df, col):
    """split up the readcounts data by column

    Args:
        df (_type_): dataframe with classified readcounts
        seg_col (str, optional): column where the original segment ids are.

    Returns:
        dict: a dictionary that contains the split up dataframe. E.g. you can access the data for the 'PA' segment via df['PA]
    """
    return dict(tuple(df.groupby(col, sort=False)))

