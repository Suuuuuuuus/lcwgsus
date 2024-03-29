import io
import os
import sys
import csv
import gzip
import time
import random
import json
import secrets
import resource
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
import scipy
from typing import Union, Tuple, List
from scipy.stats import poisson
from scipy.stats import chi2
from scipy.stats import friedmanchisquare
from scipy.stats import studentized_range
pd.options.mode.chained_assignment = None

__all__ = ["get_mem", "get_genotype", "get_imputed_dosage", "recode_indel", "encode_hla", "convert_to_str", "file_to_list", "combine_df", "find_matching_samples", "append_lst", "intersect_dfs", "fix_v_metrics"]

def get_mem() -> None:
    ### Print current memory usage
    # Input: None
    # Output: None
    current_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_memory_usage_mb = current_memory_usage / 1024
    print(f"Current memory usage: {current_memory_usage_mb:.2f} MB")

def get_genotype(df: pd.DataFrame, colname: str = 'call') -> float:
    ### Encode a column of genotypes to integers.
    # Input: df with cols "ref", "alt", and <colname>.
    # Output: a dataframe column stores int-value genotypes.
    # NB: only biallelic SNPs are retained. If a variant is multi-allelic or is a SV, its genotype will be `np.nan`.
    ref = df['ref']
    alt = df['alt']
    s = df[colname]
    if len(alt) != 1 or len(ref) != 1:
        return np.NaN
    if s == '0|0' or s == '0/0':
        return 0.
    elif s == '1|0' or s == '1/0':
        return 1.
    elif s == '0|1' or s == '0/1':
        return 1.
    elif s == '1|1' or s == '1/1':
        return 2.
    else:
        return np.nan

def get_imputed_dosage(df: pd.DataFrame, colname: str = 'call') -> float:
    ### Extract imputed dosage from QUILT imputation fields, which should come as a form of `GT:GP:DS`.
    # Input: df with cols "ref", "alt", and <colname>.
    # Output: a dataframe column stores diploid dosages.
    # NB: only biallelic SNPs are retained. If a variant is multi-allelic or is a SV, its genotype will be `np.nan`.
    ref = df['ref']
    alt = df['alt']
    s = df[colname]
    if alt == '.' or len(alt) > 1 or len(ref) > 1 :
        return np.nan
    else:
        return s.split(':')[2]
    
def recode_indel(r: pd.Series, info: str = 'INFO') -> pd.Series:
    ### Read from flanking sequence and recode ref/alt to the real nucleotide rather than '-'
    # Input: one row of df
    # Output: recoded row
    flank = r[info].split('FLANK=')[1]
    nucleotide = flank.split('[')[0][-1]
    
    if r['ref'] == '-':
        r['ref'] = nucleotide
        r['alt'] = nucleotide + r['alt']
    elif r['alt'] == '-':
        r['ref'] = nucleotide + r['ref']
        r['alt'] = nucleotide
        r['pos'] = r['pos'] - 1
    else:
        r = r
    return r

def encode_hla(s: str) -> int:
    ### Convert HLA genotypes to diploid dosage
    # Input: HLA df sample columns.
    # Output: a dataframe column stores diploid dosages.
    parts = s.split(':')[0].split('|')
    return int(parts[0]) + int(parts[1])

def convert_to_str(x: Union[float, int]) -> str:
    ### Convert floats and integers to strings.
    # Input: a number.
    # Output: the number in type of str.
    if x == int(x):
        return str(int(x))
    else:
        return str(x)

def file_to_list(df: pd.DataFrame) -> List[pd.DataFrame]:
    ### Break a single df into a list of small dfs to apply multiprocessing.
    # Input: a df with col "chr".
    # Output: a list of dfs.
    lst = []
    for i in df[df.columns[0]].unique():
        lst.append(df[df[df.columns[0]] == i])
    return lst

def combine_df(lst: List[pd.DataFrame]) -> pd.DataFrame:
    ### Bring a list of dfs into a big df.
    # Input: a list of dfs.
    # Output: a single df.
    # NB: By default, the df is sorted according to its first two columns - "chr" and "pos"
    df = lst[0]
    for i in range(1, len(lst)):
        df = pd.concat([df, lst[i]])
    return df.sort_values(by = df.columns[:2].to_list()).reset_index(drop = True)

def intersect_dfs(lst: List[pd.DataFrame], common_cols: List[str] = ['chr', 'pos', 'ref', 'alt']) -> List[pd.DataFrame]:
    common_indices = lst[0].set_index(common_cols).index
    for i in range(1, len(lst)):
        common_indices = common_indices.intersection(lst[i].set_index(common_cols).index)
    
    for i in range(len(lst)):
        lst[i] = lst[i].set_index(common_cols).loc[common_indices].reset_index()
    return lst

def find_matching_samples(lc_samples, chip_samples, rename_map):
    lc_to_retain = []
    for key, value in rename_map.items():
        if value in chip_samples:
            lc_to_retain.append(key)
    return lc_to_retain

def fix_v_metrics(res_ary, metrics):
    for i in range(len(metrics)):
        res_ary.append(metrics[i])
        if i % 2 == 1:
            res_ary.append(metrics[i])
    return res_ary

def append_lst(tmp_lst, full_lst):
    for i, l in zip(tmp_lst, full_lst):
        l.append(i)
    return full_lst