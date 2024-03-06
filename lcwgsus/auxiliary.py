import io
import os
import sys
import csv
import gzip
import time
import random
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

__all__ = ["get_mem", "get_genotype", "get_imputed_dosage", "convert_to_str", "file_to_list", "combine_df"]

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