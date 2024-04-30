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

from .auxiliary import *
from .variables import *

__all__ = ["read_metadata", "read_vcf", "parse_vcf", "multi_parse_vcf", "read_af", "multi_read_af"]

def read_metadata(file, filetype = 'gzip', comment = '#', new_cols = None):
    if filetype == 'gzip':
        with io.TextIOWrapper(gzip.open(file,'r')) as f:
            metadata = [l for l in f if l.startswith(comment)]
    else:
        with open(file, 'r') as f:
            metadata = [l for l in f if l.startswith(comment)]

    if new_cols is not None:
        tmp = metadata[-1].split('\t')[:9] + new_cols
        metadata[-1] = '\t'.join(tmp) + '\n'

    return metadata

def read_vcf(file, sample='call', q=None):
    colname = read_metadata(file)
    header = colname[-1].replace('\n', '').split('\t')
    df = pd.read_csv(file,
                     compression='gzip',
                     comment='#',
                     sep='\t',
                     header=None,
                     names=header,
                     dtype={'#CHROM': str, 'POS': int}).rename(columns={
                         '#CHROM': 'chr',
                         'POS': 'pos',
                         'REF': 'ref',
                         'ALT': 'alt'
                     }).dropna()
    if df.iloc[0, 0][:3] == 'chr':  # Check if the vcf comes with 'chr' prefix
        df = df[df['chr'].isin(['chr' + str(i) for i in range(1, 23)])]
        df['chr'] = df['chr'].str.extract(r'(\d+)').astype(int)
    else:
        df = df[df['chr'].isin([str(i) for i in range(1, 23)])]
        df['chr'] = df['chr'].astype(int)
    if len(df.columns) == 10:
        df.columns = [
            'chr', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info',
            'format', 'call'
        ]
        if sample != 'call':
            df.columns[-1] = sample
    if q is None:
        return df
    else:
        q.put(df)

def parse_vcf(file, sample = 'call', q = None,
              info_cols = ['EAF', 'INFO_SCORE'], attribute = 'info', fmt = 'format', drop_attribute = True, drop_lst = ['id', 'qual', 'filter']):
    df = read_vcf(file)
    df = extract_info(df, info_cols = info_cols, attribute = attribute, drop_attribute = drop_attribute)
    df = extract_format(df, sample, fmt = fmt)
    df = drop_cols(df, drop_lst = drop_lst)
    if q is None:
        return df
    else:
        q.put(df)

def read_af(file, q = None):
    df = pd.read_csv(file, header = None, sep = '\t', names = ['chr', 'pos', 'ref', 'alt', 'MAF'],
                      dtype = {
        'chr': 'string',
        'pos': 'Int64',
        'ref': 'string',
        'alt': 'string',
        'MAF': 'string'
    })
    df = df.dropna()
    df['MAF'] = pd.to_numeric(df['MAF'])
    df['chr'] = df['chr'].str.extract(r'(\d+)').astype(int)
    if q is None:
        return df
    else:
        q.put(df)


def multi_parse_vcf(chromosomes,
                    files,
                    parse=True,
                    sample='call',
                    combine=True,
                    info_cols=['EAF', 'INFO_SCORE'],
                    attribute='info',
                    fmt='format',
                    drop_attribute=True,
                    drop_lst=['id', 'qual', 'filter']):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    processes = []
    for i in range(len(chromosomes)):
        if parse:
            tmp = multiprocessing.Process(target=parse_vcf,
                                          args=(files[i], sample, q, info_cols,
                                                attribute, fmt, drop_attribute,
                                                drop_lst))
        else:
            tmp = multiprocessing.Process(target=read_vcf,
                                          args=(files[i], sample, q))
        tmp.start()
        processes.append(tmp)
    for process in processes:
        process.join()
    res_lst = []
    while not q.empty():
        res_lst.append(q.get())
    if combine:
        return combine_df(res_lst)
    else:
        return res_lst

def multi_read_af(chromosomes, files, combine = True):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    processes = []
    for i in range(len(chromosomes)):
        tmp = multiprocessing.Process(target=read_af, args=(files[i], q))
        tmp.start()
        processes.append(tmp)
    for process in processes:
        process.join()
    res_lst = []
    while not q.empty():
        res_lst.append(q.get())
    if combine:
        return combine_df(res_lst)
    else:
        return res_lst
