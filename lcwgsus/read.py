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

__all__ = ["read_metadata", "read_vcf", "parse_vcf", "multi_parse_vcf", "read_af", "multi_read_af", "read_hla_direct_sequencing"]

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

def read_hla_direct_sequencing(file = HLA_DIRECT_SEQUENCING_FILE, retain = 'all'):
    hla = pd.read_csv(file)
    hla = hla[['SampleID', 'Locus', 'Included Alleles', 'G code']]
    hla = hla[hla['Locus'].isin(HLA_GENES)].reset_index(drop = True)
    hla['One field1'] = ''
    hla['Two field1'] = ''

    hla = hla.apply(resolve_ambiguous_hla_type, axis = 1)
    hla = hla.drop(columns = ['Included Alleles', 'G code'])

    for s in hla['SampleID'].unique():
        tmps = hla[hla['SampleID'] == s]
        for l in HLA_GENES:
            tmpl = tmps[tmps['Locus'] == l]
            repeat = 2 - tmpl.shape[0]
            if repeat == 2:
                hla.loc[len(hla)] = [s, l, '-9', '-9']
                hla.loc[len(hla)] = [s, l, '-9', '-9']
            if repeat == 1:
                hla.loc[len(hla)] = [s, l, tmpl.iloc[0,2], tmpl.iloc[0, 3]]
    hla = hla.sort_values(by = ['SampleID', 'Locus']).reset_index(drop = True)
    hla = pd.concat([hla.iloc[::2].reset_index(drop=True), hla.iloc[1::2, 2:].reset_index(drop=True)], axis=1)
    hla.columns = ['SampleID', 'Locus', 'One field1', 'Two field1', 'One field2', 'Two field2']

    if retain == 'fv':
        fv_samples = read_tsv_as_lst('data/metadata/sample_tsvs/fv_gam_names.tsv')
        hla = hla[hla['SampleID'].isin(fv_samples)].reset_index(drop = True)
    elif retain == 'mini':
        mini_samples = read_tsv_as_lst('data/metadata/sample_tsvs/mini_gam_names.tsv')
        hla = hla[hla['SampleID'].isin(mini_samples)].reset_index(drop = True)
    else:
        pass
    return hla