import io
import os
import sys
import csv
import gzip
import time
import random
import secrets
import json
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
from .read import *

__all__ = ["aggregate_r2", "extract_info", "extract_format", "extract_hla_type", "drop_cols", "subtract_bed_by_chr", "multi_subtract_bed", "filter_afs"]

def aggregate_r2(df):
    tmp = df.copy().groupby(['AF', 'panel'])['corr'].mean().reset_index()
    res_ary = []
    for i in tmp['panel'].unique():
        imp_res = tmp[tmp['panel'] == i]
        imp_res['sort'] = imp_res['AF'].apply(lambda x: x.split('-')[0]).astype(float)
        imp_res = imp_res.sort_values(by = 'sort', ascending = True).drop(columns = 'sort')
        res_ary.append(imp_res.reset_index(drop = True))
    return res_ary
def extract_info(df, info_cols = ['EAF', 'INFO_SCORE'], attribute = 'info', drop_attribute = True):
    for i in info_cols:
        df[i] = df[attribute].str.extract( i + '=([^;]+)' ).astype(float)
    if drop_attribute:
        df = df.drop(columns = [attribute])
    return df
def extract_format(df, sample, fmt = 'format'):
    fields = df[fmt].values[0].split(':')
    try:
        df[fields] = df[sample].str.split(':', expand=True)
        df[df.columns[-1]] = df[df.columns[-1]].astype(float)
        if len(fields) != len(df[sample].values[0].split(':')):
            raise ValueError("Mismatching fields in FORMAT and Imputed results.")
    except ValueError as e:
        print(f"Error: {e}")
    return df.drop(columns = [fmt, sample])
def drop_cols(df, drop_lst = ['id', 'qual', 'filter']):
    return df.drop(columns = drop_lst)

def subtract_bed_by_chr(cov, region, q = None):
    i = 0
    tmp = 0
    for j in range(region.shape[0]):
        chr, start, end = region.iloc[j,:]
        while start > cov.iloc[i,2]:
            i += 1
        if start < cov.iloc[i,1]:
            cov.iloc[i-1, 2] = start
            if end < cov.iloc[i,2]:
                cov.iloc[i,1] = end
            elif end == cov.iloc[i,2]:
                cov.iloc[i,3] = -9
                i += 1
            else:
                tmp = i
                while end > cov.iloc[tmp,2]:
                    tmp += 1
                if end < cov.iloc[tmp, 2]:
                    cov.iloc[tmp, 1] = end
                    cov.iloc[i:tmp, 3] = -9
                    i = tmp
                else:
                    cov.iloc[i:tmp+1, 3] = -9
                    i = tmp
        elif start == cov.iloc[i,1]:
            if end < cov.iloc[i,2]:
                cov.iloc[i,1] = end
            elif end == cov.iloc[i,2]:
                cov.iloc[i, 3] = -9
            else:
                tmp = i
                while end > cov.iloc[tmp,2]:
                    tmp += 1
                if end < cov.iloc[tmp, 2]:
                    cov.iloc[tmp, 1] = end
                    cov.iloc[i:tmp+1, 3] = -9
                    i = tmp
                else:
                    cov.iloc[i:tmp, 3] = -9
                    i = tmp
        else:
            idx = cov.index.max() + 1
            cov.loc[idx] = {'chr': chr, 'start': cov.iloc[i,1], 'end': start, 'cov': cov.iloc[i,3]}
            if end < cov.iloc[i, 2]:
                cov.iloc[i, 1] = end
            elif end == cov.iloc[i, 2]:
                cov.iloc[i, 3] = -9
            else:
                tmp = i
                while end > cov.iloc[tmp,2]:
                    tmp += 1
                if end < cov.iloc[tmp, 2]:
                    cov.iloc[tmp, 1] = end
                    cov.iloc[i:tmp, 3] = -9
                    i = tmp
                else:
                    cov.iloc[i:tmp+1, 3] = -9
                    i = tmp
    res = cov[cov['cov'] >= 0].sort_values(by = cov.columns[:2].to_list()).reset_index(drop = True)
    if q is None:
        return res
    else:
        q.put(res)

def multi_subtract_bed(chromosomes, covs, regions, combine = True):
    manager = multiprocessing.Manager()
    q = manager.Queue()
    processes = []
    for i in range(len(chromosomes)):
        tmp = multiprocessing.Process(target=subtract_bed_by_chr, args=(covs[i], regions[i], q))
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

def filter_afs(df1, df2, diff=0.2, z_score=None):
    # df1 is the main vcf in which afs are to be filtered out
    # df2 is the ref panel afs
    # Either filter by z-score (suggested 2 sds so 1.96 or diff=0.2)
    res = pd.merge(df1, df2, on=['chr', 'pos', 'ref', 'alt'])
    if z_score is not None:
        res = res[(res['prop_y'] != 0) & (res['prop_y'] != 1)]
        res['z'] = (res['prop_x'] - res['prop_y']) / np.sqrt(
            res['prop_y'] * (1 - res['prop_y']))
        res = res[abs(res['z']) <= z_score]
        return res.drop(columns=['prop_x', 'prop_y', 'z'])
    else:
        res = res[abs(res['prop_x'] - res['prop_y']) < diff]
        return res.drop(columns=['prop_y']).rename(columns={'prop_x': 'prop'})

def extract_hla_type(input_vcf, csv_path, json_path):
    vcf = read_vcf(input_vcf)
    samples = list(vcf.columns[9:])

    for i in samples:
        vcf[i] = vcf[i].apply(encode_hla)

    types = vcf['ID'].str.split('*').str.get(0).unique()
    types.sort()
    hla = pd.DataFrame({'Name': samples})
    for i in types:
        hla[i + '_1'] = 0
        hla[i + '_2'] = 0
    hla.set_index('Name', inplace=True)

    num_type = len(types) * 2
    hla_abnormal = {}
    for sample in samples:
        hla_type = []
        for gene in types:
            tmp_vcf = vcf[vcf['ID'].str.contains(gene)].reset_index().drop(
                columns='index')
            hla_subtype = []
            for i in range(tmp_vcf.shape[0]):
                if tmp_vcf.loc[i, sample] == 1:
                    hla_subtype.append(tmp_vcf.iloc[i, 2])
                elif tmp_vcf.loc[i, sample] == 2:
                    hla_subtype.append(tmp_vcf.iloc[i, 2])
                    hla_subtype.append(tmp_vcf.iloc[i, 2])
                else:
                    pass
            if len(hla_subtype) < 2:
                hla_subtype = hla_subtype + ['N/A'] * (2 - len(hla_subtype))
            hla_type = hla_type + hla_subtype
        if len(hla_type) == num_type:
            hla.loc[sample, :] = hla_type
        else:
            hla_abnormal[sample] = hla_type

    hla.to_csv(csv_path, header = True, index = True)
    if hla_abnormal != {}:
        with open(json_path, "w") as json_file:
            json.dump(hla_abnormal, json_file)