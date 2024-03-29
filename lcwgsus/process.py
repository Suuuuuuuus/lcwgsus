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
from .save import *

__all__ = ["aggregate_r2", "extract_info", "encode_genotype", "extract_DS", "extract_format", "extract_hla_type", "convert_indel" , "drop_cols", "subtract_bed_by_chr", "multi_subtract_bed", "filter_afs", "imputation_calculation_preprocess"]

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
def encode_genotype(r: pd.Series, chip_prefix = 'GAM') -> float:
    ### Encode a row of genotypes to integers.
    samples = r.index[r.index.str.contains(chip_prefix)]
    for i in samples:
        if r[i] == '0|0' or r[i]  == '0/0':
            r[i] = 0.
        elif r[i]  == '1|0' or r[i]  == '1/0':
            r[i] = 1.
        elif r[i]  == '0|1' or r[i]  == '0/1':
            r[i] = 1.
        elif r[i]  == '1|1' or r[i]  == '1/1':
            r[i] = 2.
        else:
            r[i] = np.nan
    return r
def extract_DS(r, lc_prefix = 'GM'):
    samples = r.index[r.index.str.contains(lc_prefix)]
    for i in samples:
        r[i] = float(r[i].split(':')[-1]) # Now this assumes DS has to be the last INFO field, but this might not be true
        if r[i] < 0 or r[i] > 2:
            r[i] = np.nan
    return r
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
# Currently the saved version is gzip rather than bgzip
def convert_indel(vcf, save = False, prefix = 'chr', outdir = 'test.vcf.gz'):
    metadata = read_metadata(vcf)
    df = read_vcf(vcf)
    indels = df[(df['ref'] == '-') | (df['alt'] == '-')]
    indels = indels.apply(recode_indel, axis = 1)
    snps = df[(df['ref'] != '-') & (df['alt'] != '-')]
    df = pd.concat([snps, indels]).sort_values(by = ['chr', 'pos'], ascending = True)
    if save:
        save_vcf(df, metadata, prefix, save_name = outdir)
    return df

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
            
def imputation_calculation_preprocess(truth_vcf, imp_vcf, af_txt,
                                      sample_linker = 'data/metadata/sample_linker.csv', 
                                      MAF_ary = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95, 1]),
                                     chromosome = None, mini = False,
                                     common_cols = ['chr', 'pos', 'ref', 'alt'],
                                     lc_sample_prefix = 'GM', chip_sample_prefix = 'GAM', seq_sample_prefix = 'IDT'):
    
# Truth_vcf should have GT rather than DS in its FORMAT field, whereas imp_vcf has to have DS
    af = read_af(af_txt)
    lc = read_vcf(imp_vcf).sort_values(by = ['chr', 'pos'])
    chip = read_vcf(truth_vcf).sort_values(by = ['chr', 'pos'])

    sample_linker = pd.read_table(sample_linker, sep = ',')
    if not mini:
        sample_linker = sample_linker[~sample_linker['Sample_Name'].str.contains('mini')]
        lc_samples = lc.columns[lc.columns.str.contains(lc_sample_prefix) & ~lc.columns.str.contains('mini')]
    else:
        sample_linker = sample_linker[sample_linker['Sample_Name'].str.contains('mini')]
        lc_samples = lc.columns[lc.columns.str.contains(lc_sample_prefix) & lc.columns.str.contains('mini')]
    rename_map = dict(zip(sample_linker['Sample_Name'], sample_linker['Chip_Name']))
    
    if chromosome is not None:
        lc = lc[lc['chr'] == int(chromosome)]
        chip = chip[chip['chr'] == int(chromosome)]
        af = af[af['chr'] == int(chromosome)]

    lc = lc.drop(columns = ['ID', 'QUAL', 'FILTER', 'INFO', 'FORMAT'])
    chip = chip.drop(columns = ['ID', 'QUAL', 'FILTER', 'INFO', 'FORMAT'])

    res = intersect_dfs([chip, lc, af])
    chip = res[0]
    lc = res[1]
    af = res[2]

    chip_samples = chip.columns[chip.columns.str.contains(chip_sample_prefix)]
    lc_to_retain = find_matching_samples(lc_samples, chip_samples, rename_map)
    lc = lc[common_cols + lc_to_retain]

    lc = lc.apply(extract_DS, axis = 1)
    chip = chip.apply(encode_genotype, axis = 1)

    chip_order = []
    for i in lc_to_retain:
        chip_order.append(rename_map[i])
    chip = chip[common_cols + chip_order]
    return chip, lc, af