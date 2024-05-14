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
import subprocess
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

from .variables import *

__all__ = ["get_mem", "check_outdir", "generate_af_axis",
           "flip_snp", "fix_strand", "read_tsv_as_lst",
           "generate_rename_map", "get_genotype",
           "get_imputed_dosage", "recode_indel",
           "encode_hla", "convert_to_str", "file_to_list",
           "combine_df", "find_matching_samples", "append_lst",
           "intersect_dfs", "resolve_common_samples", "fix_v_metrics",
           "extract_info", "encode_genotype", "valid_sample",
           "extract_DS", "extract_format", "drop_cols",
           "convert_to_chip_format", "extract_GT", "extract_GP", "retain_likely_GP", "get_rl_distribution",
           "extract_LDS", "extract_LDS_to_DS", "reorder_cols",
           "convert_to_violin", "combine_violins", "bcftools_get_samples"]

def get_mem() -> None:
    ### Print current memory usage
    # Input: None
    # Output: None
    current_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_memory_usage_mb = current_memory_usage / 1024
    print(f"Current memory usage: {current_memory_usage_mb:.2f} MB")

def check_outdir(outdir: str) -> None:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return None

def read_tsv_as_lst(path): # tsv file should ALWAYS have a single column without header
    if os.path.exists(path):
        return list(pd.read_table(path, header = None, names = ['Code'])['Code'].values)
    else:
        return []

def generate_af_axis(x=MAF_ARY):
    x = [
        str(int(i)) if i == int(i) else str(float(i)).rstrip('0').rstrip('.')
        for i in x * 100
    ]
    y = x[:-1]
    res_ary = list(x[0])
    shift = x[1:]
    combine = res_ary + [i + '-' + j for i, j in zip(y, shift)]
    return combine


def flip_snp(n):
    if n == 'A':
        return 'T'
    elif n == 'T':
        return 'A'
    elif n == 'C':
        return 'G'
    elif n == 'G':
        return 'C'
    else:
        return '.'


def fix_strand(r):
    ref = r['ref']
    alt = r['alt']
    snp = r['SNP']
    if ((r['Strand'] == '-') and (r['ILMN Strand'] == 'TOP')) or ((r['Strand'] == '+') and (r['ILMN Strand'] == 'BOT')):
        r['ref'] = flip_snp(r['ref'])
        r['alt'] = flip_snp(r['alt'])
    return r

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


def generate_rename_map(mini=False, sample_linker=SAMPLE_LINKER_FILE):
    sample_linker = pd.read_table(sample_linker, sep=',')
    if not mini:
        sample_linker = sample_linker[~sample_linker['Sample_Name'].str.
                                      contains('mini')]
    else:
        sample_linker = sample_linker[
            sample_linker['Sample_Name'].str.contains('mini')]
    rename_map = dict(
        zip(sample_linker['Sample_Name'], sample_linker['Chip_Name']))

    return rename_map

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

def intersect_dfs(lst: List[pd.DataFrame], common_cols: List[str] = COMMON_COLS) -> List[pd.DataFrame]:
    common_indices = lst[0].set_index(common_cols).index
    for i in range(1, len(lst)):
        common_indices = common_indices.intersection(lst[i].set_index(common_cols).index)

    for i in range(len(lst)):
        lst[i] = lst[i].set_index(common_cols).loc[common_indices].reset_index()
    return lst

def find_matching_samples(chip_samples, rename_map, lc='chip'):
    if lc == 'chip':
        return chip_samples
    else:
        lc_to_retain = []
        val_to_key = {value: key for key, value in rename_map.items()}
        for s in chip_samples:
            if s in val_to_key:
                lc_to_retain.append(val_to_key[s])
        return lc_to_retain

def resolve_common_samples(df_lst, source_lst, rename_map, mini = False, vcf_cols = VCF_COLS):
    # This utility takes a list of vcfs as input, and intersect them to get a common subset of samples, and return the list. It could resolve different types of sample names, using an inherently generated rename_map.
    # source_lst can only contains lc, chip and hc, indicating where this data is originally from for name resolving.
    sample_lst = []
    for i, df in enumerate(df_lst):
        if source_lst[i] == 'lc':
            if not mini:
                df_lst[i] = df[df.columns[~df.columns.str.contains('mini')]]
            samples = df_lst[i].columns[9:]
            samples_in_gam = [rename_map[s] for s in samples]
            df_lst[i].columns = vcf_cols + samples_in_gam
            sample_lst.append(samples_in_gam)
        else:
            samples = df.columns[9:]
            sample_lst.append(samples)

    sets = [set(arr) for arr in sample_lst]
    common_elements = sets[0]
    for s in sets[1:]:
        common_elements = common_elements.intersection(s)
    common_samples = list(common_elements)
    common_vcf_cols = vcf_cols + common_samples

    for i in range(len(df_lst)):
        df_lst[i] = df_lst[i][common_vcf_cols]
    return df_lst

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

def extract_info(df, info_cols = ['EAF', 'INFO_SCORE'], attribute = 'info', drop_attribute = True):
    for i in info_cols:
        df[i] = df[attribute].str.extract( i + '=([^;]+)' ).astype(float)
    if drop_attribute:
        df = df.drop(columns = [attribute])
    return df

def encode_genotype(r: pd.Series, chip_prefix = CHIP_SAMPLE_PREFIX) -> float:
    ### Encode a row of genotypes to integers.
    samples = valid_sample(r)
    for i in samples:
        if r[i][:3] == '0|0' or r[i][:3] == '0/0':
            r[i] = 0.
        elif r[i][:3] == '1|0' or r[i][:3] == '1/0':
            r[i] = 1.
        elif r[i][:3] == '0|1' or r[i][:3] == '0/1':
            r[i] = 1.
        elif r[i][:3] == '1|1' or r[i][:3] == '1/1':
            r[i] = 2.
        else:
            r[i] = np.nan
    return r

def valid_sample(r):
    return r.index[r.index.str.contains('GM') | r.index.str.contains('GAM') | r.index.str.contains('HV') | r.index.str.contains('kb')]

def extract_GT(r):
    samples = valid_sample(r)
    pos = r['FORMAT'].split(':').index('GT') # This checks which fields is DS, but might want to twist for TOPMed imputation
    r['FORMAT'] = 'GT'
    for i in samples:
        r[i] = r[i].split(':')[pos]
    return r


def extract_DS(r):
    samples = valid_sample(r)
    pos = r['FORMAT'].split(':').index(
        'DS'
    )  # This checks which fields is DS, but might want to twist for TOPMed imputation
    r['FORMAT'] = 'DS'
    for i in samples:
        r[i] = r[i].split(':')[pos]
        if r[i] != '.':
            r[i] = float(r[i])
            if r[i] < 0 or r[i] > 2:
                r[i] = np.nan
        else:
            r[i] = np.nan
    return r

def extract_GP(r):
    samples = valid_sample(r)
    pos = r['FORMAT'].split(':').index('GP') # This checks which fields is DS, but might want to twist for TOPMed imputation
    r['FORMAT'] = 'GP'
    for i in samples:
        r[i] = r[i].split(':')[pos]
    return r

def retain_likely_GP(r):
    samples = valid_sample(r)
    r['FORMAT'] = 'GP'
    for i in samples:
        GPs = r[i].split(',')
        r[i] = max([float(i) for i in GPs])
    return r

def extract_LDS_to_DS(r):
    samples = valid_sample(r)
    pos = r['FORMAT'].split(':').index('LDS')
    for i in samples:
        LDS = r[i].split(':')[pos]
        HD = [float(i) for i in LDS.split('|')]
        r[i] = HD[0] + HD[1]
    r['FORMAT'] = 'DS'
    return r

def extract_LDS(r, convert_to_GP = True):
    samples = valid_sample(r)
    pos = r['FORMAT'].split(':').index('LDS') # This checks which fields is DS, but might want to twist for TOPMed imputation

    fmt = (lambda x: "{:.3f}".format(float(x)))

    for i in samples:
        LDS = r[i].split(':')[pos]
        if convert_to_GP:
            HD = [float(i) for i in LDS.split('|')]
            homref = fmt((1-HD[0])*(1-HD[1]))
            homalt = fmt(HD[0]*HD[1])
            het = fmt(1 - float(homref) - float(homalt))
            r[i] = homref + ',' + het + ',' + homalt
        else:
            r[i] = LDS
    if convert_to_GP:
        r['FORMAT'] = 'GP'
    else:
        r['FORMAT'] = 'LDS'
    return r

def extract_format(df, sample, fmt='format'):
    fields = df[fmt].values[0].split(':')
    try:
        df[fields] = df[sample].str.split(':', expand=True)
        df[df.columns[-1]] = df[df.columns[-1]].astype(float)
        if len(fields) != len(df[sample].values[0].split(':')):
            raise ValueError(
                "Mismatching fields in FORMAT and Imputed results.")
    except ValueError as e:
        print(f"Error: {e}")
    return df.drop(columns=[fmt, sample])

def drop_cols(df, drop_lst = ['id', 'qual', 'filter']):
    return df.drop(columns = drop_lst)

def convert_to_chip_format(r):
    ### Encode a row of imputed results to genotypes
    r['FORMAT'] = 'GT'
    samples = valid_sample(r)
    for i in samples:
        if type(r[i]) != str: # This check if this is nan, but pd.isna() is not working properly
            r[i] = './.'
        else:
            r[i] = r[i][:3]
    return r

def reorder_cols(df):
    cols = list(df.columns)
    cols.insert(2, cols.pop(4))
    return df[cols]

def convert_to_violin(df):
    # df = df.apply(lambda x: x.str.split(',').explode()).reset_index(drop = True)
    gp = df.stack().reset_index(drop=True)
    gt = pd.Series(['0/0'] * len(df.columns) + ['0/1'] * len(df.columns) + ['1/1'] * len(df.columns))
    df = pd.concat([gp,gt], ignore_index = True, axis = 1)
    df.columns = ['GP', 'GT']
    df['GP'] = df['GP'].astype(float)
    return df

def combine_violins(df_lst, labels_lst):
    for df, label in zip(df_lst, labels_lst):
        df['label'] = label
    merged = pd.concat(df_lst)
    return merged

def bcftools_get_samples(vcf):
    command = "bcftools query -l" + " " + vcf
    name = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    return name

def get_rl_distribution(file):
    command = "zcat " + file + " | awk 'NR%4==2 {print length($1)}'"
    rls = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    rls = [int(i) for i in rls]
    return rls
