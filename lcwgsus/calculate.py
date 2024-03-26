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

__all__ = ["calculate_af", "calculate_ss_cumsum_coverage", "calculate_average_info_score", "calculate_imputation_accuracy", "calculate_corrcoef", "calculate_concordance", "calculate_imputation_accuracy_metrics", "average_metrics", "calculate_h_imputation_accuracy", "generate_h_impacc"]

def calculate_af(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame: # WARNING:This utility might be erroneous!
    # df should have columns chr, pos, ref, alt and genotypes
    df['prop'] = 0
    for i in range(len(df.index)):
        count = 0
        for j in range(4, len(df.columns) - 2):
            count += df.iloc[i, j].split('/').count('1')
        df.iloc[i, -1] = count/(2*(len(df.columns) - 5))
    if drop:
        return df[['chr', 'pos', 'ref', 'alt', 'prop']]
    else:
        return df

def calculate_corrcoef(r1, r2, square = True):
    if r1.size < 2 or len(np.unique(r1)) == 1 or len(np.unique(r2)) == 1:
        return -9
    else:
        if square:
            return np.corrcoef(r1, r2)[0,1]**2
        else:
            return np.corrcoef(r1, r2)[0,1]
    
def calculate_concordance(r1, r2):
    if r1.size == 0:
        return -9
    else:
        return np.sum(np.abs(r1 - r2) < 0.5)/r1.size

def calculate_ss_cumsum_coverage(df: pd.DataFrame, num_coverage: int = 5) -> np.ndarray:
    df['bases'] = df['end'] - df['start']
    df = df.groupby(['cov']).bases.sum().reset_index()
    df['prop bases'] = df['bases']/df.bases.sum()
    df['cum prop'] = np.cumsum(df['prop bases'].to_numpy())
    df['prop genome at least covered'] = (1-df['cum prop'].shift(1))
    df = df.dropna()
    coverage_ary = df['prop genome at least covered'].values[:num_coverage]
    return coverage_ary
def calculate_average_info_score(chromosomes: Union[List[int], List[str]], vcf: pd.DataFrame, af: pd.DataFrame, chip_df: pd.DataFrame, 
                   MAF_ary: np.ndarray = np.array([0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95, 1])) -> pd.DataFrame:
    # Input vcf is a df that contains chr, pos, ref, alt, info
    # af and chip_df are used to filtered out variants by position
    # returns average INFO in each MAF bin
    info = pd.merge(vcf, chip_df, on = ['chr', 'pos', 'ref', 'alt'], how = 'inner')
    info = pd.merge(info, af, on = ['chr', 'pos', 'ref', 'alt'], how = 'inner')
    info = info[['chr', 'pos', 'ref', 'alt', 'INFO_SCORE', 'MAF']]
    info['classes'] = np.digitize(info['MAF'], MAF_ary)
    info['classes'] = info['classes'].apply(lambda x: len(MAF_ary)-1 if x == len(MAF_ary) else x)
    score = info.copy().groupby(['classes'])['INFO_SCORE'].mean().reset_index()
    return score

# This method is currently deprecated
def calculate_imputation_accuracy(df1: pd.DataFrame, df2: pd.DataFrame, af: pd.DataFrame,
                                  MAF_ary: np.ndarray = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95, 1]),
                                 how: str = 'left') -> pd.DataFrame:
    df2 = df2.copy()
    if len(df1.columns) != 5:
        df1 = df1[['chr', 'pos', 'ref', 'alt', 'DS']]
    col1 = df1.columns[-1]
    if type(df2.iloc[0, len(df2.columns)-1]) == str:
        df2['genotype'] = df2.apply(get_genotype, axis = 1)
        df2 = df2.dropna()
        df2['genotype'] = df2['genotype'].astype(float)
        df2 = df2.drop(columns = df2.columns[-2])
        col2 = 'genotype'
    else:
        col2 = df2.columns[-1]

    df = pd.merge(df2, df1, on=['chr', 'pos', 'ref', 'alt'], how=how)
    df = df.fillna(0)
    df = pd.merge(df, af, on=['chr', 'pos', 'ref', 'alt'], how='left')
    df = df.dropna()

    r2 = np.zeros((2, np.size(MAF_ary) - 1))
    for i in range(r2.shape[1]):
        tmp = df[(MAF_ary[i+1] > df['MAF']) & (df['MAF'] > MAF_ary[i])]
        if tmp.shape[0] == 0:
            r2[0,i] = 0
        else:
            r2[0, i] = np.corrcoef(tmp[col1].values, tmp[col2].values)[0,1]**2
        r2[1, i] = int(tmp.shape[0])

    r2_df = pd.DataFrame(r2.T, columns = ['Imputation Accuracy','Bin Count'], index = MAF_ary[1:])
    r2_df.index.name = 'MAF'
    return r2_df
def calculate_imputation_accuracy_metrics(r1, r2):
    valid_indices = np.logical_not(np.logical_or(pd.isnull(r1), pd.isnull(r2)))
    r1 = r1[valid_indices]
    r2 = r2[valid_indices]
    
    n_rf_all = r1.size
    
    rf_all = calculate_corrcoef(r1, r2)
    if pd.isna(rf_all) is True:
        rf_all = -9
    
    r1_homref = r1[r1 == 0]
    r2_homref = r2[r1 == 0]
    n_homref = r1_homref.size
    r1_het = r1[r1 == 1]
    r2_het = r2[r1 == 1]
    n_het = r1_het.size
    r1_homalt = r1[r1 == 2]
    r2_homalt = r2[r1 == 2]
    n_homalt = r1_homalt.size
    
    n_sample = n_het + n_homalt
    n_nrc = np.sum(np.abs(r1_het - r2_het) < 0.5) + np.sum(np.abs(r1_homalt - r2_homalt) < 0.5)
    if n_sample == 0:
        nrc = -9
    else:
        nrc = n_nrc/n_sample
    
    rf_homref = calculate_concordance(r1_homref, r2_homref)
    rf_het = calculate_concordance(r1_het, r2_het)
    rf_homalt = calculate_concordance(r1_homalt, r2_homalt)
    
    return [nrc, n_sample, rf_all, n_rf_all, rf_homref, n_homref, rf_het, n_het, rf_homalt, n_homalt]

def calculate_h_imputation_accuracy(chip, lc, af,
                                    impacc_colnames = ['NRC', 'NRC_BC', 
                                                       'r2', 'r2_BC', 
                                                       'ccd_homref', 'ccd_homref_BC', 
                                                       'ccd_het', 'ccd_het_BC', 
                                                       'ccd_homalt', 'ccd_homalt_BC'],
                                   save_file = False, outdir = None, save_name = None):
    stack_lst = [[] for _ in range(len(impacc_colnames))]

    for _, (_, r1), (_, r2) in zip(range(len(chip)), chip.iterrows(), lc.iterrows()):
        r1 = np.array(r1.values[4:]).astype(float) # Assume that quilt and chip has all cols removed except for commom_cols, modify the number 4 if necessary
        r2 = np.array(r2.values[4:]).astype(float)
        metrics = calculate_imputation_accuracy_metrics(r1, r2)
        stack_lst = append_lst(metrics, stack_lst)

    for name, col in zip(impacc_colnames, stack_lst):
        af[name] = col
    
    if save_file:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        af.to_csv(outdir + save_name, sep = '\t', index = False, header = True)
    return af

def average_metrics(df, cols = ['NRC', 'r2', 'ccd_homref', 'ccd_het', 'ccd_homalt'], placeholder = -9):
    res_ary = []
    res_ary.append(int(df.shape[0]))
    
    for c in cols:
        c_count = c + '_BC'
        tmp = df[df[c] != placeholder][[c, c_count]]
        num = tmp[c_count].sum()
        n_variants = tmp.shape[0]
        if n_variants == 0:
            res_ary.append(placeholder)
            res_ary.append(n_variants)
        else:
            avg = (tmp[c]*tmp[c_count]).sum()/num
            res_ary.append(avg)
            res_ary.append(n_variants)
        res_ary.append(num)
    return res_ary

def generate_h_impacc(df, MAF_ary = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95, 1]),
                      colnames = ['n_variants', 'NRC', 'NRC_BC', 'NRC_AC', 
                                  'r2', 'r2_BC', 'r2_AC', 
                                  'ccd_homref', 'ccd_BC', 'ccd_homref_AC', 
                                  'ccd_het', 'ccd_het_BC', 'ccd_het_AC',
                                  'ccd_homalt', 'ccd_homalt_BC', 'ccd_homalt_AC'],
                  save_impacc = False, outdir = None, save_name = None):
    impacc = pd.DataFrame({'AF': MAF_ary})
    
    stack_lst = [[] for _ in range(len(colnames))]

    tmp = df[df['MAF'] == 0]
    if tmp.shape[0] == 0:
        metrics = [0] + [-9, 0, 0]*5
    else:
        metrics = average_metrics(tmp)
    stack_lst = append_lst(metrics, stack_lst)
    
    for i in range(np.size(MAF_ary) - 1):
        tmp = df[(MAF_ary[i+1] >= df['MAF']) & (df['MAF'] > MAF_ary[i])]
        metrics = average_metrics(tmp)
        stack_lst = append_lst(metrics, stack_lst)
    
    for name, col in zip(colnames, stack_lst):
        impacc[name] = col
    
    if save_impacc:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        impacc.to_csv(outdir + save_name, sep = '\t', index = False, header = True)
    return impacc