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

__all__ = ["calculate_af", "calculate_ss_cumsum_coverage", "calculate_average_info_score", "calculate_imputation_accuracy"]

def calculate_af(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
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