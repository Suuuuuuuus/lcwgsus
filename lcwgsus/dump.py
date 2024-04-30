import io
import os
import sys
import csv
import gzip
import time
import random
import secrets
import subprocess
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

###########################################
### DUMP FILE. ONLY FOR ARCHIVE PURPOSE ###
###########################################

from .auxiliary import *
from .process import *
from .read import *
from .plot import *
from .variables import *

def read_r2(panels, samples, indir = '../imputation_accuracy/imputation_accuracy_oneKGafs/', drop=3):
    dfs = []
    for i in panels:
        for j in samples:
            tmp = pd.read_csv(indir+j+"/"+i+"_imputation_accuracy.csv", sep = ',', dtype = {
                'MAF': float,
                'Imputation Accuracy': float,
                'Bin Count': str
            }).iloc[drop:,:]
            tmp['panel'] = i
            tmp['Bin Count'] = j
            tmp.columns = ['AF', 'corr', 'sample', 'panel']
            tmp['AF'] = (100*tmp['AF']).apply(convert_to_str)
            tmp['AF'] = tmp['AF'].shift(1).fillna('0') + '-' + tmp['AF']
            tmp['AF'] = tmp['AF'].astype("category")
            dfs.append(tmp)
    bin_count = pd.read_csv(indir+j+"/"+i+"_imputation_accuracy.csv", sep = ',', dtype = {
                'MAF': float,
                'Imputation Accuracy': float,
                'Bin Count': int
            }).iloc[drop:,:].reset_index(drop = True)[['Bin Count']]
    res = pd.concat(dfs).reset_index(drop = True)
    return res, bin_count


def calculate_imputation_accuracy(df1: pd.DataFrame,
                                  df2: pd.DataFrame,
                                  af: pd.DataFrame,
                                  MAF_ary: np.ndarray = MAF_ARY,
                                  how: str = 'left') -> pd.DataFrame:
    df2 = df2.copy()
    if len(df1.columns) != 5:
        df1 = df1[['chr', 'pos', 'ref', 'alt', 'DS']]
    col1 = df1.columns[-1]
    if type(df2.iloc[0, len(df2.columns) - 1]) == str:
        df2['genotype'] = df2.apply(get_genotype, axis=1)
        df2 = df2.dropna()
        df2['genotype'] = df2['genotype'].astype(float)
        df2 = df2.drop(columns=df2.columns[-2])
        col2 = 'genotype'
    else:
        col2 = df2.columns[-1]

    df = pd.merge(df2, df1, on=['chr', 'pos', 'ref', 'alt'], how=how)
    df = df.fillna(0)
    df = pd.merge(df, af, on=['chr', 'pos', 'ref', 'alt'], how='left')
    df = df.dropna()

    r2 = np.zeros((2, np.size(MAF_ary) - 1))
    for i in range(r2.shape[1]):
        tmp = df[(MAF_ary[i + 1] > df['MAF']) & (df['MAF'] > MAF_ary[i])]
        if tmp.shape[0] == 0:
            r2[0, i] = 0
        else:
            r2[0, i] = np.corrcoef(tmp[col1].values, tmp[col2].values)[0, 1]**2
        r2[1, i] = int(tmp.shape[0])

    r2_df = pd.DataFrame(r2.T,
                         columns=['Imputation Accuracy', 'Bin Count'],
                         index=MAF_ary[1:])
    r2_df.index.name = 'MAF'
    return r2_df

def plot_imputation_accuracy_deprecated(r2, single_sample = True, aggregate = True, save_fig = False, save_name = 'imputation_corr_vs_af.png', outdir = 'graphs/'):
    plt.figure(figsize = (10,6))
    if single_sample:
        if type(r2) == pd.DataFrame:
            plt.plot(r2.index, r2['Imputation Accuracy'], color = 'g')
        else:
            for i in range(len(r2)):
                plt.plot(r2[i].index, r2[i]['Imputation Accuracy'])
        plt.xlabel('gnomAD AF (%)')
        plt.ylabel('$r^2$')
        plt.title(plot_title)
        plt.xscale('log')
    else:
        if aggregate:
            for df in r2:
                panel = df['panel'].values[0]
                plt.plot(np.arange(1, df.shape[0]+1), df['corr'], label = panel)
            plt.xticks(np.arange(1, r2[0].shape[0]+1), r2[0]['AF'], rotation = 45)
            plt.xlabel('Allele frequencies (%)')
            plt.legend()
            plt.text(x = -1.5, y = 1.02, s = 'Aggregated imputation accuracy ($r^2$)')
            plt.grid(alpha = 0.5)
        else:
            sns.set(style="whitegrid")
            sns.stripplot(data=r2, x="corr", y="AF", hue="panel", dodge=True)
            plt.xlabel('Imputation Accuracy')
            plt.ylabel('gnomAD allele frequencies')
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches = "tight", dpi=300)
    return None