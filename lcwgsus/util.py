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

from .auxiliary import *
from .process import *
from .read import *
from .plot import *
from .variables import *

__all__ = ["visualise_single_variant", "visualise_single_variant_v2", "get_badly_imputed_regions"]

def visualise_single_variant(c, pos, vcf_lst, source_lst, labels_lst, vcf_cols = VCF_COLS, mini = False, save_fig = False, outdir = None, save_name = None):
    site = 'chr' + str(c) + ':' + str(pos) + '-' + str(pos)
    df_ary = []
    n = len(vcf_lst)
    rename_map = generate_rename_map(mini = mini)

    for i in vcf_lst:
        command = "tabix" + " " + i + " " + site + " | tail -n 1"
        data = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\t')
        command = "bcftools query -l" + " " + i
        name = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
        col = vcf_cols + name
        df = pd.DataFrame([data], columns=col)
        df_ary.append(df)

    df_ary = resolve_common_samples(df_ary, source_lst, rename_map)

    for i in range(n):
        if 'GP' in df_ary[i].loc[0, 'FORMAT']:
            df_ary[i] = df_ary[i].apply(extract_GP, axis=1)
        else:
            df_ary[i] = df_ary[i].apply(extract_LDS, axis=1)
        df_ary[i] = df_ary[i].drop(columns = vcf_cols)
        df_ary[i] = convert_to_violin(df_ary[i])

    res = combine_violins(df_ary, labels_lst)

    plot_violin(res, x = 'GT', y = 'GP', hue = 'label', title = site, save_fig = save_fig, outdir = outdir, save_name = save_name)
    return None

def visualise_single_variant_v2(c, pos, vcf_lst, source_lst, labels_lst, vcf_cols = VCF_COLS, mini = False, save_fig = False, outdir = None, save_name = None):
    site = 'chr' + str(c) + ':' + str(pos) + '-' + str(pos)
    df_ary = []
    n = len(vcf_lst)
    rename_map = generate_rename_map(mini = mini)

    for i in vcf_lst:
        command = "tabix" + " " + i + " " + site + " | head -n 1"
        data = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\t')
        command = "bcftools query -l" + " " + i
        name = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
        col = vcf_cols + name
        df = pd.DataFrame([data], columns=col)
        df_ary.append(df)

    df_ary = resolve_common_samples(df_ary, source_lst, rename_map)

    df_ary[0] = df_ary[0].apply(extract_GT, axis = 1)
    df_ary[0] = df_ary[0].drop(columns = vcf_cols)
    df_ary[0] = df_ary[0].T.rename(columns = {0: 'GT'})

    res_ary = []

    for i in range(1, n):
        if 'DS' in df_ary[i].loc[0, 'FORMAT']:
            df_ary[i] = df_ary[i].apply(extract_DS, axis=1)
        else:
            df_ary[i] = df_ary[i].apply(extract_LDS_to_DS, axis=1)

        df_ary[i] = df_ary[i].drop(columns = vcf_cols)
        df_ary[i] = df_ary[i].T.rename(columns = {0: 'DS'})
        res = pd.merge(df_ary[i].reset_index(), df_ary[0].reset_index(), on = 'index').drop(columns = ['index'])
        res['label'] = labels_lst[i]
        res_ary.append(res)

    res = pd.concat(res_ary)

    plot_violin(res, x = 'GT', y = 'DS', hue = 'label', title = site, save_fig = save_fig, outdir = outdir, save_name = save_name)
    return None


def get_badly_imputed_regions(indir,
                              on = 'r2',
                              threshold=0.5,
                              placeholder=-9,
                              chromosomes=CHROMOSOMES_ALL,
                              retain_cols='',
                              save_file=False,
                              outdir='',
                              save_name=''):
    hs = [indir + "chr" + c + ".h.tsv" for c in chromosomes]
    hs_lst = [pd.read_csv(i, sep='\t') for i in hs]
    merged = pd.concat(hs_lst).reset_index(drop=True)
    merged = merged[merged[on] != placeholder]
    res_df = merged[merged[on] < threshold]
    res_df = res_df.sort_values(by = on, ascending = True)

    if retain_cols != '':
        res_df = res_df[retain_cols]

    if save_file:
        check_outdir(outdir)
        res_df.to_csv(outdir + save_name, sep='\t', header=True, index=False)
    return res_df
