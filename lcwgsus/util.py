import io
import os
import re
import sys
import csv
import gzip
import glob
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
from .save import *
from .process import *
from .read import *
from .plot import *
from .variables import *
from .calculate import *

__all__ = [
    "merge_metrics", "plot_hist_coverage", "plot_hist_hla_coverage", "plot_qc_metrics", "plot_sex_coverage",
    "visualise_single_variant", "visualise_single_variant_v2",
    "get_badly_imputed_regions", "get_n_variants_vcf", "get_n_variants_impacc", "calculate_bqsr_error_rate"
]

def merge_metrics(cov_all, uncov_all, dup_all, sex_all, hla_all, save=False, outdir=None, save_name=None):
    coverage = pd.read_csv(cov_all, sep = '\t', header = None)
    coverage.columns = ['sample', 'coverage']

    hla_coverage = pd.read_csv(hla_all, sep = '\t', header = None)
    hla_coverage.columns = ['sample', 'hla_coverage']

    uncoverage = pd.read_csv(uncov_all, sep = '\t', header = None)
    uncoverage.columns = ['sample', 'uncoverage']
    
    dup_rate = pd.read_csv(dup_all, sep = '\t', header = None)
    dup_rate.columns = ['sample', 'dup_rate']

    sex = pd.read_csv(sex_all, sep = '\t', header = None)
    sex.columns = ['sample', 'chrX', 'chrY']
    
    metrics = pd.merge(coverage, hla_coverage, on = 'sample')
    metrics = pd.merge(metrics, dup_rate, on = 'sample')
    metrics = pd.merge(metrics, uncoverage, on = 'sample')
    metrics['skew'] = 0
    
    def calculate_skew(r):
        uncov = r['uncoverage']
        cov = r['coverage']
        r['skew'] = uncov/poisson.pmf(0, cov)
        return r
    
    metrics = metrics.apply(calculate_skew, axis = 1)
    metrics = pd.merge(metrics, sex, on = 'sample')
    
    save_tsv(metrics, save, outdir, save_name)
    return metrics

def plot_hist_coverage(metrics, samples = None, save_fig=False, outdir=None, save_name=None):
    metrics = pd.read_csv(metrics, sep = '\t')

    if samples is not None:
        metrics = metrics[metrics['sample'].isin(samples)].reset_index(drop = True)

    plt.figure(figsize=(8, 6), dpi = 300)
    plt.hist(metrics['coverage'], bins = 20, ec = 'black', alpha = 0.75)
    plt.xlabel('De-duplicated genome coverage (X)')
    plt.ylabel('Counts')

    save_figure(save_fig, outdir, save_name)
    return None

def plot_hist_hla_coverage(metrics, samples = None, save_fig=False, outdir=None, save_name=None):
    metrics = pd.read_csv(metrics, sep = '\t')

    if samples is not None:
        metrics = metrics[metrics['sample'].isin(samples)].reset_index(drop = True)

    plt.figure(figsize=(8, 6), dpi = 300)
    plt.hist(metrics['hla_coverage'], bins = 20, ec = 'black', alpha = 0.75)
    plt.xlabel('HLA coverage (X)')
    plt.ylabel('Counts')

    save_figure(save_fig, outdir, save_name)
    return None

def plot_qc_metrics(metrics, samples = None, save_fig=False, outdir=None, save_name=None):
    metrics = pd.read_csv(metrics, sep = '\t')
    if samples is not None:
        metrics = metrics[metrics['sample'].isin(samples)].reset_index(drop = True)

    plt.figure(figsize=(8, 6), dpi = 300)        
    cmap = plt.get_cmap(COLORBAR_CMAP_STR)
    magnitude = 6
    bounds = np.linspace(0, 3, num = magnitude+1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = plt.scatter(metrics['dup_rate']*100, metrics['uncoverage']*100, c = metrics['coverage'],
                     edgecolor='black', cmap=cmap, norm=norm, s = 30, marker = 'o')
    
    plt.xlabel('Duplication rate (%)')
    plt.ylabel('Percent of genome not covered (%)')
    plt.colorbar(im, boundaries=bounds, ticks = bounds, label='De-duplicated genome coverage (X)')
    
    save_figure(save_fig, outdir, save_name)
    return None 
    
# metadata is currently deprecated. We can't assume what data format looks like from other users.
def plot_sex_coverage(metrics, metadata = None, samples = None, save_fig=False, outdir=None, save_name=None):
    metrics = pd.read_csv(metrics, sep = '\t')
    if samples is not None:
        metrics = metrics[metrics['sample'].isin(samples)].reset_index(drop = True)

    c1 = pd.read_csv('data/metadata/33bCN_data.csv')[['id', 'gender']]
    c1.columns = ['sample', 'sex']
    c2 = pd.read_csv('data/metadata/processed/35CN_master.tsv', sep = '\t')[['sample', 'sex']]
    sex = pd.concat([c1, c2])

    df = pd.merge(metrics[['sample', 'chrX', 'chrY']], sex, how = 'inner')

    
    plt.figure(figsize=(8, 6), dpi = 300)
    
    colors = plt.get_cmap(CATEGORY_CMAP_STR).colors[:(len(df['sex'].unique()))]
    colors = [mcolors.to_hex(color) for color in colors]
            
    for i, g in enumerate(df['sex'].unique()):
        tmp = df[df['sex'] == g]
        plt.scatter(tmp['chrX'], tmp['chrY'], color = colors[i], label = g, 
                    ec = 'black', s = 30, marker = 'o')
    plt.xlabel('Coverage on chrX')
    plt.ylabel('Coverage on chrY')
    plt.grid(alpha = 0.5)
    plt.legend()
    
    save_figure(save_fig, outdir, save_name)
    return None

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
                              on='r2',
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
    res_df = res_df.sort_values(by=on, ascending=True)

    if retain_cols != '':
        res_df = res_df[retain_cols]

    if save_file:
        check_outdir(outdir)
        res_df.to_csv(outdir + save_name, sep='\t', header=True, index=False)
    return res_df

def get_n_variants_vcf(vcf):
    if type(vcf) == str:
        command = "zgrep -v ^# " + vcf + " | wc -l"
        count = subprocess.run(command, shell = True, capture_output = True, text = True).stdout.rstrip('\n')
        return int(count)
    elif type(vcf) == list:
        count_sum = 0
        for df in vcf:
            command = "zgrep -v ^# " + df + " | wc -l"
            count = subprocess.run(command, shell = True, capture_output = True, text = True).stdout.rstrip('\n')
            count_sum += int(count)
        return count_sum
    else:
        print('Invalid input types. It has to be a str of a vcf file or a list of vcf files.')
        return -9

def get_n_variants_impacc(impacc, colname):
    if type(impacc) == str:
        df = pd.read_csv(impacc, sep = '\t')
        count = df[colname].sum()
        return count
    elif type(impacc) == list:
        count_sum = 0
        for i in impacc:
            df = pd.read_csv(i, sep = '\t')
            count = df[colname].sum()
            count_sum += count
        return count_sum
    else:
        print('Invalid input types. It has to be a str of an impacc file or a list of impacc files.')
        return -9

def calculate_bqsr_error_rate(indir, subset_samples = None, positions = ['-1', '-151', '1', '151']):
    pattern = indir + '*.report'
    target_line = '#:GATKTable:RecalTable2:'
    dfs = []
    paths = glob.glob(pattern)

    for file_path in paths:
        file_name = file_path.split('/')[-1].split('.')[0]

        start_reading = False
        data_lines = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if start_reading:
                    data_lines.append(line)
                if line == target_line:
                    start_reading = True

        data = "\n".join(data_lines)
        data = re.sub(r'\n', '+++', data)
        data = re.sub(r'\s+', ' ', data)
        data = re.sub(r'\+\+\+', r'\n', data)
        data = io.StringIO(data)

        df = pd.read_csv(data, sep = ' ').drop(columns = ['QualityScore', 'ReadGroup', 'CovariateName', 'EventType', 'EmpiricalQuality'])
        df['sample'] = file_name
        dfs.append(df)

    subsets = []
    if subset_samples is not None:
        for df in dfs:
            if df.loc[0, 'sample'] in subset_samples:
                subsets.append(df)
        dfs = subsets
    error_ary = []
    for i, df in enumerate(dfs):
        df = df[df['CovariateValue'].isin(positions)]
        error = df.groupby('CovariateValue').sum().reset_index()
        error['prob'] = error['Errors']/error['Observations']
        error_ary.append(error)
    errors = pd.concat(error_ary).reset_index(drop = True)
    mean_error = {}
    for i in positions:
        tmp = errors[errors['CovariateValue'] == i]
        mean_error[i] = tmp['prob'].mean()
    return mean_error
