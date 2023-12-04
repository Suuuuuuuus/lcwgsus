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

__all__ = ["plot_afs", "plot_imputation_accuracy", "plot_sequencing_skew", "plot_info_vs_af", "plot_r2_vs_info"]

def plot_afs(df1: pd.DataFrame, df2: pd.DataFrame, save_fig: bool = False, outdir: str = 'graphs/', save_name: str = 'af_vs_af.png') -> float:
    # df1 is the chip df with cols chr, pos, ref, alt and prop
    # df2 is the other df with the same cols
    df = pd.merge(df1, df2, on = ['chr', 'pos', 'ref', 'alt'], how = 'inner')
    plt.scatter(df['prop_x']*100, df['prop_y']*100)
    plt.xlabel('ChIP MAF (%)')
    plt.ylabel('GGVP AF (%)')
    plt.title('Check AFs')
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches = "tight", dpi=300)
    return np.corrcoef(df['prop_x'], df['prop_y'])[0,1]
def plot_imputation_accuracy(r2, single_sample = True, aggregate = True, save_fig = False, save_name = 'imputation_corr_vs_af.png', outdir = 'graphs/'):
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
def plot_sequencing_skew(arys, avg_coverage, n_se = 1.96, code = None, num_coverage=5, save_fig = False, save_name = 'prop_genome_at_least_coverage.png', outdir = 'graphs/'):
    poisson_expectation = 1 - np.cumsum(poisson.pmf(np.arange(num_coverage), mu=avg_coverage, loc=0))
    se = np.sqrt(avg_coverage/len(arys))
    x_coordinate = np.arange(1, num_coverage+1)
    plt.figure(figsize=(16,12))
    for i in range(len(arys)):
        coverage_ary = arys[i]
        plt.plot(x_coordinate, coverage_ary/poisson_expectation[0], label = code) # Can put code in as well
    plt.plot(x_coordinate, poisson_expectation/poisson_expectation[0], label = 'Poisson', ls='--', color = 'k', linewidth = 5)
    plt.plot(x_coordinate, (poisson_expectation + n_se*se)/poisson_expectation[0], ls='--', color = 'k', linewidth = 5)
    plt.plot(x_coordinate, (poisson_expectation - n_se*se)/poisson_expectation[0], ls='--', color = 'k', linewidth = 5)
    plt.xticks(x_coordinate)
    plt.xlabel('Coverage (x)')
    plt.ylabel('Sequencing Skew')
    #plt.legend()
    plt.title('Sequencing Skew')
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches = "tight", dpi=300)
def plot_info_vs_af(vcf, afs, MAF_ary = np.array([0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.95, 1]),
                   save_fig = False, outdir = 'graphs/', save_name = 'info_vs_af.png'):
    df = pd.merge(vcf[['chr', 'pos', 'ref', 'alt', 'info']], afs, on=['chr', 'pos', 'ref', 'alt'], how="left").dropna()
    df['classes'] = np.digitize(df['MAF'], MAF_ary)
    plt.figure(figsize = (12,8))
    sns.violinplot(data=df, x="classes", y="info")
    plt.xlabel('Allele Frequencies (%)')
    plt.ylabel('INFO_SCORE')
    plt.title('INFO Score vs Allele Frequencies')
    ax = plt.gca()
    ax.set_xticklabels(MAF_ary[np.sort(df['classes'].unique()) - 1])
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches = "tight", dpi=300)
def plot_r2_vs_info(df, save_fig = False, outdir = 'graphs/', save_name = 'r2_vs_info.png'):
    # Input df has AF bins, r2, avg_info, and bin counts
    pivot = df.pivot('corr', 'INFO_SCORE', 'Bin Count')
    plt.figure(figsize = (8,6))
    plt.imshow(pivot, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Bin Count')
    y_ticks = sorted(df['corr'].unique().round(3))
    x_ticks = sorted(df['INFO_SCORE'].unique().round(3))
    plt.xlabel('Average INFO')
    plt.ylabel('Average $r^2$')
    plt.title('Heatmap of correlation vs info_score with bin counts')
    plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation = 45)
    plt.yticks(np.arange(len(y_ticks)), y_ticks)
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches = "tight", dpi=300)