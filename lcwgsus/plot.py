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
from matplotlib.ticker import FuncFormatter
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
from .calculate import *
from .variables import *
from .process import *

__all__ = [
    "save_figure", "plot_afs", "plot_imputation_accuracy_typed", "plot_imputation_accuracy_gw",
    "plot_sequencing_skew", "plot_info_vs_af", "plot_r2_vs_info", "plot_pc", "plot_violin", "plot_rl_distribution", "plot_imputation_metric_in_region", "plot_hla_diversity", "plot_hla_allele_frequency"
]

def save_figure(save: bool, outdir: str, name: str) -> None:
    if save:
        check_outdir(outdir)
        plt.savefig(outdir + name, bbox_inches="tight", dpi=300)
    return None


def plot_afs(df1: pd.DataFrame,
             df2: pd.DataFrame,
             save_fig: bool = False,
             outdir: str = 'graphs/',
             save_name: str = 'af_vs_af.png') -> float:
    # df1 is the chip df with cols chr, pos, ref, alt and prop
    # df2 is the other df with the same cols
    df = pd.merge(df1, df2, on=['chr', 'pos', 'ref', 'alt'], how='inner')
    plt.scatter(df['prop_x'] * 100, df['prop_y'] * 100)
    plt.xlabel('ChIP MAF (%)')
    plt.ylabel('GGVP AF (%)')
    plt.title('Check AFs')
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return np.corrcoef(df['prop_x'], df['prop_y'])[0, 1]
# Currently deprecated


def plot_sequencing_skew(arys,
                         avg_coverage,
                         n_se=1.96,
                         code=None,
                         num_coverage=5,
                         save_fig=False,
                         save_name='prop_genome_at_least_coverage.png',
                         outdir='graphs/'):
    poisson_expectation = 1 - np.cumsum(
        poisson.pmf(np.arange(num_coverage), mu=avg_coverage, loc=0))
    se = np.sqrt(avg_coverage / len(arys))
    x_coordinate = np.arange(1, num_coverage + 1)
    plt.figure(figsize=(16, 12))
    for i in range(len(arys)):
        coverage_ary = arys[i]
        plt.plot(x_coordinate,
                 coverage_ary / poisson_expectation[0],
                 label=code)  # Can put code in as well
    plt.plot(x_coordinate,
             poisson_expectation / poisson_expectation[0],
             label='Poisson',
             ls='--',
             color='k',
             linewidth=5)
    plt.plot(x_coordinate,
             (poisson_expectation + n_se * se) / poisson_expectation[0],
             ls='--',
             color='k',
             linewidth=5)
    plt.plot(x_coordinate,
             (poisson_expectation - n_se * se) / poisson_expectation[0],
             ls='--',
             color='k',
             linewidth=5)
    plt.xticks(x_coordinate)
    plt.xlabel('Coverage (x)')
    plt.ylabel('Sequencing Skew')
    #plt.legend()
    plt.title('Sequencing Skew')
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return None


def plot_info_vs_af(vcf,
                    afs,
                    MAF_ary=MAF_ARY,
                    save_fig=False,
                    outdir='graphs/',
                    save_name='info_vs_af.png'):
    df = pd.merge(vcf[['chr', 'pos', 'ref', 'alt', 'info']],
                  afs,
                  on=['chr', 'pos', 'ref', 'alt'],
                  how="left").dropna()
    df['classes'] = np.digitize(df['MAF'], MAF_ary)
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, x="classes", y="info")
    plt.xlabel('Allele Frequencies (%)')
    plt.ylabel('INFO_SCORE')
    plt.title('INFO Score vs Allele Frequencies')
    ax = plt.gca()
    ax.set_xticklabels(MAF_ary[np.sort(df['classes'].unique()) - 1])
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return None


def plot_r2_vs_info(df,
                    save_fig=False,
                    outdir='graphs/',
                    save_name='r2_vs_info.png'):
    # Input df has AF bins, r2, avg_info, and bin counts
    pivot = df.pivot('corr', 'INFO_SCORE', 'Bin Count')
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Bin Count')
    y_ticks = sorted(df['corr'].unique().round(3))
    x_ticks = sorted(df['INFO_SCORE'].unique().round(3))
    plt.xlabel('Average INFO')
    plt.ylabel('Average $r^2$')
    plt.title('Heatmap of correlation vs info_score with bin counts')
    plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation=45)
    plt.yticks(np.arange(len(y_ticks)), y_ticks)
    if save_fig:
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return None


def plot_pc(df, num_PC=2, save_fig=False, save_name='graphs/PCA.png') -> None:
    # Input df has 'PC_1', 'PC_2', ... columns and an additional column called 'ethnic'
    plt.figure(figsize=(10, 8))

    PC1 = df.columns[df.columns.str.contains('PC')][0]
    PC2 = df.columns[df.columns.str.contains('PC')][1]
    if num_PC == 2:
        labels = df['ethnic']
        targets = labels.unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets, colors):
            indices_to_keep = labels == target
            plt.scatter(
                df.loc[indices_to_keep, PC1],
                df.loc[indices_to_keep, PC2],
                color=color,
                label=target,
                s=50,
            )
        plt.legend()
        plt.title('PCA Plot')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
    elif num_PC > 2:
        plot = sns.pairplot(df[['PC_' + str(i)
                                for i in range(1, num_PC + 1)] + ['ethnic']],
                            hue="ethnic",
                            diag_kind="kde",
                            diag_kws={
                                "linewidth": 0,
                                "shade": False
                            })
        plot.fig.suptitle('PCA Plot', y=1.02)
    else:
        print("You should at least plot the first two PCs.")
    if save_fig:
        plt.savefig(save_name, bbox_inches="tight", dpi=300)
    plt.show()
    return None


def plot_imputation_accuracy_typed(impacc_lst,
                             metric='r2',
                             labels=None,
                             title='',
                             marker_size=100,
                             cmap_str='GnBu',
                             save_fig=False,
                             outdir=None,
                             save_name=None):
    ceil = 0
    floor = 100
    cols = ['AF', metric, metric + '_AC']
    df_lst = [impacc[cols] for impacc in impacc_lst]

    for triplet in df_lst:
        c0, c1, c2 = tuple(list(triplet.columns))
        triplet[c1] = triplet[c1].replace(-9, 0)
        magnitude_ceil = round_to_nearest_magnitude(triplet[c2].max())
        magnitude_floor = round_to_nearest_magnitude(triplet[c2].min(), False)
        if ceil < magnitude_ceil:
            ceil = magnitude_ceil
        if floor > magnitude_floor:
            floor = magnitude_floor

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    plt.grid(False)

    cmap = plt.get_cmap(cmap_str)
    magnitude = ceil - floor
    bounds = np.logspace(floor, ceil, magnitude + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fmt = lambda x, pos: '{:.0e}'.format(x)

    for i in range(len(df_lst)):
        triplet = df_lst[i]
        c0, c1, c2 = tuple(list(triplet.columns))

        label = c1 if labels is None else labels[i]

        x = np.arange(triplet.shape[0])
        afs = generate_af_axis(triplet[c0].values)
        vals = triplet[c1]
        color = triplet[c2]

        plt.plot(x, vals, label=label)
        plt.xticks(x, afs, rotation=45)

        im = ax.scatter(x,
                        vals,
                        c=color,
                        edgecolor='black',
                        cmap=cmap,
                        norm=norm,
                        s=marker_size)
    plt.colorbar(im,
                 boundaries=bounds,
                 ticks=bounds,
                 format=FuncFormatter(fmt),
                 label='Allele Frequency Counts')

    plt.xlabel('gnomAD allele frequencies (%)')
    plt.title(title)
    plt.legend()
    plt.ylabel('Aggregated imputation accuracy ($r^2$)')
    ax = plt.gca()
    ax.grid()

    if save_fig:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return None


def plot_imputation_accuracy_gw(impacc_lst,
                                metric='r2',
                                labels=None,
                                threshold=None,
                                title='',
                                marker_size=100,
                                cmap_str='GnBu',
                                save_fig=False,
                                outdir=None,
                                save_name=None):
    cols = ['AF', metric, metric + '_AC']
    df_lst = [impacc[cols] for impacc in impacc_lst]

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    plt.grid(False)

    cmap = plt.get_cmap('GnBu')
    magnitude = 5
    bounds = np.logspace(3, 8, magnitude + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fmt = lambda x, pos: '{:.0e}'.format(x)

    for i in range(len(df_lst)):
        triplet = df_lst[i]
        if threshold is not None:
            triplet = triplet[triplet['AF'] >= threshold]
        c0, c1, c2 = tuple(list(triplet.columns))

        label = c1 if labels is None else labels[i]

        x = np.arange(triplet.shape[0])
        afs = generate_af_axis(triplet[c0].values)
        vals = triplet[c1]
        color = triplet[c2]

        plt.plot(x, vals, label=label)
        plt.xticks(x, afs, rotation=45)

        im = ax.scatter(x,
                        vals,
                        c=color,
                        edgecolor='black',
                        cmap=cmap,
                        norm=norm,
                        s=marker_size)
    plt.colorbar(im,
                 boundaries=bounds,
                 ticks=bounds,
                 format=FuncFormatter(fmt),
                 label='Allele Frequency Counts')

    plt.xlabel('gnomAD allele frequencies (%)')
    plt.title(title)
    plt.legend()
    plt.ylabel('Aggregated imputation accuracy ($r^2$)')
    ax = plt.gca()
    ax.grid()

    if save_fig:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        plt.savefig(outdir + save_name, bbox_inches="tight", dpi=300)
    return None


def plot_violin(df,
                x,
                y,
                hue=None,
                title=None,
                save_fig=False,
                outdir=None,
                save_name=None):
    plt.figure(figsize=(10, 6))
    if hue is None:
        sns.violinplot(data=df, x=x, y=y, cut=0)
    else:
        sns.violinplot(data=df, x=x, y=y, hue=hue, cut=0)

    if title is not None:
        plt.title(title)

    save_figure(save_fig, outdir, save_name)
    return None


def plot_rl_distribution(lst,
                         title='Read length distribution',
                         save_fig=False,
                         outdir=None,
                         save_name=None):
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    plt.hist(lst, bins=20, ec='black')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel('Length (bases)')
    ax.set_ylabel('Count')

    if title is not None:
        ax.set_title(title)

    save_figure(save_fig, outdir, save_name)
    return None


def plot_imputation_metric_in_region(
        h,
        chr,
        pos,
        metric='r2',
        start=None,
        end=None,
        window=1e5,
        title='Imputation accuracy at selected region',
        show_fig=True,
        save_fig=False,
        outdir=None,
        save_name=None,
        ax=None):  
    h = h[h['chr'] == chr]
    h = h[h[metric] != -9]
    if start is not None:
        s = start
        e = end
    else:
        s = max(pos - window / 2, 0)
        e = pos + window / 2

    df = h[(h['pos'] < e) & (h['pos'] > s)]
    scale = max(0, (len(str(e - s)) - 2))
    buffer = (10**scale)

    if show_fig & (len(df) != 0):
        if ax is None:
            ax = plt.gca()
        ax.scatter(df['pos'],
                   df[metric],
                   c=df['MAF'],
                   cmap='GnBu',
                   s=30,
                   ec='black')
        ax.plot(df['pos'], df[metric], linewidth=1)
        ax.grid()
        ax.set_xticks(np.linspace(max(-10, s - buffer), e + buffer, num = 11))
        label_format = '{:,.0f}'
        ax.set_xticklabels([label_format.format(x) for x in ax.get_xticks().tolist()], rotation = 45)
        ax.set_xlim((max(-10, s - buffer), e + buffer))
        ax.set_ylim((-0.05, 1.05))
        plt.colorbar(ax.collections[0], ax=ax) 
        ax.set_xlabel('chr' + str(chr) + ':' + str(s) + '-' + str(e))
        ax.set_ylabel(metric)
        ax.set_title(title)
        
        save_figure(save_fig, outdir, save_name)
    return df[metric].mean()

def plot_hla_diversity(hla_alleles_df):
    hla_counts = hla_alleles_df.groupby(['Locus', 'Allele']).size().unstack(fill_value=0)

    top_hla_counts = hla_counts.apply(group_top_n_alleles, axis=1)
    lst = []
    for i in HLA_GENES:
        cols = top_hla_counts.columns[top_hla_counts.columns.str.startswith(i + '*')]
        tmp = top_hla_counts[cols]
        sorted_columns = tmp.loc[i].sort_values(ascending = True).index
        sorted_df = tmp[sorted_columns]

        lst.append(sorted_df)
    res = pd.concat(lst, axis = 1)
    res['Others'] = top_hla_counts['Others']
    cols = ['Others'] + [col for col in res.columns if col != 'Others']
    res = res[cols]

    cumulative_sums = res.cumsum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, col in enumerate(res.columns):
        ax.bar(res.index, res[col], bottom=res.iloc[:, :idx].sum(axis=1))

        for category in res.index:
            height = res.loc[category, col]
            if height > 0:
                bottom = cumulative_sums.loc[category, col] - height
                ax.text(x=category, y=bottom + height / 2, s=col, ha='center', va='center', fontsize=8, color='white')

    ax.set_xlabel('HLA gene')
    ax.set_ylabel('Frequency')
    ax.set_title('HLA allelic diversity for 250 people')

    plt.show()
    return None

def plot_hla_allele_frequency(hla_alleles_df, gene):
    tmp = hla_alleles_df[hla_alleles_df['Locus'] == gene]
    counts = tmp['Allele'].value_counts()
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('HLA-' + gene)
    plt.xlabel('Alleles')
    plt.ylabel('Counts')
    plt.xticks(rotation = 45, fontsize=9)
    plt.show()
    return None