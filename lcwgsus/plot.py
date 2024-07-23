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
from plotnine import *
import patchworklib as pw
from PIL import Image
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
from .read import *
from .process import *

__all__ = [
    "save_figure", "plot_afs", "plot_imputation_accuracy_typed", "plot_imputation_accuracy_gw", "plot_imputation_accuracy_by_genotype",  "combine_imputation_accuracy_plots", "plot_imputation_accuracy_sequential",
    "plot_sequencing_skew", "plot_info_vs_af", "plot_r2_vs_info", "plot_pc", "plot_violin", "plot_rl_distribution", "plot_imputation_metric_in_region", "plot_hla_diversity", "plot_hla_allele_frequency", "plot_hla_imputation_accuracy"
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
                             threshold=None,
                             title='',
                             marker_size=100,
                             colorbar_cmap=COLORBAR_CMAP,
                             line_colors=CATEGORY_CMAP_HEX,
                             set_ylim=True,
                             subplot=False,
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

    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    plt.grid(False)

    magnitude = ceil - floor
    bounds = np.logspace(floor, ceil, magnitude + 1)
    norm = mcolors.BoundaryNorm(bounds, COLORBAR_CMAP.N)
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
        allele_counts = triplet[c2]

        plt.plot(x, vals, label=label, c=line_colors[i])
        if set_ylim:
            plt.ylim((-0.05, 1.05))
        if not subplot:
            plt.xticks(x, afs, rotation=45)
        else:
            ax.set_xticks(x, ['' for _ in afs])

        im = ax.scatter(x,
                        vals,
                        c=allele_counts,
                        edgecolor='black',
                        cmap=colorbar_cmap,
                        norm=norm,
                        s=marker_size)
    if not subplot:
        plt.colorbar(im,
                    boundaries=bounds,
                    ticks=bounds,
                    format=FuncFormatter(fmt),
                    label='Allele frequency counts')

        plt.xlabel('GnomAD allele frequencies (%)')
        plt.title(title)
        ax.legend(
            loc = 'upper right',
            bbox_to_anchor=(1, 0.77),
            framealpha=1,
            prop={'size': 9}
        )
        plt.ylabel('Aggregated imputation accuracy ($r^2$)')

    ax.grid()
    fig.tight_layout()

    save_figure(save_fig, outdir, save_name)
    return fig


def plot_imputation_accuracy_gw(impacc_lst,
                                metric='r2',
                                labels=None,
                                threshold=None,
                                title='',
                                marker_size=100,
                                colorbar_cmap=COLORBAR_CMAP, line_colors=CATEGORY_CMAP_HEX,
                                set_ylim=True,
                                subplot=False,
                                save_fig=False,
                                outdir=None,
                                save_name=None):
    cols = ['AF', metric, metric + '_AC']
    df_lst = [impacc[cols] for impacc in impacc_lst]

    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    plt.grid(False)

    magnitude = 5
    bounds = np.logspace(3, 8, magnitude + 1)
    norm = mcolors.BoundaryNorm(bounds, colorbar_cmap.N)
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
        allele_counts = triplet[c2]

        plt.plot(x, vals, label=label, c = line_colors[i])
        if set_ylim:
            plt.ylim((-0.05, 1.05))
        if not subplot:
            plt.xticks(x, afs, rotation=45)
        else:
            ax.set_xticks(x, ['' for _ in afs])

        im = ax.scatter(x,
                        vals,
                        c=allele_counts,
                        edgecolor='black',
                        cmap=colorbar_cmap,
                        norm=norm,
                        s=marker_size)
    if not subplot:
        plt.colorbar(im,
                    boundaries=bounds,
                    ticks=bounds,
                    format=FuncFormatter(fmt),
                    label='Allele frequency counts')

        plt.xlabel('GnomAD allele frequencies (%)')
        plt.title(title)
        ax.legend(
            loc = 'upper right',
            bbox_to_anchor=(0.99, 0.77),
            framealpha=1,
            prop={'size': 9}
        )
        plt.ylabel('Aggregated imputation accuracy ($r^2$)')
    ax.grid()
    fig.tight_layout()

    save_figure(save_fig, outdir, save_name)
    return fig

def plot_imputation_accuracy_by_genotype(impacc,
                                         metrics = ['ccd_homref', 'ccd_het', 'ccd_homalt'],
                                         threshold=None,
                                         title='',
                                         marker_size=100,
                                         colorbar_cmap=COLORBAR_CMAP,
                                         line_colors=CATEGORY_CMAP_HEX,
                                         set_ylim=True,
                                         subplot=False,
                                         save_fig=False,
                                         outdir=None,
                                         save_name=None):
    ceil = 0
    floor = 100

    for m in metrics:
        cols = ['AF', m, m + '_AC']
        triplet = impacc[cols]
        c0, c1, c2 = tuple(list(triplet.columns))
        triplet[c1] = triplet[c1].replace(-9, 0)
        magnitude_ceil = round_to_nearest_magnitude(triplet[c2].max())
        magnitude_floor = round_to_nearest_magnitude(triplet[c2].min(), False)
        if ceil < magnitude_ceil:
            ceil = magnitude_ceil
        if floor > magnitude_floor:
            floor = magnitude_floor

    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    plt.grid(False)

    magnitude = ceil - floor
    bounds = np.logspace(floor, ceil, magnitude + 1)
    norm = mcolors.BoundaryNorm(bounds, colorbar_cmap.N)
    fmt = lambda x, pos: '{:.0e}'.format(x)

    for i, m in enumerate(metrics):
        cols = ['AF', m, m + '_AC']
        triplet = impacc[cols]
        if threshold is not None:
            triplet = triplet[triplet['AF'] >= threshold]
        c0, c1, c2 = tuple(list(triplet.columns))
        label = m
        
        x = np.arange(triplet.shape[0])
        afs = generate_af_axis(triplet[c0].values)
        vals = triplet[c1]
        allele_counts = triplet[c2]

        plt.plot(x, vals, label=label, c=line_colors[i])
        if set_ylim:
            plt.ylim((-0.05, 1.05))
        if not subplot:
            plt.xticks(x, afs, rotation=45)
        else:
            ax.set_xticks(x, ['' for _ in afs])

        im = ax.scatter(x,
                        vals,
                        c=allele_counts,
                        edgecolor='black',
                        cmap=colorbar_cmap,
                        norm=norm,
                        s=marker_size)
    if not subplot:
        plt.colorbar(im,
                    boundaries=bounds,
                    ticks=bounds,
                    format=FuncFormatter(fmt),
                    label='Allele frequency counts')

        plt.xlabel('GnomAD allele frequencies (%)')
        plt.title(title)
        ax.legend(
            loc = 'upper right',
            bbox_to_anchor=(1, 0.77),
            framealpha=1,
            prop={'size': 9}
        )
        plt.ylabel('Average concordance')

    ax.grid()
    fig.tight_layout()

    save_figure(save_fig, outdir, save_name)
    return fig

def combine_imputation_accuracy_plots(fig1, 
                                      fig2, 
                                      inset_position=[0.395, 0.2, 0.382, 0.37], 
                                      threshold=0.01,
                                      save_fig=False,
                                      outdir=None,
                                      save_name=None):
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    plt.close(fig1)
    buf1.seek(0)
    f1 = Image.open(buf1)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    buf2.seek(0)
    f2 = Image.open(buf2)

    if threshold == 0.01:
        # Left, bottom, width, height
        inset_position = [0.395, 0.2, 0.382, 0.37]
    elif threshold == 0.02:
        inset_position = [0.441, 0.2, 0.333, 0.34]
    else:
        pass

    left, bottom, width, height = inset_position

    fig_width, fig_height = f1.size

    inset_width, inset_height = int(fig_width * width), int(fig_height * height)
    inset_left, inset_bottom = int(fig_width * left), int(fig_height * bottom)

    f2_resized = f2.resize((inset_width, inset_height), Image.ANTIALIAS)
    f1.paste(f2_resized, (inset_left, fig_height - inset_bottom - inset_height), f2_resized)

    if save_fig:
        check_outdir(outdir)
        f1.save(outdir + save_name) 
    return f1

def plot_imputation_accuracy_sequential(ix, impaccs, labels, title, 
                                        threshold, func = plot_imputation_accuracy_gw, 
                                        save_fig = False, outdir = '', save_prefix = ''):
    dfs = [impaccs[i] for i in ix] # ix should be ordered
    figs = []

    for i in range(1, len(ix) + 1):
        seq_indices = list(range(i))
        tmp_dfs = [dfs[j] for j in seq_indices]
        tmp_labels = [labels[j] for j in seq_indices]
        
        fig1 = func(tmp_dfs, labels = tmp_labels, title = title)
        fig2 = func(tmp_dfs, labels = tmp_labels, threshold = threshold, title = title, set_ylim = False, subplot = True)
        f1 = combine_imputation_accuracy_plots(fig1, fig2, 
                                               threshold = threshold, 
                                               save_fig = save_fig, 
                                               outdir = outdir, 
                                               save_name = save_prefix + str(i) + '.png')
        figs.append(f1)
    return figs

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

def plot_hla_imputation_accuracy(hla, hla_dirs, labels, indices = None, cmap = CATEGORY_CMAP_STR, plot_onefield = True, save_fig=False, outdir=None, save_name=None):
    hla_reports = []
    
    colors = plt.get_cmap(cmap).colors[:(len(labels))]
    hex_codes = [mcolors.to_hex(color) for color in colors]
    colors = dict(zip(labels, hex_codes))
    
    if indices is not None:
        hla_dirs = [hla_dirs[i] for i in indices]
        labels = [labels[i] for i in indices]
        
    for d, l in zip(hla_dirs, labels):
        report = calculate_hla_imputation_accuracy(d, hla, l)
        hla_reports.append(report)
    report = pd.concat(hla_reports)
    report['Locus'] = pd.Categorical(report['Locus'], categories=HLA_GENES[::-1], ordered=True)
    report['Source'] = pd.Categorical(report['Source'], categories=labels, ordered=True)
    report = report.sort_values(by = 'Locus')

    twofield = report[report['Resolution'] == 'Two field']
    plot2 = (
        ggplot(twofield, aes(x='Concordance', y='Locus', color='Source')) +
        geom_point(size=2) + ggtitle('Two Field') +
        scale_color_manual(values=colors) +
#         labs(x='Concordance', y='Locus', color='Source') + 
        theme_minimal() + 
        theme(axis_text_y=element_text(angle=0), title=element_text(hjust=2))
    )
    p2 = pw.load_ggplot(plot2, figsize=(4,4))
    
    if plot_onefield:
        onefield = report[report['Resolution'] == 'One field']
        plot1 = (
            ggplot(onefield, aes(x='Concordance', y='Locus', color='Source')) +
            geom_point(size=2) + ggtitle('One Field') + theme_minimal() +
            scale_color_manual(values=colors) +
            theme(axis_text_y=element_text(angle=0), title=element_text(hjust='1'))
        )
        p1 = pw.load_ggplot(plot1, figsize=(4,4))
        p12 = (p1|p2)
    else:
        p12 = p2

    if save_fig:
        check_outdir(outdir)
        p12.savefig(outdir + save_name)
    return p12

def plot_hla_imputation_accuracy_by_type(hla, 
                                         hla_dirs, labels, cmap = CATEGORY_CMAP_STR, combine = True, save_fig=False, outdir=None, save_name=None):
    lc = read_hla_lc_imputation_results(hla_dirs[0], retain = 'fv')
    chip = read_hla_chip_imputation_results(hla_dirs[1], retain = 'fv')
    
    ccd_dict_chip = compare_hla_types_by_type(hla, chip)
    ccd_dict_chip = calculate_hla_concordance_by_type(ccd_dict_chip, verbose = False)
    
    samples = chip['SampleID'].unique()
    lc = lc[lc['SampleID'].isin(samples)].sort_values(by = ['SampleID', 'Locus'])

    ccd_dict_lc = compare_hla_types_by_type(hla, lc)
    ccd_dict_lc = calculate_hla_concordance_by_type(ccd_dict_lc, verbose = False)
    
    if combine:
        fig = plt.figure(figsize=(8, 6), dpi = 300)
        for i, l in enumerate(HLA_GENES):
            df = ccd_dict_chip[l]
            plt.scatter(df['Sum'], df['Accuracy'], c = CATEGORY_CMAP_HEX[i], label = f'{l} ({labels[0]})', marker = 'o')

#         for i, l in enumerate(HLA_GENES):
            df = ccd_dict_lc[l]
            plt.scatter(df['Sum'], df['Accuracy'], c = CATEGORY_CMAP_HEX[i], label = f'{l} ({labels[1]})', marker = 'x')
       
        ax = plt.gca()
        ax.grid(True)
        ax.set_xlabel('HLA (true) allele counts')
        ax.set_ylabel('Concordance')
        ax.set_title('HLA imputation accuracy by type')
        plt.show()
    
    else:
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize = (10,8), dpi = 300)
        
        for i, l in enumerate(HLA_GENES):
            df = ccd_dict_chip[l]
            ax1.scatter(df['Sum'], df['Accuracy'], c = CATEGORY_CMAP_HEX[i], label = l)
        ax1.grid(True)
        ax1.set_xlabel('HLA (true) allele counts')
        ax1.set_ylabel('Concordance')
        ax1.set_title('HLA imputation accuracy by type (chip)')

        for i, l in enumerate(HLA_GENES):
            df = ccd_dict_lc[l]
            ax2.scatter(df['Sum'], df['Accuracy'], c = CATEGORY_CMAP_HEX[i], label = l)
        ax2.grid(True)
        ax2.set_xlabel('HLA (true) allele counts')
        ax2.set_ylabel('Concordance')
        ax2.set_title('HLA imputation accuracy by type (lc)')
        plt.show()
        
    save_figure(save_fig, outdir, save_name)
    return None