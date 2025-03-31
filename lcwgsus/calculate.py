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
from collections import Counter
from typing import Union, Tuple, List
pd.options.mode.chained_assignment = None

from .auxiliary import *
from .variables import *
from .read import *

__all__ = [
    "calculate_per_bin_kmer_error_rate", "calculate_af", "calculate_ss_cumsum_coverage",
    "calculate_average_info_score",
    "calculate_corrcoef", "calculate_concordance",
    "calculate_imputation_accuracy_metrics", "average_h_metrics",
    "calculate_h_imputation_accuracy", "generate_h_impacc",
    "calculate_v_imputation_accuracy", "average_v_metrics",
    "generate_v_impacc", "calculate_weighted_average", "average_impacc_by_chr",
    "round_to_nearest_magnitude", "calculate_imputation_summary_metrics",
    "calculate_imputation_sumstats", "calculate_shannon_entropy", "calculate_hla_concordance", "calculate_hla_concordance_by_type", 
    "generate_hla_imputation_report", "calculate_hla_entropy", "calculate_hla_imputation_accuracy"
]

# To clean
def calculate_per_bin_kmer_error_rate(code, strand, file, intervals = [50,100,150,200,250,300,350,400,450,500,550,600]):
    result = pd.read_csv(file, sep = '\t', usecols = ['read_id', 'number_of_kmers_at_threshold', 'number_of_solid_kmers_at_threshold'])
    result['read_id'] = result['read_id'].str.split().str[0]
    bins = pd.read_csv('results/kmer/'+code+'/fragment_length.tsv', sep = '\t', header = None, names = ['read_id', 'bin'], dtype={'read_id': 'string', 'bin': 'int'})
    df = pd.merge(result, bins, on='read_id', how="left")

    error_rate_ary=[]
    for i in intervals:
        error_rate = df.loc[df['bin'] == i, 'number_of_solid_kmers_at_threshold'].sum()/df.loc[df['bin'] == i, 'number_of_kmers_at_threshold'].sum()
        error_rate_ary.append(error_rate)
    error_rate = df.loc[df['bin']>intervals[-1], 'number_of_solid_kmers_at_threshold'].sum()/df.loc[df['bin']>intervals[-1], 'number_of_kmers_at_threshold'].sum()
    error_rate_ary.append(error_rate)

    res = pd.DataFrame({code: error_rate_ary}).T
    res.to_csv('results/kmer/'+code+'/read'+strand+'/per_bin_kmer_error_rate_read'+strand+'.txt', header=False, sep='\t')


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

def round_to_nearest_magnitude(number, ceil = True):
    if number == 0:
        return 0
    if ceil:
        magnitude = int(np.ceil(np.log10(abs(number))))
    else:
        magnitude = int(np.floor(np.log10(abs(number))))
    return magnitude

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


def calculate_ss_cumsum_coverage(df: pd.DataFrame,
                                 num_coverage: int = 5) -> np.ndarray:
    df['bases'] = df['end'] - df['start']
    df = df.groupby(['cov']).bases.sum().reset_index()
    df['prop bases'] = df['bases'] / df.bases.sum()
    df['cum prop'] = np.cumsum(df['prop bases'].to_numpy())
    df['prop genome at least covered'] = (1 - df['cum prop'].shift(1))
    df = df.dropna()
    coverage_ary = df['prop genome at least covered'].values[:num_coverage]
    return coverage_ary


def calculate_average_info_score(
        chromosomes: Union[List[int], List[str]],
        vcf: pd.DataFrame,
        af: pd.DataFrame,
        chip_df: pd.DataFrame,
        MAF_ary: np.ndarray = MAF_ARY) -> pd.DataFrame:
    # Input vcf is a df that contains chr, pos, ref, alt, info
    # af and chip_df are used to filtered out variants by position
    # returns average INFO in each MAF bin
    info = pd.merge(vcf, chip_df, on=['chr', 'pos', 'ref', 'alt'], how='inner')
    info = pd.merge(info, af, on=['chr', 'pos', 'ref', 'alt'], how='inner')
    info = info[['chr', 'pos', 'ref', 'alt', 'INFO_SCORE', 'MAF']]
    info['classes'] = np.digitize(info['MAF'], MAF_ary)
    info['classes'] = info['classes'].apply(lambda x: len(MAF_ary) - 1
                                            if x == len(MAF_ary) else x)
    score = info.copy().groupby(['classes'])['INFO_SCORE'].mean().reset_index()
    return score

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

    n_sample = n_rf_all - np.sum((r1 == 0) & (r2 <= 0.5)) # This is the number of sample which is not (0/0 and also imputed to be 0/0)
    n_nrc = np.sum(np.abs(r1_het - r2_het) < 0.5) + np.sum(np.abs(r1_homalt - r2_homalt) < 0.5)
    if n_sample == 0:
        nrc = -9
    else:
        nrc = n_nrc/n_sample

    rf_homref = calculate_concordance(r1_homref, r2_homref)
    rf_het = calculate_concordance(r1_het, r2_het)
    rf_homalt = calculate_concordance(r1_homalt, r2_homalt)

    return [nrc, n_sample, rf_all, n_rf_all, rf_homref, n_homref, rf_het, n_het, rf_homalt, n_homalt]


def calculate_h_imputation_accuracy(chip,
                                    lc,
                                    af,
                                    impacc_colnames=IMPACC_COLS,
                                    save_file=False,
                                    outdir=None,
                                    save_name=None):
    stack_lst = [[] for _ in range(len(impacc_colnames))]

    for _, (_, r1), (_, r2) in zip(range(len(chip)), chip.iterrows(),
                                   lc.iterrows()):
        r1 = np.array(r1.values[4:]).astype(
            float
        )  # Assume that quilt and chip has all cols removed except for commom_cols, modify the number 4 if necessary
        r2 = np.array(r2.values[4:]).astype(float)
        metrics = calculate_imputation_accuracy_metrics(r1, r2)
        stack_lst = append_lst(metrics, stack_lst)

    for name, col in zip(impacc_colnames, stack_lst):
        af[name] = col

    if save_file:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        af.to_csv(outdir + save_name, sep='\t', index=False, header=True)
    return af

def average_h_metrics(df, cols = IMPACC_METRICS, placeholder = -9):
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


def generate_h_impacc(df,
                      MAF_ary=MAF_ARY,
                      colnames=IMPACC_H_COLS,
                      save_impacc=False,
                      outdir=None,
                      save_name=None):
    impacc = pd.DataFrame({'AF': MAF_ary})

    stack_lst = [[] for _ in range(len(colnames))]

    for i in range(np.size(MAF_ary)):
        if i == 0:
            tmp = df[df['MAF'] == 0]
        else:
            tmp = df[(MAF_ary[i] >= df['MAF']) & (df['MAF'] > MAF_ary[i - 1])]
        metrics = average_h_metrics(tmp)
        stack_lst = append_lst(metrics, stack_lst)

    for name, col in zip(colnames, stack_lst):
        impacc[name] = col

    if save_impacc:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        impacc.to_csv(outdir + save_name, sep='\t', index=False, header=True)
    return impacc


def calculate_v_imputation_accuracy(chip,
                                    lc,
                                    af,
                                    MAF_ary=MAF_ARY,
                                    impacc_colnames=IMPACC_V_COLS,
                                    common_cols=COMMON_COLS,
                                    save_file=False,
                                    outdir=None,
                                    save_name=None):
    stack_lst = [[] for _ in range(len(impacc_colnames))]

    chip_af = pd.merge(chip, af, on=common_cols)
    lc_af = pd.merge(lc, af, on=common_cols)

    chip_samples = valid_sample(chip.iloc[0, :])
    lc_samples = valid_sample(lc.iloc[0, :])

    for chip_sample, lc_sample in zip(chip_samples, lc_samples):
        chip = chip_af[['MAF', chip_sample]]
        lc = lc_af[['MAF', lc_sample]]

        for i in range(np.size(MAF_ary)):
            if i == 0:
                tmp_chip = chip[chip_af['MAF'] == 0]
                tmp_lc = lc_af[lc_af['MAF'] == 0]
            else:
                tmp_chip = chip[(MAF_ary[i] >= chip['MAF'])
                                & (chip['MAF'] > MAF_ary[i - 1])]
                tmp_lc = lc[(MAF_ary[i] >= lc['MAF'])
                            & (lc['MAF'] > MAF_ary[i - 1])]

            r1 = np.array(tmp_chip[chip_sample].values).astype(float)
            r2 = np.array(tmp_lc[lc_sample].values).astype(float)
            tmp_metrics = calculate_imputation_accuracy_metrics(r1, r2)
            metrics = [chip_sample, MAF_ary[i], tmp_chip.shape[0]]
            metrics = fix_v_metrics(metrics, tmp_metrics)
            stack_lst = append_lst(metrics, stack_lst)

    res_df = pd.DataFrame(dict(zip(impacc_colnames, stack_lst)))

    if save_file:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        res_df.to_csv(outdir + save_name, sep='\t', index=False, header=True)
    return res_df


def average_v_metrics(df, cols=IMPACC_METRICS, placeholder=-9):
    res_ary = []
    res_ary.append(df['n_variants'].values[0])

    for c in cols:
        c_count = c + '_BC'
        tmp = df[df[c] != placeholder][[c, c_count]]
        num = tmp[c_count].sum()
        n_variants = tmp[c_count].mean() if num != 0 else 0
        if n_variants == 0:
            res_ary.append(placeholder)
        else:
            avg = (tmp[c] * tmp[c_count]).sum() / num
            res_ary.append(avg)
        res_ary.append(n_variants)
        res_ary.append(num)
    return res_ary


def generate_v_impacc(df,
                      MAF_ary=MAF_ARY,
                      colnames=IMPACC_H_COLS,
                      save_impacc=False,
                      outdir=None,
                      save_name=None):
    # BC entries for the vertical calculation does not make sense, as each person can have the same genotypes at different variants, so it does not make sense to either calculate the sum nor the average.
    impacc = pd.DataFrame({'AF': MAF_ary})

    stack_lst = [[] for _ in range(len(colnames))]

    for i in range(np.size(MAF_ary)):
        tmp = df[df['AF'] == MAF_ary[i]]
        metrics = average_v_metrics(tmp)
        stack_lst = append_lst(metrics, stack_lst)

    for name, col in zip(colnames, stack_lst):
        impacc[name] = col

    if save_impacc:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        impacc.to_csv(outdir + save_name, sep='\t', index=False, header=True)
    return impacc

def calculate_weighted_average(ary, weights):
    ary = np.array(ary)
    weights = np.array(weights)
    if ary.size == 0 or weights.size == 0 or weights.sum() == 0:
        return -9
    else:
        num = weights.sum()
        avg = (ary*weights).sum()/num
    return avg


def average_impacc_by_chr(impacc_lst,
                          MAF_ary=MAF_ARY,
                          colnames=IMPACC_H_COLS,
                          save_file=False,
                          outdir=None,
                          save_name=None):
    impacc = pd.DataFrame({'AF': MAF_ary})
    for i in range(MAF_ary.size):
        tmp_lst = [df[df['AF'] == MAF_ary[i]] for df in impacc_lst]
        merge_df = pd.concat(tmp_lst)
        metrics = [merge_df[colnames[0]].sum()]
        for j in range(len(colnames)):
            if j % 3 == 1:
                triplet = merge_df[colnames[j:j + 3]]
                triplet = triplet[triplet[colnames[j]] != -9]
                if triplet.shape[0] == 0:
                    metrics = metrics + [-9, 0, 0]
                elif triplet.shape[0] == 1:
                    metrics = metrics + list(triplet.iloc[0, :].values)
                else:
                    metrics = metrics + [
                        calculate_weighted_average(triplet.iloc[:, 0].values,
                                                   triplet.iloc[:, 2]),
                        triplet.iloc[:, 1].sum(), triplet.iloc[:, 2].sum()
                    ]
        if i == 0:
            res_ary = metrics
        else:
            res_ary = np.vstack([res_ary, metrics])
    res_ary = res_ary.T
    for name, col in zip(colnames, res_ary):
        impacc[name] = col

    if save_file:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        impacc.to_csv(outdir + save_name, sep='\t', index=False, header=True)

    return impacc


def calculate_imputation_summary_metrics(df,
                                         threshold,
                                         to_average=IMPACC_METRICS,
                                         decimal=3):
    df = df[df['AF'] > threshold]
    res_ary = []
    for i in to_average:
        res_ary.append(
            np.round(calculate_weighted_average(df[i], df[i + '_AC']),
                     decimals=decimal))
    return tuple(res_ary)


def calculate_imputation_sumstats(
        imp_dir,
        thresholds=[0.01, 0.05],
        axis='v',
        subset=False,
        chromosomes=CHROMOSOMES_ALL,
        case_controls=CASE_CONTROLS,
        ethnicities=ETHNICITIES,
        save_file=False,
        save_name="summary_metrics.tsv"):
    cols = [
        'Comparison', 'Subset', 'Threshold', 'NRC', 'r2', 'ccd_homref',
        'ccd_het', 'ccd_homalt'
    ]
    sumstats = pd.DataFrame(columns=cols)
    file_lst = [
        imp_dir + "impacc/all_samples/by_sample/chr" + i + "." + axis +
        ".impacc.tsv" for i in chromosomes
    ]
    dfs = [pd.read_csv(i, sep='\t') for i in file_lst]
    df = average_impacc_by_chr(dfs)
    analysis_name = imp_dir.split("/")[-2]

    for t in thresholds:
        res = [analysis_name, 'all'] + [t] + list(
            calculate_imputation_summary_metrics(df, t))
        sumstats.loc[len(sumstats)] = res

    if subset:
        for c in case_controls:
            file_lst = [
                imp_dir + "impacc/by_cc/by_sample/" + c + ".chr" + i + "." +
                axis + ".impacc.tsv" for i in chromosomes
            ]
            dfs = [pd.read_csv(i, sep='\t') for i in file_lst]
            df = average_impacc_by_chr(dfs)
            for t in thresholds:
                res = [analysis_name, c] + [t] + list(
                    calculate_imputation_summary_metrics(df, t))
                sumstats.loc[len(sumstats)] = res

        for e in ethnicities:
            file_lst = [
                imp_dir + "impacc/by_eth/by_sample/" + e + ".chr" + i + "." +
                axis + ".impacc.tsv" for i in chromosomes
            ]
            dfs = [pd.read_csv(i, sep='\t') for i in file_lst]
            df = average_impacc_by_chr(dfs)
            for t in thresholds:
                res = [analysis_name, e] + [t] + list(
                    calculate_imputation_summary_metrics(df, t))
                sumstats.loc[len(sumstats)] = res
    if save_file:
        sumstats.to_csv(imp_dir + save_name,
                        index=False,
                        header=True,
                        sep='\t')
    return sumstats

def calculate_shannon_entropy(str_lst):
    counts = Counter(str_lst)
    total_count = sum(counts.values())
    proportions = np.array(list(counts.values())) / total_count
    shannon_index = -np.sum(proportions * np.log(proportions))
    return shannon_index

def calculate_hla_concordance(df):
    ccd_one = df['One field match'].sum()/(df['Two field total'].sum())
    ccd_two = df['Two field match'].sum()/(df['Two field total'].sum())
    return [ccd_one, ccd_two]

def calculate_hla_concordance_by_type(ccd_dict, verbose = False):
    res_dict = {}
    for l in HLA_GENES:
        df = ccd_dict[l]
        df['Accuracy'] = 0
        df['Sum'] = 0
        df['Match'] = 0
        for a in df.index:
            r = df.loc[a]            
            index_set = set(a.split('/'))
            match = 0
            for column in df.columns:
                column_set = set(column.split('/'))
                if index_set & column_set:  # Check if there is an intersection
                    match += df.loc[a, column]

            total = r.sum()
            df.loc[a, 'Accuracy'] = match/total
            df.loc[a, 'Sum'] = total
            df.loc[a, 'Match'] = match
            
        if not verbose:
            res_dict[l] = df[['Accuracy', 'Sum', 'Match']]
        else:
            res_dict[l] = df
    return res_dict

def generate_hla_imputation_report(df, source, loci = HLA_GENES):
    report = pd.DataFrame(columns = ['Locus', 'Concordance', 'Resolution', 'Source'])
    for l in loci:
        tmp = df[df['Locus'] == l]
        ccd = calculate_hla_concordance(tmp)
        report.loc[len(report)] = [l, ccd[0], 'One field', source]
        report.loc[len(report)] = [l, ccd[1], 'Two field', source]
    return report

def calculate_hla_entropy(hla_alleles_df):
    entropy = []
    n_allele = []
    n_ambiguous = []
    for i in HLA_GENES:
        tmp = hla_alleles_df[hla_alleles_df['Locus'] == i]
        n_allele.append(len(tmp['Allele'].unique()))
        entropy.append(calculate_shannon_entropy(tmp['Allele'].values))
        n_ambiguous.append(len(tmp[tmp['Allele'].str.contains('/')]))

    summary = pd.DataFrame({'Gene': HLA_GENES, 'Shannon Entropy': entropy, 
                            'Number of distinct alleles': n_allele,
                           'Number of ambiguous type from SBT': n_ambiguous})
    return summary

def calculate_hla_imputation_accuracy(indir, hla, label, ct = 0, exclude_alleles = None, combined = 'combined', mode = 'old', recode_two_field = False, retain = 'fv'):
    if indir[-1] == '/':
        source = 'lc'
    else:
        source = 'chip'
        
    if source == 'lc':
        imputed = read_hla_lc_imputation_results(indir, combined, mode = mode, recode_two_field = recode_two_field, retain = retain)
    elif source == 'chip':
        imputed = read_hla_chip_imputation_results(indir, recode_two_field, retain)
    else:
        print('Invalid source input.')
        return None
    
    original_N = []
    for g in HLA_GENES:
        n = imputed[imputed['Locus'] == g].shape[0]
        original_N = original_N + [n, n]
    
    imputed = imputed[imputed['prob'] >= ct]
    hla = hla.iloc[imputed.index,:]
    
    cr = []
    for i, g in enumerate(HLA_GENES):
        n = imputed[imputed['Locus'] == g].shape[0]
        cr = cr + [n/original_N[2*i], n/original_N[2*i]]
    
    compared = compare_hla_types(hla, imputed, exclude_alleles)
    hla_report = generate_hla_imputation_report(compared, label)
    hla_report['CR'] = cr
    hla_report['CT'] = ct
    
    hla_report = hla_report[['Source', 'Locus', 'Resolution', 'CT', 'CR', 'Concordance']]
    return hla_report