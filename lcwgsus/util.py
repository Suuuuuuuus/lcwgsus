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

__all__ = ["visualise_single_variant"]

def visualise_single_variant(c, pos, vcf_lst, source_lst, labels_lst, vcf_cols = [
    'chr', 'pos', 'ID', 'ref', 'alt', 'QUAL', 'FILTER', 'INFO', 'FORMAT'
], mini = False, save_fig = False, outdir = None, save_name = None):
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

    plot_violin(res, x = 'GT', y = 'GP', hue = 'labels', title = site, save_fig = save_fig, outdir = outdir, save_name = save_name)
    return None